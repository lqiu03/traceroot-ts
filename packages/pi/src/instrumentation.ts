/**
 * Patch layer: wraps AgentSession.prototype.prompt once per module namespace
 * (wrap-once guarded, matching packages/traceroot/src/claude-agent-sdk.ts's
 * verified real idiom) and builds the entire span tree from a single
 * session.subscribe() listener registered once per session instance.
 *
 * No AsyncLocalStorage: an AgentSession processes one run at a time, so a
 * plain per-session state object (closed over per subscribe() call) is
 * sufficient to correlate events — matching the CLI extension's own
 * "explicit state, never ambient context" convention.
 *
 * Cleanup: instrumentPiCodingAgent() never unsubscribes its own listener.
 * This is verified, not assumed — read directly from the real, installed
 * @earendil-works/pi-coding-agent@0.80.6 (node_modules/.../dist/core/
 * agent-session.js, the exact file `AgentSession` is exported from in
 * dist/index.js): AgentSession.subscribe(listener) pushes the listener onto
 * one private `_eventListeners` array and returns an unsubscribe closure
 * that splices it back out of that same array; `_emit(event)` — the only
 * dispatcher, used for every AgentEvent — reads `this._eventListeners` fresh
 * on every call rather than closing over a snapshot. AgentSession.dispose()
 * reassigns `this._eventListeners = []` (after aborting retry/compaction/
 * branch-summary/bash, aborting the agent, and disconnecting from the
 * underlying Agent's own subscription). Because `_emit()` always re-reads
 * `this._eventListeners` off `this` rather than an array reference captured
 * earlier, dispose()'s reassignment means every subsequent `_emit()` call —
 * for our listener and every other caller's — iterates zero listeners.
 * `grep -rn "_eventListeners"` across the package's whole dist/ tree confirms
 * that field is private to AgentSession and never aliased or copied
 * elsewhere, so this holds unconditionally. There is no separate "session
 * ended" event to hook for cleanup, and none is needed.
 */
import { context, ROOT_CONTEXT, trace } from '@opentelemetry/api';
import type { Context, Span, Tracer } from '@opentelemetry/api';
import { resolveConfig } from './config';
import type { PiInstrumentationConfig, ResolvedPiInstrumentationConfig } from './config';
import { createTracing } from './provider';
import {
  closeDanglingSpan,
  closeLlmSpan,
  closeRootSpan,
  closeToolSpan,
  openLlmSpan,
  openRootSpan,
  openToolSpan,
} from './spans';
import type { AgentEvent, AgentSessionInstance, PiCodingAgentModule } from './types';

const WRAPPED = Symbol.for('traceroot.pi_coding_agent.wrapped');

// A queued prompt() call's text, boxed in its own object rather than stored
// as a raw string: proto.prompt's rejection handler below needs to remove
// EXACTLY the entry it pushed (by reference identity) if that specific call
// never reaches agent_start, without accidentally removing a different,
// still-pending queue entry that happens to hold an equal string value.
interface QueuedPrompt {
  text: string;
}

interface SessionSpanState {
  rootSpan: Span | undefined;
  rootCtx: Context | undefined;
  llmSpan: Span | undefined;
  llmCtx: Context | undefined;
  toolSpans: Map<string, Span>;
  // The input text this run's root span was opened with (dequeued from
  // pendingInput at agent_start) — kept around so agent_end can hand it
  // back to the queue if this run turns out to be retried.
  pendingInputText: string | undefined;
}

export function instrumentPiCodingAgent(sdk: unknown, config?: PiInstrumentationConfig): unknown {
  const mod = sdk as PiCodingAgentModule;

  const resolved = resolveConfig(config);
  if (!resolved.apiKey) {
    console.warn(
      '[traceroot-pi] TRACEROOT_API_KEY (or config.apiKey) is not set — instrumentation disabled.',
    );
    return sdk;
  }

  // The wrap-once guard is stamped on AgentSession.prototype, never on `mod`
  // itself. When `sdk` is obtained via `import * as pi from "..."` (the
  // README's own documented usage, and the only way to consume an ESM-only
  // package like @earendil-works/pi-coding-agent from Node), `mod` is an ES
  // module namespace exotic object — the spec makes those permanently
  // non-extensible, so `Object.defineProperty(mod, ...)` always throws
  // TypeError. AgentSession.prototype is an ordinary, extensible object, and
  // it's the thing actually being patched below, so it's also the correct
  // place to record that the patch already happened.
  const proto = mod?.AgentSession?.prototype as
    | (AgentSessionInstance & { [WRAPPED]?: boolean })
    | undefined;
  if (typeof proto?.prompt !== 'function' || typeof proto?.subscribe !== 'function') {
    console.warn(
      '[traceroot-pi] AgentSession.prototype.prompt/subscribe not found — instrumentation disabled.',
    );
    return sdk;
  }

  if (proto[WRAPPED]) {
    console.warn(
      "[traceroot-pi] instrumentPiCodingAgent() was already called for this sdk — this call's config is ignored.",
    );
    return sdk;
  }
  Object.defineProperty(proto, WRAPPED, { value: true, enumerable: false });

  const { tracer, forceFlush } = createTracing(resolved);
  // The BatchSpanProcessor holds spans for up to a couple seconds before
  // exporting — without this, a short-lived script (the common case for a
  // one-shot Pi prompt) would exit before anything is ever sent, matching
  // packages/traceroot/src/traceroot.ts's own process.once('beforeExit', ...)
  // auto-flush convention. Deliberately forceFlush(), not shutdown(): this
  // handler only fires once (the first time the event loop drains), but a
  // long-lived host process can keep running new Pi sessions afterward —
  // shutdown() would permanently disable all further export from that point
  // on, while forceFlush() only flushes what's pending and leaves the
  // pipeline usable for every subsequent session in the same process.
  process.once('beforeExit', () => {
    void forceFlush();
  });

  // Per-session FIFO queue, not a single slot: a second prompt() call can
  // fire before the first run's agent_start event has arrived (overlapping
  // runs on the same session), and each run's agent_start must claim the
  // text from the prompt() call that actually triggered it, in order —
  // never let a later prompt() call's text clobber an earlier one's.
  const pendingInput = new WeakMap<AgentSessionInstance, QueuedPrompt[]>();
  const subscribedSessions = new WeakSet<AgentSessionInstance>();

  const originalPrompt = proto.prompt;
  proto.prompt = function (this: AgentSessionInstance, text, options) {
    let entry: QueuedPrompt | undefined;
    if (typeof text === 'string') {
      entry = { text };
      const queue = pendingInput.get(this);
      if (queue) queue.push(entry);
      else pendingInput.set(this, [entry]);
    }
    if (!subscribedSessions.has(this)) {
      subscribedSessions.add(this);
      attachSpanListener(this, tracer, resolved, pendingInput);
    }
    // A prompt() call that never reaches agent_start (a synchronous throw,
    // or its returned Promise rejecting — e.g. a validation failure inside
    // Pi's own prompt() before the agent loop starts) must not leave its
    // text sitting in the FIFO queue: agent_start will never fire to shift()
    // it back out, so it would otherwise silently become the wrong (stale)
    // input text attached to whatever LATER, genuinely-successful prompt()
    // call on this same session shifts it out instead — and every prompt()
    // after that would be off by one, permanently. Remove by reference
    // identity, not by string match, so a different still-pending queue
    // entry that happens to hold an equal string is never removed instead.
    const removeIfStillQueued = (): void => {
      if (!entry) return;
      const queue = pendingInput.get(this);
      if (!queue) return;
      const idx = queue.indexOf(entry);
      if (idx !== -1) queue.splice(idx, 1);
    };
    let result: Promise<void>;
    try {
      result = originalPrompt.call(this, text, options);
    } catch (err) {
      removeIfStillQueued();
      throw err;
    }
    result.catch(removeIfStillQueued);
    return result;
  };

  return sdk;
}

function attachSpanListener(
  session: AgentSessionInstance,
  tracer: Tracer,
  config: ResolvedPiInstrumentationConfig,
  pendingInput: WeakMap<AgentSessionInstance, QueuedPrompt[]>,
): void {
  const state: SessionSpanState = {
    rootSpan: undefined,
    rootCtx: undefined,
    llmSpan: undefined,
    llmCtx: undefined,
    toolSpans: new Map(),
    pendingInputText: undefined,
  };

  session.subscribe((event: AgentEvent) => {
    try {
      handleEvent(event, session, tracer, config, pendingInput, state);
    } catch (err) {
      // A handler throw must never reach Pi's event dispatcher — that would
      // crash or destabilize the host app's agent loop over a tracing bug.
      console.warn(
        '[traceroot-pi] instrumentation handler failed (a span may be missing or incomplete):',
        err,
      );
    }
  });
}

function handleEvent(
  event: AgentEvent,
  session: AgentSessionInstance,
  tracer: Tracer,
  config: ResolvedPiInstrumentationConfig,
  pendingInput: WeakMap<AgentSessionInstance, QueuedPrompt[]>,
  state: SessionSpanState,
): void {
  switch (event.type) {
    case 'agent_start': {
      // A fresh agent_start while a previous run's root span is still open
      // means that run's agent_end never fired (e.g. the loop crashed and
      // restarted rather than cleanly finishing) — force-close everything
      // left open from it instead of silently leaking those spans, and
      // instead of leaving their now-stale context around to be picked up
      // as the parent of this new run's spans.
      //
      // The tool-span and LLM-span sweeps are each gated on their own state
      // independently of state.rootSpan: a stray tool_execution_start OR a
      // stray message_start (with no matching message_end) can arrive AFTER
      // a prior run's agent_end already cleared rootSpan (e.g. an async tool
      // callback resolving late, or a straggler stream event) — leaving an
      // orphaned entry with no rootSpan to gate on. Without gating each one
      // independently, that orphan would never be swept here and would
      // either stay open forever (tool span) or have its reference silently
      // overwritten below without ever calling .end() on it (LLM span) —
      // and a span that never has .end() called on it is never
      // recorded/exported at all, not merely "left open".
      if (state.toolSpans.size > 0) {
        for (const span of state.toolSpans.values()) closeDanglingSpan(span);
        state.toolSpans.clear();
      }
      if (state.llmSpan) {
        closeDanglingSpan(state.llmSpan);
      }
      if (state.rootSpan) {
        closeDanglingSpan(state.rootSpan);
      }
      const parentCtx = context.active();
      // pendingInput is a per-session FIFO queue (see instrumentPiCodingAgent's
      // own comment) — dequeue the oldest queued text so overlapping prompt()
      // calls each hand their text to the correct run's agent_start, in order.
      const inputText = pendingInput.get(session)?.shift()?.text;
      state.pendingInputText = inputText;
      state.rootSpan = openRootSpan(tracer, parentCtx, {
        text: inputText,
        sessionId: session.sessionId,
        captureContent: config.captureContent,
      });
      state.rootCtx = trace.setSpan(parentCtx, state.rootSpan);
      state.llmSpan = undefined;
      state.llmCtx = undefined;
      break;
    }
    case 'message_start': {
      if (event.message.role !== 'assistant') return;
      // A second message_start with no intervening message_end/turn_end
      // means the previous LLM span was abandoned (e.g. a stream error
      // skipped straight to a new message) — force-close it first so it
      // still exports instead of having its state.llmSpan slot silently
      // overwritten below, mirroring tool_execution_start's identical
      // duplicate-open handling further down in this switch.
      if (state.llmSpan) closeDanglingSpan(state.llmSpan);
      // Falls back to ROOT_CONTEXT, never context.active(): the latter is
      // whatever the host process's own OTel context manager happens to have
      // ambiently active right now, which has nothing to do with this Pi
      // session. Parenting under it would risk cross-trace contamination in
      // a multi-tenant host — ROOT_CONTEXT instead starts a fresh,
      // standalone trace for a stray/parentless event.
      const parentCtx = state.rootCtx ?? ROOT_CONTEXT;
      state.llmSpan = openLlmSpan(tracer, parentCtx, event.message);
      state.llmCtx = trace.setSpan(parentCtx, state.llmSpan);
      break;
    }
    case 'message_end': {
      if (event.message.role !== 'assistant') return;
      if (state.llmSpan) closeLlmSpan(state.llmSpan, event.message, config.captureContent);
      state.llmSpan = undefined;
      // llmCtx stays alive on purpose: tool_execution_start/end for this
      // turn's tool calls fire after message_end but before turn_end, and
      // must still parent under this (now-ended) LLM span rather than
      // falling back to the root span.
      break;
    }
    case 'turn_end': {
      // Normally message_end already closed and cleared state.llmSpan before
      // turn_end fires. If it didn't (e.g. a stream error cut the turn short
      // so message_end never arrived), the LLM span would otherwise stay
      // open indefinitely until something else happens to overwrite the
      // state reference — force-close it here instead of leaking it.
      if (state.llmSpan) closeDanglingSpan(state.llmSpan);
      state.llmSpan = undefined;
      state.llmCtx = undefined;
      // Tool calls are turn-scoped: any tool span still open when the turn
      // ends (its tool_execution_end never arrived) must be force-closed
      // here too, instead of staying open until agent_end — mirroring
      // agent_end's own identical toolSpans sweep.
      for (const span of state.toolSpans.values()) closeDanglingSpan(span);
      state.toolSpans.clear();
      break;
    }
    case 'tool_execution_start': {
      // A tool_execution_start for a toolCallId that is already open (no
      // intervening tool_execution_end) would otherwise have its Map slot
      // silently overwritten below — since a span only exports once end()
      // is called, the abandoned first span would be lost forever rather
      // than merely "left open". Force-close it first so it still surfaces,
      // matching the same never-leak-silently philosophy applied to dangling
      // spans everywhere else in this handler (agent_start, turn_end, agent_end).
      const existing = state.toolSpans.get(event.toolCallId);
      if (existing) closeDanglingSpan(existing);
      // See message_start's comment: fall back to ROOT_CONTEXT, never the
      // ambient context.active(), to avoid parenting a stray event under
      // whatever unrelated span the host process happens to have active.
      const parentCtx = state.llmCtx ?? state.rootCtx ?? ROOT_CONTEXT;
      const span = openToolSpan(
        tracer,
        parentCtx,
        event.toolCallId,
        event.toolName,
        event.args,
        config.captureToolIo,
      );
      state.toolSpans.set(event.toolCallId, span);
      break;
    }
    case 'tool_execution_end': {
      const span = state.toolSpans.get(event.toolCallId);
      if (span) closeToolSpan(span, event.result, event.isError, config.captureToolIo);
      state.toolSpans.delete(event.toolCallId);
      break;
    }
    case 'agent_end': {
      for (const span of state.toolSpans.values()) closeDanglingSpan(span);
      state.toolSpans.clear();
      closeDanglingSpan(state.llmSpan);
      state.llmSpan = undefined;
      state.llmCtx = undefined;
      if (state.rootSpan)
        closeRootSpan(state.rootSpan, event.messages, event.willRetry, config.captureContent);
      state.rootSpan = undefined;
      state.rootCtx = undefined;
      // A retry re-enters the agent loop with a fresh agent_start but no new
      // prompt() call, so it must be able to read the SAME input text this
      // attempt used. agent_start already dequeued (shifted) that text out
      // of pendingInput, so hand it back to the front of the queue here
      // instead of discarding it — the opposite of an unconditional delete,
      // which would lose it forever on a retry. On a non-retry, do nothing:
      // the text was already correctly consumed by agent_start's shift(),
      // and anything still queued behind it belongs to separate,
      // not-yet-started prompt() calls that must be left untouched.
      if (event.willRetry && state.pendingInputText !== undefined) {
        const queue = pendingInput.get(session);
        const retryEntry: QueuedPrompt = { text: state.pendingInputText };
        if (queue) queue.unshift(retryEntry);
        else pendingInput.set(session, [retryEntry]);
      }
      state.pendingInputText = undefined;
      break;
    }
    default:
      break;
  }
}
