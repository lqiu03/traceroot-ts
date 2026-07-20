/**
 * Patch layer: wraps AgentSession.prototype.prompt once per module namespace
 * (wrap-once guarded, matching claude-agent-sdk.ts's idiom) and builds the
 * entire span tree from a single session.subscribe() listener registered
 * once per session instance.
 *
 * No AsyncLocalStorage: an AgentSession processes one run at a time, so a
 * plain per-session state object (closed over per subscribe() call) is
 * sufficient to correlate events.
 *
 * ── One trace per prompt() call ───────────────────────────────────────────
 * The root AGENT span is anchored on the wrapped prompt() call's own promise
 * window — opened at entry, closed when that promise settles — mirroring
 * claude-agent-sdk.ts's one-root-per-query() idiom. This is deliberately NOT
 * anchored on agent_start/agent_end: verified against the real, installed
 * @earendil-works/pi-coding-agent@0.80.6 (dist/core/agent-session.js),
 * AgentSession.prompt() awaits _runAgentPrompt(), whose
 * `while (await this._handlePostAgentRun()) { await this.agent.continue(); }`
 * loop performs EVERY retry/compaction/follow-up continuation inside the
 * awaited promise, with each attempt's own agent_end handler completing
 * before prompt() resolves. So a single root spanning the whole prompt()
 * promise captures every continuation attempt as a child LLM/tool span
 * instead of pi's internal retry/compaction/follow-up control flow leaking
 * into trace *structure* as multiple sibling traces. See proto.prompt below
 * for the open/close mechanics and the DECIDED boundary policies.
 *
 * Cleanup: instrumentPiCodingAgent() never unsubscribes its own listener.
 * Verified against @earendil-works/pi-coding-agent@0.80.6
 * (dist/core/agent-session.js): AgentSession.subscribe(listener) pushes onto
 * a private `_eventListeners` array; `_emit(event)` reads that array fresh on
 * every call rather than a captured snapshot. AgentSession.dispose()
 * reassigns `this._eventListeners = []`, so every subsequent `_emit()` call —
 * ours and every other caller's — iterates zero listeners. There is no
 * separate "session ended" event to hook for cleanup, and none is needed.
 */
import { context, ROOT_CONTEXT, SpanStatusCode, trace } from '@opentelemetry/api';
import type { Context, Span } from '@opentelemetry/api';
import { resolveConfig, TRACER_NAME } from './config';
import type { PiInstrumentationConfig, ResolvedPiInstrumentationConfig } from './config';
import { SDK_VERSION } from '../processor';
import {
  closeDanglingSpan,
  closeLlmSpan,
  closeToolSpan,
  finalizeRootSpan,
  openLlmSpan,
  openRootSpan,
  openToolSpan,
  stampRootOutput,
} from './spans';
import { createReresolvingTracer } from '../reresolving-tracer';
import type { SpanFactory } from '../reresolving-tracer';
import type { AgentEvent, AgentSessionInstance, PiCodingAgentModule } from './types';

// Symbol.for(), not a module-scoped Symbol(): stamped onto
// AgentSession.prototype, which two independently-loaded copies of this
// package (e.g. a monorepo hoisting/dedup failure) could both reach. A
// process-wide registry key means both copies see the same symbol, so a
// second instrumentPiCodingAgent() call is rejected with a console.warn
// instead of silently double-instrumenting every session forever. Do not
// flip this back to a bare Symbol() without first building multiplexing
// support for two configs sharing one patched prototype.
const WRAPPED = Symbol.for('traceroot.pi_coding_agent.wrapped');

interface SessionSpanState {
  rootSpan: Span | undefined;
  rootCtx: Context | undefined;
  llmSpan: Span | undefined;
  llmCtx: Context | undefined;
  toolSpans: Map<string, Span>;
  // Observable retry-attempt count for the CURRENT prompt() window, reset to
  // 0 when proto.prompt opens a fresh root and incremented once per
  // agent_end{willRetry:true}. Stamped onto the root as
  // traceroot.pi.retry_count when the window's prompt() promise settles (see
  // finalizeRootSpan in spans.ts).
  retryCount: number;
}

// Force-closes every span left open by an abandoned run: every open tool
// span, then the LLM span, and — only when explicitly requested — the root
// span. Shared by agent_start, turn_end, agent_end, proto.prompt (the
// OVERLAP SAFETY sweep), and dispose(). Safe to call unconditionally: every
// step is already a no-op on undefined/empty, so callers never need their own
// guard before calling this, and one span's close failure never aborts the
// rest of the sweep.
function sweepDanglingSpans(
  state: SessionSpanState,
  options: { includeRoot?: boolean } = {},
): void {
  for (const [, span] of state.toolSpans) {
    closeDanglingSpan(span);
  }
  state.toolSpans.clear();
  closeDanglingSpan(state.llmSpan);
  state.llmSpan = undefined;
  if (options.includeRoot) {
    closeDanglingSpan(state.rootSpan);
    state.rootSpan = undefined;
  }
}

// Shared by proto.prompt/proto.steer/proto.followUp below: each of those
// three independently-callable entry points can be a session's first
// interaction, so each must guarantee session.subscribe() has been called
// exactly once for that session, and a SessionSpanState created for it,
// before delegating to the real method. proto.prompt relies on this —
// `sessionSpanState.get(this)` immediately after calling this is guaranteed
// non-undefined.
function ensureSubscribed(
  session: AgentSessionInstance,
  tracer: SpanFactory,
  config: ResolvedPiInstrumentationConfig,
  subscribedSessions: WeakSet<AgentSessionInstance>,
  sessionSpanState: WeakMap<AgentSessionInstance, SessionSpanState>,
): void {
  if (subscribedSessions.has(session)) return;
  subscribedSessions.add(session);
  attachSpanListener(session, tracer, config, sessionSpanState);
}

export function instrumentPiCodingAgent(sdk: unknown, config?: PiInstrumentationConfig): unknown {
  const mod = sdk as PiCodingAgentModule;

  const resolved = resolveConfig(config);

  // The wrap-once guard is stamped on AgentSession.prototype, never on `mod`
  // itself: when `sdk` is obtained via `import * as pi from "..."` (required
  // for an ESM-only package like @earendil-works/pi-coding-agent), `mod` is
  // an ES module namespace exotic object — the spec makes those permanently
  // non-extensible, so `Object.defineProperty(mod, ...)` always throws
  // TypeError. AgentSession.prototype is an ordinary, extensible object, and
  // it's the thing actually being patched below.
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

  // core's TraceRoot.initialize() guarantees a real, globally registered OTel
  // TracerProvider is already in place before this function runs, so the
  // tracer below can re-resolve that global provider on every span-open
  // (see createReresolvingTracer).
  const tracer = createReresolvingTracer(TRACER_NAME, SDK_VERSION);

  const subscribedSessions = new WeakSet<AgentSessionInstance>();
  // attachSpanListener() below creates one SessionSpanState per session and
  // closes over it for its own subscribe() callback, but
  // AgentSession.prototype.dispose (patched once, below) has no closure over
  // any particular session — it needs this out-of-band map to reach
  // whichever session's SessionSpanState (if any) it was called on.
  const sessionSpanState = new WeakMap<AgentSessionInstance, SessionSpanState>();

  // Install every prototype method patch, then — only then — stamp the
  // wrap-once guard, so a mid-setup failure can't leave the prototype
  // permanently marked "wrapped" while never actually patched. Each patch
  // records an undo on `rollback`, restoring the prototype exactly to how it
  // was found before rethrowing a clear install error.
  const rollback: Array<() => void> = [];
  try {
    // Process-exit flushing is core's responsibility (see traceroot.ts's own
    // process.once('beforeExit', ...)) — no flush hook to register here.

    const originalPrompt = proto.prompt;
    proto.prompt = function (this: AgentSessionInstance, text, options) {
      ensureSubscribed(this, tracer, resolved, subscribedSessions, sessionSpanState);

      // MID-STREAM STEER: verified against @earendil-works/pi-coding-agent@0.80.6
      // (dist/core/agent-session.js) — when a run is ALREADY streaming and the
      // caller passes options.streamingBehavior, prompt() injects into the
      // CURRENTLY-running run's queue and returns early rather than starting a
      // new run. Must be detected and delegated straight through BEFORE any
      // root management below, or opening a fresh root here (or the OVERLAP
      // SAFETY sweep further down) would force-close the ACTIVE run's
      // still-open root mid-flight. The only early-return path that skips
      // root management entirely — the "/command" and 'input'-hook returns
      // below are non-streaming, so their trivial root (policy 2) is
      // unaffected.
      const isQueueOnlySteer = this.isStreaming === true && !!options?.streamingBehavior;
      if (isQueueOnlySteer) {
        return originalPrompt.call(this, text, options);
      }

      // Guaranteed non-undefined: ensureSubscribed() either found an existing
      // entry or attachSpanListener() just created one for `this` session.
      const state = sessionSpanState.get(this) as SessionSpanState;

      // OVERLAP SAFETY: a previous prompt() window's root is still open here.
      // Not the mid-stream steer case above (already detected and returned
      // before this point) — this is a last-resort safety net for a
      // genuinely-new, non-streaming prompt() call that races a still-open
      // prior window. Force-close that stale window (root included) rather
      // than silently overwriting state.rootSpan and leaking it unended
      // forever (a span that never has .end() called on it is never exported).
      if (state.rootSpan) {
        sweepDanglingSpans(state, { includeRoot: true });
        // sweepDanglingSpans() only clears state.llmSpan, not the parenting
        // context that goes with it — left stale, a tool_execution_start
        // racing in right after this sweep would wrongly parent under the
        // now-force-closed LLM span instead of falling back to the fresh
        // root about to be opened below.
        state.llmCtx = undefined;
      }

      // context.active(), matching claude-agent-sdk.ts's own query() span —
      // lets a host that wraps prompt() in its own span nest this trace
      // under it.
      const parentCtx = context.active();
      const rootSpan = openRootSpan(tracer, parentCtx, {
        text: typeof text === 'string' ? text : undefined,
        sessionId: this.sessionId,
        captureContent: resolved.captureContent,
      });
      state.rootSpan = rootSpan;
      state.rootCtx = trace.setSpan(parentCtx, rootSpan);
      state.retryCount = 0;

      // Ends and clears THIS window's root exactly once, guarded by identity
      // (state.rootSpan === rootSpan) so a mid-run dispose() or a later
      // overlapping prompt() call's own OVERLAP SAFETY sweep — either of
      // which may already have force-closed and cleared it — is never
      // double-ended or clobbers a DIFFERENT, newer window's root.
      const finalize = (
        status: { code: SpanStatusCode; message?: string },
        error?: unknown,
      ): void => {
        if (state.rootSpan !== rootSpan) return;
        state.rootSpan = undefined;
        state.rootCtx = undefined;
        // claude-agent-sdk.ts parity (see its own endInFlight()): a
        // rejection (or an early settle racing a crashed attempt) can leave
        // a tool/LLM span from the in-flight attempt never closed by its own
        // normal event. Force-close those BEFORE ending the root so they
        // still export instead of being silently dropped forever. includeRoot
        // is omitted: finalizeRootSpan below is THIS window's own root close.
        sweepDanglingSpans(state);
        state.llmCtx = undefined;
        finalizeRootSpan(rootSpan, state.retryCount, status, error);
      };

      // If this call throws synchronously (before ever returning a promise —
      // e.g. a validation failure inside pi's own prompt() before the agent
      // loop starts), it never reached agent_start: finalize the root as
      // ERROR right here and rethrow, matching claude-agent-sdk.ts's own
      // sync-throw handling in wrapQuery.
      let result: Promise<void>;
      try {
        result = originalPrompt.call(this, text, options);
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        finalize({ code: SpanStatusCode.ERROR, message }, err);
        throw err;
      }
      // A SEPARATE .then chain (not a replacement of `result`) so the host
      // still sees the original resolution/rejection unmodified. Attaching
      // this handler is enough to mark `result`'s rejection as handled for
      // Node's unhandledRejection detection without re-throwing.
      //
      // Four boundary policies:
      //  1. A bypass run (e.g. sendCustomMessage) never opens a root — see
      //     agent_start below.
      //  2. A synchronous early-return prompt() call (handled "/command", an
      //     'input' hook) resolves with no children — a trivial, OK root.
      //  3. A rejection (or the sync-throw path above) finalizes the root
      //     ERROR with the rejection recorded as an exception.
      //  4. A queue-only steer/followUp call returns before this point
      //     (isQueueOnlySteer above) and never reaches finalize() at all.
      result.then(
        () => finalize({ code: SpanStatusCode.OK }),
        (err: unknown) => {
          const message = err instanceof Error ? err.message : String(err);
          finalize({ code: SpanStatusCode.ERROR, message }, err);
        },
      );
      return result;
    };
    rollback.push(() => {
      proto.prompt = originalPrompt;
    });

    // steer() and followUp() are standalone public SDK entry points, not
    // wrappers of prompt() — a host can call them directly without ever
    // calling prompt() on that session first. Without this patch, a session
    // whose first interaction was steer()/followUp() would never get
    // session.subscribe() called, so every AgentEvent would silently produce
    // zero spans. Neither opens a root span itself (only proto.prompt does),
    // so a run triggered purely by steer()/followUp() is a BYPASS run under
    // the rootless boundary policy: see agent_start below. Patched
    // defensively (only when present), like dispose() below.
    if (typeof proto.steer === 'function') {
      const originalSteer = proto.steer;
      proto.steer = function (this: AgentSessionInstance, text: string, images?: unknown[]) {
        ensureSubscribed(this, tracer, resolved, subscribedSessions, sessionSpanState);
        return originalSteer.call(this, text, images);
      };
      rollback.push(() => {
        proto.steer = originalSteer;
      });
    }
    if (typeof proto.followUp === 'function') {
      const originalFollowUp = proto.followUp;
      proto.followUp = function (this: AgentSessionInstance, text: string, images?: unknown[]) {
        ensureSubscribed(this, tracer, resolved, subscribedSessions, sessionSpanState);
        return originalFollowUp.call(this, text, images);
      };
      rollback.push(() => {
        proto.followUp = originalFollowUp;
      });
    }

    // Patched once here (guarded by the same WRAPPED check/stamp above that
    // guards proto.prompt), never per-session: dispose() is a single shared
    // prototype method, same as prompt(). Only patched when dispose actually
    // exists as a function — defensive, so a minimal/partial double never
    // disables prompt instrumentation over a missing, unrelated method.
    if (typeof proto.dispose === 'function') {
      const originalDispose = proto.dispose;
      proto.dispose = function (this: AgentSessionInstance): void {
        // The real dispose() only reassigns the SDK's private
        // _eventListeners array — it does nothing to spans this session's
        // listener already opened. If a host calls dispose() mid-run,
        // force-close any still-open rootSpan/llmSpan/toolSpans here first,
        // exactly like agent_start's own sweep, then delegate to the real
        // dispose().
        const state = sessionSpanState.get(this);
        if (state) {
          try {
            sweepDanglingSpans(state, { includeRoot: true });
          } catch (err) {
            // A misbehaving span/exporter must never make dispose() throw —
            // the host still needs its session torn down even if this
            // best-effort tracing cleanup failed.
            console.warn(
              '[traceroot-pi] failed to force-close in-flight spans during dispose() (a span may leak):',
              err,
            );
          } finally {
            state.rootSpan = undefined;
            state.rootCtx = undefined;
            state.llmSpan = undefined;
            state.llmCtx = undefined;
            // Avoids re-sweeping already-ended spans if dispose() is ever
            // called twice on the same session.
            sessionSpanState.delete(this);
            // subscribedSessions gates the "already subscribed?" check in
            // proto.prompt/steer/followUp above. dispose() only clears
            // _eventListeners — it doesn't make the session instance
            // unusable, and nothing stops a host calling prompt()/steer()/
            // followUp() again after dispose(). Without this delete, a
            // reused session would find subscribedSessions.has(this) still
            // true and silently never re-subscribe.
            subscribedSessions.delete(this);
          }
        }
        return originalDispose.call(this);
      };
      rollback.push(() => {
        proto.dispose = originalDispose;
      });
    }

    // Setup fully succeeded: only now is it correct to mark this prototype
    // wrapped. This MUST be the last statement INSIDE the try: if
    // Object.defineProperty throws (e.g. a frozen/sealed prototype), that
    // failure has to unwind the method patches already applied, like any
    // other mid-setup failure. Left outside the try, the prototype would
    // stay patched but UNSTAMPED, so the next instrumentPiCodingAgent() call
    // would re-patch already-patched methods and emit duplicate spans
    // forever.
    Object.defineProperty(proto, WRAPPED, { value: true, enumerable: false });
  } catch (err) {
    // Undo every patch already applied, newest first, so a failed install
    // leaves AgentSession.prototype exactly as it was found rather than
    // half-patched — then surface the failure loudly instead of leaving a
    // silently-broken, "wrapped"-but-uninstrumented prototype behind.
    for (let i = rollback.length - 1; i >= 0; i--) {
      rollback[i]();
    }
    throw new Error('[traceroot-pi] failed to install instrumentation on AgentSession.prototype', {
      cause: err,
    });
  }

  return sdk;
}

function attachSpanListener(
  session: AgentSessionInstance,
  tracer: SpanFactory,
  config: ResolvedPiInstrumentationConfig,
  sessionSpanState: WeakMap<AgentSessionInstance, SessionSpanState>,
): void {
  const state: SessionSpanState = {
    rootSpan: undefined,
    rootCtx: undefined,
    llmSpan: undefined,
    llmCtx: undefined,
    toolSpans: new Map(),
    retryCount: 0,
  };
  // Reachable from AgentSession.prototype.dispose so a mid-run dispose() can
  // force-close whatever this session's listener callback below left open.
  sessionSpanState.set(session, state);

  session.subscribe((event: AgentEvent) => {
    try {
      handleEvent(event, tracer, config, state);
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
  tracer: SpanFactory,
  config: ResolvedPiInstrumentationConfig,
  state: SessionSpanState,
): void {
  switch (event.type) {
    case 'agent_start': {
      // Sweep dangling LLM/tool spans from a crashed prior ATTEMPT within
      // the same prompt() window (e.g. the loop restarted a retry/
      // compaction/follow-up continuation without its own agent_end ever
      // firing for the previous attempt). Deliberately WITHOUT includeRoot:
      // the root now belongs to the enclosing prompt() call's promise
      // window, not to any one attempt, so agent_start must never close it —
      // doing so would end the trace early and orphan every later
      // continuation attempt's spans onto a fresh, disconnected root.
      sweepDanglingSpans(state);
      state.llmSpan = undefined;
      state.llmCtx = undefined;
      // Boundary policy 1 (rootless bypass): if state.rootSpan is undefined
      // here, this run never went through the wrapped prompt() call at all
      // (e.g. sendCustomMessage({triggerTurn:true}), or a steer()/followUp()-
      // only session with no enclosing prompt()). Do NOT synthesize a root
      // for it — its LLM/tool children fall back to ROOT_CONTEXT below
      // and form a parentless mini-trace instead of a fabricated, input-less
      // AGENT span.
      break;
    }
    case 'message_start': {
      if (event.message.role !== 'assistant') return;
      // A second message_start with no intervening message_end/turn_end
      // means the previous LLM span was abandoned (e.g. a stream error
      // skipped straight to a new message) — force-close it first so it
      // still exports instead of having its state.llmSpan slot silently
      // overwritten below.
      closeDanglingSpan(state.llmSpan);
      // Falls back to ROOT_CONTEXT, never context.active(): the latter is
      // whatever the host process's own OTel context manager happens to have
      // ambiently active right now, unrelated to this Pi session. Parenting
      // under it would risk cross-trace contamination in a multi-tenant host.
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
      // turn_end fires. If it didn't (e.g. a stream error cut the turn short),
      // force-close it here instead of leaking it. Tool calls are
      // turn-scoped too: any tool span still open when the turn ends must be
      // force-closed here as well. Root span is deliberately left untouched —
      // turn_end isn't session end (nor even attempt end).
      sweepDanglingSpans(state);
      state.llmCtx = undefined;
      break;
    }
    case 'tool_execution_start': {
      // A tool_execution_start for a toolCallId that is already open (no
      // intervening tool_execution_end) would otherwise have its Map slot
      // silently overwritten below, losing the abandoned span rather than
      // merely leaving it open — force-close it first so it still surfaces.
      const existing = state.toolSpans.get(event.toolCallId);
      closeDanglingSpan(existing);
      // See message_start's comment: fall back to ROOT_CONTEXT, never the
      // ambient context.active().
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
      // Defensive cleanup for any tool/LLM span this attempt left dangling,
      // mirroring turn_end's identical sweep. The root span is NOT part of
      // this sweep: agent_end no longer owns closing the root at all (that
      // is proto.prompt's job, on the enclosing promise settling) — it only
      // stamps this attempt's output onto whatever root is currently open.
      sweepDanglingSpans(state);
      state.llmCtx = undefined;
      if (state.rootSpan) {
        // The LAST attempt's call wins: a retry/compaction/follow-up
        // continuation's own later agent_end simply overwrites
        // OI_OUTPUT_VALUE with ITS final assistant message, so the trace's
        // output always reflects the prompt() call's true final result
        // rather than an intermediate attempt's.
        stampRootOutput(state.rootSpan, event.messages, config.captureContent);
      }
      // Stamped onto the root as traceroot.pi.retry_count once the enclosing
      // prompt() call's own promise settles (see finalizeRootSpan in spans.ts).
      if (event.willRetry) {
        state.retryCount += 1;
      }
      break;
    }
    default:
      break;
  }
}
