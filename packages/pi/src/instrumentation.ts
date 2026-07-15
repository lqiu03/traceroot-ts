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

// Deliberately Symbol.for(), not a module-scoped Symbol(): this guard is
// stamped onto AgentSession.prototype, a shared object that can legitimately
// be reached by TWO independently-loaded copies of this package (e.g. a
// monorepo with a hoisting/dedup failure that leaves two differently-versioned
// installs of @traceroot-ai/pi both resolving to the same underlying
// @earendil-works/pi-coding-agent instance). Weighing the two options by what
// happens in that dual-copy scenario:
//
// - Symbol.for(key): looked up in the process-wide global symbol registry, so
//   both copies compute the SAME symbol value and see each other's stamp. The
//   second instrumentPiCodingAgent() call (e.g. a second tenant's config) is
//   rejected with a console.warn and its tracing never activates. That failure
//   is bounded to exactly one dropped config and is visible in logs — a
//   maintainer or on-call engineer can grep for it and understand immediately
//   what happened and why.
// - Symbol() (module-scoped, no registry key): each copy gets its own
//   distinct symbol, invisible to the other copy's guard check. Both copies
//   successfully patch prompt/steer/followUp/dispose, so EVERY real session
//   call runs through two independent listener layers — every span gets
//   exported twice, forever, for both configs, with zero warning. That is
//   strictly worse: unbounded (not one dropped config but every span,
//   indefinitely), silent (no log line to notice or grep for), and it doubles
//   real OTLP ingestion cost in production before anyone realizes.
//
// Properly supporting two distinct configs patching one shared prototype
// simultaneously would require multiplexing every event to multiple tracers —
// a real feature, out of scope here. Given the choice between those two
// failure modes, Symbol.for()'s "warn and drop the second config" is the
// safer default. Do not flip this back to a bare Symbol() without building
// that multiplexing support first.
const WRAPPED = Symbol.for('traceroot.pi_coding_agent.wrapped');

// A queued prompt() call's text, boxed in its own object rather than stored
// as a raw string: proto.prompt's rejection handler below needs to remove
// EXACTLY the entry it pushed (by reference identity) if that specific call
// never reaches agent_start, without accidentally removing a different,
// still-pending queue entry that happens to hold an equal string value.
interface QueuedPrompt {
  text: string;
  // Wall-clock time (Date.now()) this entry was enqueued, consulted at dequeue
  // to detect an entry that was stranded in the FIFO because its prompt() call
  // never reached agent_start (and, resolving rather than rejecting, never hit
  // removeIfStillQueued either). See MAX_QUEUED_PROMPT_AGE_MS.
  enqueuedAt: number;
}

// A queued prompt() entry older than this when a later agent_start arrives to
// claim it is treated as abandoned and skipped rather than trusted. A prompt()
// call whose run never starts — it hangs with no resolve/reject and no
// agent_start, so neither removeIfStillQueued (rejection-only) nor a normal
// agent_start dequeue ever removes it — would otherwise sit in the FIFO
// forever and be shifted out by some unrelated LATER run's agent_start,
// misattributing its text to that run and throwing every subsequent call off
// by one, permanently. This window is deliberately generous — far longer than
// any realistic single agent run a legitimately-queued overlapping prompt()
// could be waiting behind — so a genuinely-queued entry is never discarded,
// while a truly-stranded entry can only ever corrupt a call within this
// bounded window instead of cascading indefinitely.
const MAX_QUEUED_PROMPT_AGE_MS = 60 * 60 * 1000; // 1 hour

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
  // One-shot reservation set by agent_end when event.willRetry is true,
  // consumed by exactly the next agent_start. See agent_end's and
  // agent_start's own comments below for why only the explicitly-observable
  // retry case gets this strong "reuse regardless of what's queued"
  // guarantee, while compaction/follow-up continuations (which are NOT
  // observable via any AgentEvent field) only get the weaker empty-queue
  // fallback.
  reserveInputForRetry: boolean;
  // True when dispose()'s force-close swept this run's still-open root span,
  // so agent_end can tell that apart from having already closed it normally
  // itself. A host listener that calls session.dispose() synchronously while
  // handling agent_end (dispose() reassigns the listener array rather than
  // mutating it, so pi's own agent_end handler still runs afterward in the
  // same dispatch) would otherwise leave pi silently skipping the real close
  // purely because state.rootSpan is already undefined — producing an
  // incomplete trace with no signal. This flag lets agent_end detect and
  // surface that instead.
  rootForceClosedBySweep: boolean;
}

// Rate-limit for the dangling-span close-failure warning below. A Span
// implementation whose force-close throws on EVERY attempt (a broken exporter/
// processor, or a Span whose setAttribute always throws) would otherwise emit
// one console.warn for every dangling span, on every sweep, indefinitely —
// flooding the host's logs and burying the very first, genuinely-useful
// warning. After MAX_DANGLING_SPAN_WARNINGS failures inside a rolling
// DANGLING_SPAN_WARNING_WINDOW_MS window, one "further warnings suppressed"
// notice is emitted and the rest go quiet until the window rolls over — so an
// isolated, genuine failure is still surfaced, while a pathological one can no
// longer drown out the logs. Deliberately a small per-process counter, not a
// redesign: this is a defensive nicety on an already best-effort cleanup path.
const MAX_DANGLING_SPAN_WARNINGS = 10;
const DANGLING_SPAN_WARNING_WINDOW_MS = 60 * 1000;
let danglingSpanWarningCount = 0;
let danglingSpanWarningWindowStart = 0;

function warnDanglingSpanCloseFailure(label: string, err: unknown): void {
  const now = Date.now();
  if (now - danglingSpanWarningWindowStart > DANGLING_SPAN_WARNING_WINDOW_MS) {
    // First failure ever, or the previous burst's window has fully elapsed —
    // open a fresh window so a later, unrelated failure is never permanently
    // muted by an earlier burst that already hit the cap.
    danglingSpanWarningCount = 0;
    danglingSpanWarningWindowStart = now;
  }
  danglingSpanWarningCount += 1;
  if (danglingSpanWarningCount <= MAX_DANGLING_SPAN_WARNINGS) {
    console.warn(
      `[traceroot-pi] failed to force-close a dangling ${label} span during sweep (it may leak):`,
      err,
    );
  } else if (danglingSpanWarningCount === MAX_DANGLING_SPAN_WARNINGS + 1) {
    console.warn(
      `[traceroot-pi] more than ${MAX_DANGLING_SPAN_WARNINGS} dangling-span close failures in ` +
        `${DANGLING_SPAN_WARNING_WINDOW_MS / 1000}s — further such warnings suppressed until the ` +
        'failures stop.',
    );
  }
}

// closeDanglingSpan() (spans.ts) calls setAttr() before endSpanSafe() —
// setAttr()'s underlying span.setAttribute() is NOT wrapped in try/catch the
// way endSpanSafe() explicitly is ("Never let a misbehaving OTel exporter/
// processor crash the host app"), so a single misbehaving Span implementation
// can still throw out of closeDanglingSpan() itself. Catching per-span here —
// rather than only around the sweep as a whole, as dispose()'s own outer
// try/catch does — means one bad span can never prevent the sweep from
// reaching every OTHER span still queued up to close: without this, a throw
// on (say) the 2nd of 5 open tool spans would abort the loop and leave the
// remaining 3 tool spans, the LLM span, and (when includeRoot) the root span
// never closed — and a span with no .end() call is never exported at all,
// not merely "left open". `label` is logged so a real failure is traceable to
// the specific span kind (and, for tool spans, call id) that misbehaved; the
// warning itself is rate-limited (see warnDanglingSpanCloseFailure) so a
// systematically-failing span can't turn this into unbounded log spam.
function safeCloseDanglingSpan(span: Span | undefined, label: string): void {
  try {
    closeDanglingSpan(span);
  } catch (err) {
    warnDanglingSpanCloseFailure(label, err);
  }
}

// Force-closes every span left open by an abandoned run: every open tool
// span, then the LLM span, and — only when explicitly requested — the root
// span. Shared by all 4 places that need this exact "abandon whatever state
// was left dangling" sweep: agent_start (a previous run's agent_end never
// fired), turn_end (a turn ending with an abandoned LLM/tool span), agent_end
// (defensive cleanup before its own proper closeRootSpan() call), and
// dispose() (mid-run teardown). closeDanglingSpan() is already a no-op on
// `undefined`, and iterating + .clear()-ing an already-empty Map is already
// a no-op, so callers never need their own `if (span)` / `if (size > 0)`
// guard before calling this — a future change to sweep order or a new span
// type added to SessionSpanState only has to be made here, once. Each
// individual close goes through safeCloseDanglingSpan() (above), so one
// span's close throwing never aborts the rest of the sweep.
function sweepDanglingSpans(
  state: SessionSpanState,
  options: { includeRoot?: boolean } = {},
): void {
  for (const [toolCallId, span] of state.toolSpans) {
    safeCloseDanglingSpan(span, `tool (toolCallId=${toolCallId})`);
  }
  state.toolSpans.clear();
  safeCloseDanglingSpan(state.llmSpan, 'LLM');
  state.llmSpan = undefined;
  if (options.includeRoot) {
    safeCloseDanglingSpan(state.rootSpan, 'root');
    state.rootSpan = undefined;
  }
}

// Shifts the oldest genuinely-fresh entry off a session's FIFO input queue,
// discarding any head entries that have gone stale (their prompt() call never
// reached agent_start — see MAX_QUEUED_PROMPT_AGE_MS). Skips past every stale
// head entry rather than stopping at the first, so a run of stranded entries
// can never block the fresh one queued behind them. Returns undefined when the
// queue holds nothing but stale entries (or is empty), so agent_start falls
// back to its usual state.pendingInputText path instead of trusting stale
// text. Warns once per dequeue that discarded anything, matching this file's
// other console.warn conventions, so a real host-side "prompt never started"
// anomaly is visible rather than silently swallowed.
function dequeueFreshQueuedPrompt(queue: QueuedPrompt[]): QueuedPrompt | undefined {
  const now = Date.now();
  let discarded = 0;
  let entry = queue.shift();
  while (entry && now - entry.enqueuedAt > MAX_QUEUED_PROMPT_AGE_MS) {
    discarded += 1;
    entry = queue.shift();
  }
  if (discarded > 0) {
    console.warn(
      `[traceroot-pi] discarded ${discarded} stale queued prompt(s) whose run never started ` +
        '(older than the staleness window) instead of misattributing their text to this run.',
    );
  }
  return entry;
}

// Shared by proto.prompt/proto.steer/proto.followUp below: each of those
// three independently-callable entry points can be a session's first
// interaction, and each must guarantee session.subscribe() has been called
// exactly once for that session before delegating to the real method. All
// three previously copy-pasted this identical 3-line
// has()/add()/attachSpanListener() sequence; extracted here so future
// changes to the guard (or to attachSpanListener's argument list) only need
// to be made once. Pure structural extraction — no behavior change: the
// three call sites always passed session, tracer, resolved, pendingInput,
// sessionSpanState (all closed over here identically) to attachSpanListener.
function ensureSubscribed(
  session: AgentSessionInstance,
  tracer: Tracer,
  config: ResolvedPiInstrumentationConfig,
  pendingInput: WeakMap<AgentSessionInstance, QueuedPrompt[]>,
  subscribedSessions: WeakSet<AgentSessionInstance>,
  sessionSpanState: WeakMap<AgentSessionInstance, SessionSpanState>,
): void {
  if (subscribedSessions.has(session)) return;
  subscribedSessions.add(session);
  attachSpanListener(session, tracer, config, pendingInput, sessionSpanState);
}

export function instrumentPiCodingAgent(sdk: unknown, config?: PiInstrumentationConfig): unknown {
  const mod = sdk as PiCodingAgentModule;

  const resolved = resolveConfig(config);

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

  // Build the tracing pipeline BEFORE stamping the wrap-once guard, so the
  // shared-vs-private decision can gate the apiKey requirement. Shared mode (a
  // real global provider already registered — e.g. by TraceRoot.initialize()
  // earlier in the same process) never constructs an OTLP exporter, so it
  // needs no apiKey at all; only the private-provider fallback that builds its
  // own OTLP exporter does. A _spanExporter override likewise builds a private
  // provider but needs no apiKey (a test rig, or a caller supplying its own
  // exporter). Requiring an apiKey up front — as this used to, before the
  // shared pipeline existed — silently no-op'd the entire feature for every
  // host that configured TraceRoot programmatically and never set
  // TRACEROOT_API_KEY, even though a fully valid shared pipeline was available.
  const tracing = createTracing(resolved);
  if (tracing.ownsProvider && !resolved.apiKey && !resolved.spanExporterOverride) {
    console.warn(
      '[traceroot-pi] no global OTel TracerProvider is registered and no TRACEROOT_API_KEY ' +
        '(or config.apiKey) is set — cannot build an export pipeline, so instrumentation is ' +
        'disabled. Call this after TraceRoot.initialize() (shared mode), or provide an apiKey.',
    );
    return sdk;
  }

  const { tracer, forceFlush, ownsProvider } = tracing;

  // Per-session FIFO queue, not a single slot: a second prompt() call can
  // fire before the first run's agent_start event has arrived (overlapping
  // runs on the same session), and each run's agent_start must claim the
  // text from the prompt() call that actually triggered it, in order —
  // never let a later prompt() call's text clobber an earlier one's.
  const pendingInput = new WeakMap<AgentSessionInstance, QueuedPrompt[]>();
  const subscribedSessions = new WeakSet<AgentSessionInstance>();
  // Mirrors pendingInput: attachSpanListener() below creates one
  // SessionSpanState per session and closes over it for its own subscribe()
  // callback, but AgentSession.prototype.dispose (patched once, below) has
  // no closure over any particular session — it needs this out-of-band map
  // to reach whichever session's SessionSpanState (if any) it was called on.
  const sessionSpanState = new WeakMap<AgentSessionInstance, SessionSpanState>();

  // Install everything below (the flush hook plus every prototype method
  // patch), then — and only then — stamp the wrap-once guard. Stamping the
  // guard up front and then throwing partway through setup (a bad option, an
  // exotic duck-type check that throws) would leave the prototype permanently
  // marked "wrapped" while the SDK was never actually patched: every later
  // instrumentPiCodingAgent() call would then be silently rejected with
  // "config ignored" instead of the real failure ever surfacing. Each patch
  // records an undo on `rollback`, so a mid-setup failure restores the
  // prototype to exactly how it was found (no half-patched state that a later
  // retry would double-wrap) before rethrowing a clear install error.
  const rollback: Array<() => void> = [];
  try {
    // Only register our own flush hook when we built our own private
    // provider. In shared mode, a globally-registered provider elsewhere in
    // the process (e.g. TraceRoot.initialize()) already owns flush -- via its
    // own 'beforeExit' hook -- so registering a second one here would be
    // redundant at best, and calling forceFlush() on a provider we don't own
    // is not this package's responsibility to manage.
    if (ownsProvider) {
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
      // Registration is bounded per DISTINCT AgentSession module object
      // instrumented, not deduped process-wide: each successful call here
      // builds and owns its own private OTLP pipeline (see createTracing()
      // above) and therefore needs its own flush hook. `rollback` only
      // removes this listener on a mid-setup failure (see the try/catch
      // below) -- there is deliberately no public API to remove it after a
      // SUCCESSFUL install. That is fine for the common case (a host calls
      // instrumentPiCodingAgent() once per process, guarded by WRAPPED
      // above, and the listener lives for the process's lifetime alongside
      // the provider it flushes), but it IS a real, intentional gap for a
      // host that instruments multiple independent SDK objects over its
      // lifetime (HMR module reload, per-tenant pooling): each one leaves
      // its own permanent listener + provider behind with no way to tear
      // either down, even after every session built from that SDK object has
      // long since been disposed. proto.dispose (patched below) only clears
      // per-session span state; it intentionally does not touch this
      // process-level hook, since one session's dispose() must not silently
      // kill flush for every other session still sharing this same private
      // pipeline. Bounded-per-process growth (proportional to the number of
      // distinct SDK objects a host instruments, not to session count) is
      // the accepted tradeoff for now rather than building real
      // multi-pipeline teardown; see
      // tests/test-helpers.ts's makeRig() and the bounded-listener-growth
      // test in tests/test-helpers.test.ts for how this is verified and kept
      // from silently regressing behind Node's MaxListenersExceededWarning
      // threshold.
      const flushOnExit = (): void => {
        void forceFlush();
      };
      process.once('beforeExit', flushOnExit);
      rollback.push(() => process.removeListener('beforeExit', flushOnExit));
    }

    const originalPrompt = proto.prompt;
    proto.prompt = function (this: AgentSessionInstance, text, options) {
      // Before choosing a proactive (enumerate-the-early-returns) vs reactive
      // (clean up by reference identity once THIS call's own promise settles)
      // strategy here, the real question is: for the path that DOES reach
      // _runAgentPrompt, is agent_start GUARANTEED to have already fired (and
      // already shifted some queue entry) by the time prompt()'s own returned
      // promise settles? Reading the real, installed
      // @earendil-works/pi-coding-agent@0.80.6 source alone says yes:
      // `_runAgentPrompt` is the last expression `prompt()` awaits, and inside
      // it `await this.agent.prompt(messages)` cannot itself resolve before the
      // underlying Agent has already emitted agent_start. That would make a
      // purely reactive `result.finally(removeIfStillQueued)` sound in theory.
      //
      // It is NOT sound in practice for this specific FIFO design, though —
      // verified empirically, not assumed: the very feature this queue exists
      // for (two prompt() calls queuing before EITHER's agent_start has fired
      // yet — see the "two overlapping prompt() calls" and retry/compaction
      // continuation tests) means a still-genuinely-in-flight call's own
      // promise can legitimately still be unsettled/settled independently of
      // when ITS OWN agent_start arrives, from the perspective of anything
      // that can only observe promise settlement, not the SDK's real internal
      // ordering. Trying `result.finally(removeIfStillQueued)` here and running
      // the full suite reproduced exactly that: 13 unrelated, previously-green
      // tests started failing because their queued entries were reactively
      // stripped before their own later-emitted agent_start could claim them.
      // A reactive strategy is therefore the wrong choice here, regardless of
      // what the raw promise-ordering guarantee alone would suggest — this
      // package's own FIFO semantics don't preserve the "settle implies
      // consumed" property a reactive check needs. Proactive enumeration,
      // extended below to cover as many early-return paths as can be
      // determined SYNCHRONOUSLY and precisely from public SDK surface, is the
      // safe option.
      //
      // (a) isStreaming + streamingBehavior — verified against the real,
      // installed SDK (dist/core/agent-session.js:812-824): when
      // `this.isStreaming` is true and `options.streamingBehavior` is set,
      // prompt() calls `_queueSteer`/`_queueFollowUp` (which inject the message
      // into the CURRENTLY-running Agent loop, not a new run) and returns —
      // `_runAgentPrompt` is never reached. `this.isStreaming` is read
      // synchronously here, before `originalPrompt` is invoked — matching the
      // real prompt()'s own synchronous read of the same getter.
      const isStreamedQueueOnly = this.isStreaming === true && !!options?.streamingBehavior;
      // (b) a leading "/" matched by a registered extension command — verified
      // against the real, installed SDK (dist/core/agent-session.js:783-790):
      // `if (expandPromptTemplates && text.startsWith("/")) { const handled =
      // await this._tryExecuteExtensionCommand(text); if (handled) {
      // preflightResult?.(true); return; } }`. This is decidable precisely
      // (not just heuristically) from outside: _tryExecuteExtensionCommand's
      // own command lookup (agent-session.js:903-908) parses the command name
      // identically to below, and its try/catch (913-925) means ANY registered
      // command — even one whose handler throws — still returns true. So a
      // truthy `getCommand()` lookup via the SDK's own public
      // `session.extensionRunner` getter (agent-session.js:2629-2630, and
      // ExtensionRunner.getCommand is itself public — dist/core/extensions/
      // runner.d.ts:128) deterministically means this call will never reach
      // agent_start, with no false-positive case.
      const expandPromptTemplates = options?.expandPromptTemplates ?? true;
      let matchesExtensionCommand = false;
      if (expandPromptTemplates && typeof text === 'string' && text.startsWith('/')) {
        const spaceIndex = text.indexOf(' ');
        const commandName = spaceIndex === -1 ? text.slice(1) : text.slice(1, spaceIndex);
        matchesExtensionCommand = !!this.extensionRunner?.getCommand?.(commandName);
      }
      // (c) any extension with an 'input' hook returning action: 'handled' —
      // verified against the real, installed SDK (dist/core/agent-session.js:
      // 794-799): `if (this._extensionRunner.hasHandlers("input")) { const
      // inputResult = await this._extensionRunner.emitInput(...); if
      // (inputResult.action === "handled") { preflightResult?.(true); return; }
      // }`. Unlike (b), this is NOT precisely decidable from outside: whether a
      // specific text ends up "handled" depends on the registered hook
      // function's own logic, which this patch layer cannot inspect or
      // pre-invoke (calling it ourselves to find out would double-invoke a
      // potentially side-effecting extension). The best available signal is
      // the SDK's own public `session.hasExtensionHandlers('input')`
      // (agent-session.js:618, mirrors the exact same `hasHandlers("input")`
      // check prompt() itself makes) — true whenever ANY 'input' hook is
      // registered, whether or not it will actually intercept THIS text. This
      // is a deliberate, documented best-effort heuristic, not a precise
      // detection: a session with an 'input' hook that only intercepts SOME
      // messages will, for every message it does NOT intercept, still take
      // this proactive branch and skip queuing — losing that real run's
      // input.value (falling back to agent_start's empty-queue heuristic)
      // rather than corrupting a LATER call's attribution. That tradeoff —
      // graceful degradation (a missing/stale input.value) over the
      // cross-call corruption this queue exists to prevent — is preferred
      // given the SDK exposes no way to precisely predict a specific hook's
      // decision without invoking it. This (and the enumeration of early-return
      // paths above) must be kept in sync with the SDK and may miss future
      // early-return paths a later SDK version adds.
      const mayBeHandledByInputHook = this.hasExtensionHandlers?.('input') === true;
      const willQueueWithoutAgentStart =
        isStreamedQueueOnly || matchesExtensionCommand || mayBeHandledByInputHook;
      let entry: QueuedPrompt | undefined;
      if (typeof text === 'string' && !willQueueWithoutAgentStart) {
        entry = { text, enqueuedAt: Date.now() };
        const queue = pendingInput.get(this);
        if (queue) queue.push(entry);
        else pendingInput.set(this, [entry]);
      }
      ensureSubscribed(this, tracer, resolved, pendingInput, subscribedSessions, sessionSpanState);
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
      // Deliberately still gated on rejection only (.catch(), not .finally())
      // — see this function's opening comment for why widening this to also
      // run on resolve is unsafe for this FIFO's own overlapping-call
      // semantics, even though it would be sound against the raw SDK guarantee
      // alone.
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
    rollback.push(() => {
      proto.prompt = originalPrompt;
    });

    // steer() and followUp() are standalone public SDK entry points, not
    // aliases or wrappers of prompt() — a host embedding an interactive Pi
    // session (this package's own README target) can call
    // session.steer(text)/session.followUp(text) directly, without ever
    // calling session.prompt() on that session first. Before this patch,
    // attachSpanListener() was only ever reached from inside proto.prompt
    // above, so a session whose first (or only) interaction was steer()/
    // followUp() never got session.subscribe() called on it at all — every
    // subsequent AgentEvent (agent_start through agent_end) silently produced
    // zero spans for the entire lifetime of that session.
    //
    // Deliberately NOT enqueuing steer()/followUp() text into pendingInput the
    // way prompt() does above: verified against the real, installed
    // @earendil-works/pi-agent-core@0.80.6 (dist/agent.js:169-176),
    // Agent.steer()/Agent.followUp() only ever push onto an internal queue —
    // neither one ever itself triggers a fresh run (only Agent.prompt()/
    // continue() do, via runPromptMessages()). Because pendingInput is a FIFO
    // consumed exclusively by agent_start, queuing steer()/followUp() text
    // into it would misattribute that text to whatever LATER, unrelated run's
    // agent_start happens to fire next — a worse bug than the "no tracing at
    // all" defect this patch fixes. Only patched when present as a function —
    // defensive, like the dispose() patch below, so a minimal/partial double
    // never disables prompt instrumentation over a missing, unrelated method.
    if (typeof proto.steer === 'function') {
      const originalSteer = proto.steer;
      proto.steer = function (this: AgentSessionInstance, text: string, images?: unknown[]) {
        ensureSubscribed(
          this,
          tracer,
          resolved,
          pendingInput,
          subscribedSessions,
          sessionSpanState,
        );
        return originalSteer.call(this, text, images);
      };
      rollback.push(() => {
        proto.steer = originalSteer;
      });
    }
    if (typeof proto.followUp === 'function') {
      const originalFollowUp = proto.followUp;
      proto.followUp = function (this: AgentSessionInstance, text: string, images?: unknown[]) {
        ensureSubscribed(
          this,
          tracer,
          resolved,
          pendingInput,
          subscribedSessions,
          sessionSpanState,
        );
        return originalFollowUp.call(this, text, images);
      };
      rollback.push(() => {
        proto.followUp = originalFollowUp;
      });
    }

    // Patched once here (guarded by the same WRAPPED check/stamp above that
    // guards proto.prompt — this whole function body only runs once per sdk),
    // never per-session: dispose() is a single shared prototype method, same
    // as prompt(). Only patched when dispose actually exists as a function —
    // AgentSessionInstance's type declares it as required, but this stays
    // defensive so a minimal/partial double never disables prompt
    // instrumentation over a missing, unrelated method.
    if (typeof proto.dispose === 'function') {
      const originalDispose = proto.dispose;
      proto.dispose = function (this: AgentSessionInstance): void {
        // See this file's module header and types.ts's AgentSessionInstance
        // doc comment: the real dispose() only reassigns the SDK's private
        // _eventListeners array, which stops our subscribe() callback from
        // ever firing again but does nothing to whatever spans that callback
        // had already opened. If a host calls dispose() mid-run — after
        // agent_start but before agent_end — this session's SessionSpanState
        // (if it was ever subscribed) can still have an open rootSpan/llmSpan/
        // toolSpans that will now never see their normal close event. Force-
        // close them here first, exactly like agent_start's own dangling-span
        // sweep, so they still export instead of leaking silently (a span
        // that never has .end() called on it is never recorded/exported at
        // all) — then delegate to the real dispose().
        const state = sessionSpanState.get(this);
        if (state) {
          // Whether this dispose() force-closed a still-open root span. If a
          // host's own agent_end listener triggered this dispose() reentrantly,
          // pi's own agent_end handler will still run afterward and must be
          // able to tell its root span was force-closed out from under it (see
          // agent_end's handler and SessionSpanState.rootForceClosedBySweep).
          const hadOpenRootSpan = state.rootSpan !== undefined;
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
            if (hadOpenRootSpan) state.rootForceClosedBySweep = true;
            state.rootSpan = undefined;
            state.rootCtx = undefined;
            state.llmSpan = undefined;
            state.llmCtx = undefined;
            // Avoids re-sweeping already-ended spans if dispose() is ever
            // called twice on the same session; a second call now takes the
            // same no-op path as a session with no SessionSpanState at all.
            sessionSpanState.delete(this);
            pendingInput.delete(this);
            // subscribedSessions gates every "already subscribed?" check at
            // the top of proto.prompt/steer/followUp above. Nothing in the
            // real SDK stops a host from calling prompt()/steer()/followUp()
            // again on a session instance after dispose() — dispose() only
            // clears the SDK's own _eventListeners array (see this file's
            // module header), it does not make the session instance itself
            // unusable. Without this delete, a session reused after dispose()
            // would find subscribedSessions.has(this) still true forever and
            // silently never call attachSpanListener()/session.subscribe()
            // again — every span from every run after the first dispose()
            // would be dropped with no warning. Deleting here, alongside
            // sessionSpanState, means the next prompt()/steer()/followUp()
            // call on this same instance re-subscribes and resumes tracing
            // normally, exactly like a brand-new session would.
            subscribedSessions.delete(this);
          }
        }
        // A session that was never subscribed (no prompt() call ever reached
        // the attachSpanListener() branch above) or that has no open spans
        // left (the common, already-idle case — e.g. dispose() called after
        // agent_end already closed everything normally) hits nothing but the
        // `if (state)` check above, so this call into the real dispose() is
        // unchanged: same arguments, same return value, same timing as before
        // this patch existed.
        return originalDispose.call(this);
      };
      rollback.push(() => {
        proto.dispose = originalDispose;
      });
    }

    // Setup fully succeeded: only now is it correct to mark this prototype
    // wrapped, so the wrap-once guard can never be left true over a prototype
    // that was never actually patched. This MUST be the last statement INSIDE
    // the try: if the prototype was frozen/sealed between the patches above and
    // here (or a conflicting non-configurable WRAPPED-keyed property already
    // exists), Object.defineProperty throws — and that failure has to unwind
    // the method patches already applied, exactly like any other mid-setup
    // failure. Left outside the try (as it once was), a throw here escaped
    // rollback: the prototype stayed patched but UNSTAMPED, so the very next
    // instrumentPiCodingAgent() call saw no guard, re-patched the already-
    // patched methods, and every event emitted duplicate spans forever. No
    // rollback entry is pushed for the stamp itself: Object.defineProperty
    // never partially applies, so on failure the property was never defined and
    // there is nothing to undo for it — only the PRIOR patches need unwinding.
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
  tracer: Tracer,
  config: ResolvedPiInstrumentationConfig,
  pendingInput: WeakMap<AgentSessionInstance, QueuedPrompt[]>,
  sessionSpanState: WeakMap<AgentSessionInstance, SessionSpanState>,
): void {
  const state: SessionSpanState = {
    rootSpan: undefined,
    rootCtx: undefined,
    llmSpan: undefined,
    llmCtx: undefined,
    toolSpans: new Map(),
    pendingInputText: undefined,
    reserveInputForRetry: false,
    rootForceClosedBySweep: false,
  };
  // Reachable from AgentSession.prototype.dispose (patched once, in
  // instrumentPiCodingAgent() above) so a mid-run dispose() can force-close
  // whatever this session's listener callback below left open.
  sessionSpanState.set(session, state);

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
      // sweepDanglingSpans() force-closes the tool spans, the LLM span, and
      // (with includeRoot: true here) the root span unconditionally — it
      // never needs to gate on state.rootSpan first. That matters because a
      // stray tool_execution_start OR a stray message_start (with no
      // matching message_end) can arrive AFTER a prior run's agent_end
      // already cleared rootSpan (e.g. an async tool callback resolving
      // late, or a straggler stream event), leaving an orphaned entry with
      // no rootSpan to gate on. Without sweeping each one unconditionally,
      // that orphan would never be swept here and would either stay open
      // forever (tool span) or have its reference silently overwritten below
      // without ever calling .end() on it (LLM span) — and a span that never
      // has .end() called on it is never recorded/exported at all, not
      // merely "left open".
      sweepDanglingSpans(state, { includeRoot: true });
      const parentCtx = context.active();
      // pendingInput is a per-session FIFO queue (see instrumentPiCodingAgent's
      // own comment) — dequeue the oldest queued text so overlapping prompt()
      // calls each hand their text to the correct run's agent_start, in order.
      //
      // EXCEPTION: if agent_end just reserved this exact next agent_start for
      // a retry (state.reserveInputForRetry — set only when event.willRetry
      // was true, see agent_end's comment below), that reservation wins even
      // when the queue is non-empty. A retry's phantom continuation must
      // reuse the retrying run's OWN text, never whatever a second, genuinely
      // distinct prompt() call already queued behind it — see
      // prompt-queue.test.ts's "reuses ITS OWN pendingInputText even when a
      // second ... prompt() call is already queued behind it" test for the
      // exact corruption this prevents. The
      // flag is consumed here exactly once: it is set immediately before
      // this specific agent_start (agent.continue() is called synchronously,
      // with no other event able to interleave in between — see agent_end's
      // comment), so it can never leak forward onto some LATER, unrelated
      // agent_start.
      //
      // If the reservation is not set AND nothing is queued, this agent_start
      // has NO corresponding new prompt() call at all. Verified against the
      // real, installed @earendil-works/pi-coding-agent@0.80.6
      // (dist/core/agent-session.js): AgentSession._runAgentPrompt() runs
      // `while (await this._handlePostAgentRun()) { await
      // this.agent.continue(); }`, and _handlePostAgentRun() returns true for
      // THREE independent reasons — a retryable error, an ordinary
      // auto-compaction continuation (_checkCompaction), or an
      // extension-queued follow-up from its own agent_end handler
      // (agent.hasQueuedMessages()) — and agent.continue() unconditionally
      // emits a fresh agent_start in every case. None of those three push
      // anything onto pendingInput, so an empty queue here (with no
      // reservation active) means this run is a continuation of whichever
      // run just used state.pendingInputText (deliberately left untouched by
      // agent_end — see its comment) rather than a genuinely new call. This
      // empty-queue heuristic is the best available fallback for compaction/
      // follow-up specifically because — unlike a retry — the SDK gives no
      // event field that distinguishes those two from a genuinely new call,
      // so they cannot get the same positive-priority reservation a retry
      // gets.
      let inputText: string | undefined;
      if (state.reserveInputForRetry) {
        state.reserveInputForRetry = false;
        inputText = state.pendingInputText;
      } else {
        const queue = pendingInput.get(session);
        const queuedEntry = queue ? dequeueFreshQueuedPrompt(queue) : undefined;
        inputText = queuedEntry ? queuedEntry.text : state.pendingInputText;
      }
      state.pendingInputText = inputText;
      state.rootSpan = openRootSpan(tracer, parentCtx, {
        text: inputText,
        sessionId: session.sessionId,
        captureContent: config.captureContent,
      });
      state.rootCtx = trace.setSpan(parentCtx, state.rootSpan);
      state.llmSpan = undefined;
      state.llmCtx = undefined;
      // Fresh run: clear any close-disposition left over from a prior run on
      // this reused state object, so agent_end's reentrant-dispose detection
      // can never fire on a stale flag.
      state.rootForceClosedBySweep = false;
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
      safeCloseDanglingSpan(state.llmSpan, 'LLM');
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
      // state reference — force-close it here instead of leaking it. Tool
      // calls are turn-scoped too: any tool span still open when the turn
      // ends (its tool_execution_end never arrived) must be force-closed
      // here as well, instead of staying open until agent_end. Root span is
      // deliberately left untouched — turn_end isn't session end.
      sweepDanglingSpans(state);
      state.llmCtx = undefined;
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
      safeCloseDanglingSpan(existing, `tool (toolCallId=${event.toolCallId})`);
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
      // Defensive cleanup for any tool/LLM span this run left dangling,
      // mirroring turn_end's identical sweep — normally both are already
      // empty by the time agent_end fires. The root span is NOT part of this
      // sweep: it gets its own proper closeRootSpan() below rather than a
      // force-close, since agent_end is the real, expected end of a run.
      sweepDanglingSpans(state);
      state.llmCtx = undefined;
      if (state.rootSpan) {
        closeRootSpan(state.rootSpan, event.messages, event.willRetry, config.captureContent);
      } else if (state.rootForceClosedBySweep) {
        // This run's root span is gone not because agent_end already ran, but
        // because a reentrant dispose() (a host's own earlier-registered
        // agent_end listener disposing the session synchronously) force-closed
        // it before this handler got to run for the same event. The real close
        // — which stamps output.value and the retry flag — can no longer
        // happen (the root span is already ended), so the exported AGENT span
        // is a FORCE_CLOSED one missing this run's final output. Surface that
        // rather than silently dropping the completion data.
        console.warn(
          "[traceroot-pi] agent_end arrived after this run's root span was already force-closed " +
            'by a reentrant dispose(); the exported AGENT span is missing its final output/retry ' +
            'attributes.',
        );
      }
      state.rootSpan = undefined;
      state.rootCtx = undefined;
      // Deliberately do NOT clear state.pendingInputText here, and do NOT
      // gate reuse on event.willRetry. `willRetry` reflects ONLY Pi's
      // auto-retry heuristic (computed purely from _isRetryableError on the
      // last assistant message — verified against the real, installed
      // @earendil-works/pi-coding-agent@0.80.6's
      // AgentSession._willRetryAfterAgentEnd). But
      // AgentSession._runAgentPrompt()'s own loop (`while (await
      // this._handlePostAgentRun()) { await this.agent.continue(); }`)
      // re-enters — firing a fresh agent_start with no corresponding new
      // prompt() call — for two other, completely unrelated reasons too: an
      // ordinary auto-compaction continuation (_checkCompaction returning
      // true on a routine successful turn) and an extension queueing a
      // follow-up message from its own agent_end handler
      // (agent.hasQueuedMessages()). Both leave willRetry false, so gating
      // the reuse on willRetry alone silently discarded the input text (or
      // let the next agent_start mis-dequeue an unrelated queued call's
      // text) on those two paths. Rather than special-case each mechanism
      // here — this event carries no signal that distinguishes them —
      // simply leave state.pendingInputText as whatever this run used.
      // agent_start's own handler (see its comment) decides whether to reuse
      // it (nothing new queued — a continuation of any kind) or overwrite it
      // with a genuinely new prompt() call's text (queue non-empty).
      //
      // willRetry DOES get one extra, stronger guarantee on top of that
      // shared fallback: when it's true, reserve this run's pendingInputText
      // for the very next agent_start via a one-shot flag (consumed in
      // agent_start's handler above), so the retry's phantom continuation
      // reuses its own text even if a second, genuinely distinct prompt()
      // call is already sitting at the front of the queue (see that test's
      // "even when a second ... prompt() call is already queued behind it"
      // case). This asymmetry — positive priority for retry, but only the
      // weaker empty-queue heuristic for compaction/follow-up — exists
      // because retry is the ONE reason _handlePostAgentRun() can return true
      // that IS explicitly observable on this event (event.willRetry).
      // Compaction and extension-queued-follow-up are not: both leave
      // willRetry false and this event's shape ({messages, willRetry?}) has
      // no other field to tell them apart from a genuinely new prompt() call,
      // so there is no SDK signal available to give them the same strong
      // guarantee — the empty-queue fallback is the best available
      // approximation for those two.
      if (event.willRetry) {
        state.reserveInputForRetry = true;
      }
      break;
    }
    case 'auto_retry_end': {
      // The SDK cancelled the retry (session.abort() -> abortRetry() aborts
      // _prepareRetry's backoff sleep, which returns false so NO continuation
      // agent_start fires) or exhausted it. Either way the reservation armed
      // by agent_end{willRetry:true} above will never be consumed by a retry
      // continuation; clear it so it can't be mis-consumed by a LATER,
      // genuinely-new prompt() call's agent_start (which would then export
      // that new run's span carrying the previous run's input.value, shifting
      // every overlapping call off by one). Clearing on the retries-exhausted
      // path is harmless: willRetry is false at maxRetries so the reservation
      // was never armed there.
      if (event.success === false) {
        state.reserveInputForRetry = false;
      }
      break;
    }
    default:
      break;
  }
}
