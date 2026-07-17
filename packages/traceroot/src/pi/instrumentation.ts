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
 * ── One trace per prompt() call ───────────────────────────────────────────
 * The root AGENT span is anchored on the wrapped prompt() call's own promise
 * window — opened at entry, closed when that promise settles — mirroring
 * claude-agent-sdk.ts's one-root-per-query() idiom. This is deliberately NOT
 * anchored on agent_start/agent_end: verified against the real, installed
 * @earendil-works/pi-coding-agent@0.80.6 (dist/core/agent-session.js),
 * AgentSession.prompt() awaits _runAgentPrompt(), whose
 * `while (await this._handlePostAgentRun()) { await this.agent.continue(); }`
 * loop performs EVERY retry/compaction/follow-up continuation inside the
 * awaited promise, and each attempt's own agent_end handler completes before
 * prompt() resolves. So a single root spanning the whole prompt() promise
 * captures every continuation attempt as a child LLM/tool span (an ERROR
 * attempt followed by an OK attempt, for example) instead of pi's internal
 * retry/compaction/follow-up control flow leaking into trace *structure* as
 * multiple sibling traces with identical input.value. See proto.prompt below
 * for the open/close mechanics and the three DECIDED boundary policies.
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

// This in-tree integration never builds or owns its own OTel TracerProvider —
// core (traceroot.ts / TraceRoot.initialize()) guarantees a real, globally
// registered provider is already in place before instrumentPiCodingAgent()
// ever runs. It still acquires its tracer through the shared
// createReresolvingTracer() (see ../reresolving-tracer) rather than a captured
// trace.getTracer() call, so spans survive a TraceRoot.shutdown()/initialize()
// cycle's ProxyTracerProvider swap.

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
  // finalizeRootSpan in spans.ts) — supersedes the old per-attempt
  // traceroot.pi.will_retry flag, which could not represent "this prompt()
  // call retried N times" on a single trace now that every attempt shares
  // one root.
  retryCount: number;
  // True when dispose()'s force-close swept this window's still-open root
  // span, so agent_end can tell that apart from the root simply still being
  // mid-flight. Only dispose() ever sets this true (in its own
  // `if (hadOpenRootSpan) state.rootForceClosedBySweep = true;` branch) —
  // proto.prompt's OVERLAP SAFETY sweep never does, and proto.prompt
  // unconditionally resets this to false the moment it opens a fresh root
  // for a new window, so a stale true value can only ever be observed by
  // THIS SAME window's own still-in-flight agent_end. A host listener that
  // calls session.dispose()
  // synchronously while handling agent_end (dispose() reassigns the listener
  // array rather than mutating it, so pi's own agent_end handler still runs
  // afterward in the same dispatch) would otherwise leave pi silently
  // skipping the real close purely because state.rootSpan is already
  // undefined — producing an incomplete trace with no signal. This flag lets
  // agent_end detect and surface that instead.
  rootForceClosedBySweep: boolean;
}

// Force-closes every span left open by an abandoned run: every open tool
// span, then the LLM span, and — only when explicitly requested — the root
// span. Shared by every call site that needs this exact "abandon whatever
// state was left dangling" sweep: agent_start (a previous ATTEMPT within the
// same prompt() window crashed before its own agent_end), turn_end (a turn
// ending with an abandoned LLM/tool span), agent_end (defensive cleanup
// before stamping this attempt's output), proto.prompt (the OVERLAP SAFETY
// sweep — see its own comment), and dispose() (mid-run teardown).
// closeDanglingSpan() (spans.ts) is already a no-op on `undefined`, already
// guards its own setAttr(FORCE_CLOSED) call so one misbehaving span can never
// stop it from reaching endSpanSafe(), and iterating + .clear()-ing an
// already-empty Map is already a no-op — so callers never need their own
// `if (span)` / `if (size > 0)` guard before calling this, and one span's
// close failure never aborts the rest of the sweep.
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
// interaction, and each must guarantee session.subscribe() has been called
// exactly once for that session before delegating to the real method. All
// three previously copy-pasted this identical 3-line
// has()/add()/attachSpanListener() sequence; extracted here so future
// changes to the guard (or to attachSpanListener's argument list) only need
// to be made once. Pure structural extraction — no behavior change: the
// three call sites always passed session, tracer, resolved, sessionSpanState
// (all closed over here identically) to attachSpanListener.
//
// Guarantees a SessionSpanState exists for `session` once this returns:
// either it already did (subscribedSessions.has(session) was true), or
// attachSpanListener() below just created and stored one. proto.prompt relies
// on this — `sessionSpanState.get(this)` immediately after calling this
// function is guaranteed non-undefined.
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

  // This in-tree integration never builds an export pipeline of its own —
  // unlike the old standalone package, there is no createTracing()/
  // hasRealGlobalProvider() probe and no "no export pipeline available,
  // disable instrumentation" failure mode to gate on here. Core
  // (traceroot.ts / TraceRoot.initialize()) guarantees a real, globally
  // registered OTel TracerProvider is already in place before this function
  // ever runs, so the tracer below can simply re-resolve that global provider
  // on every span-open (see createReresolvingTracer for why that indirection
  // still matters even with a guaranteed provider).
  const tracer = createReresolvingTracer(TRACER_NAME, SDK_VERSION);

  const subscribedSessions = new WeakSet<AgentSessionInstance>();
  // attachSpanListener() below creates one SessionSpanState per session and
  // closes over it for its own subscribe() callback, but
  // AgentSession.prototype.dispose (patched once, below) has no closure over
  // any particular session — it needs this out-of-band map to reach
  // whichever session's SessionSpanState (if any) it was called on.
  const sessionSpanState = new WeakMap<AgentSessionInstance, SessionSpanState>();

  // Install everything below (every prototype method patch), then — and only
  // then — stamp the wrap-once guard. Stamping the guard up front and then
  // throwing partway through setup (a bad option, an exotic duck-type check
  // that throws) would leave the prototype permanently marked "wrapped" while
  // the SDK was never actually patched: every later instrumentPiCodingAgent()
  // call would then be silently rejected with "config ignored" instead of the
  // real failure ever surfacing. Each patch records an undo on `rollback`, so
  // a mid-setup failure restores the prototype to exactly how it was found (no
  // half-patched state that a later retry would double-wrap) before rethrowing
  // a clear install error.
  const rollback: Array<() => void> = [];
  try {
    // Process-exit flushing is core's responsibility now (see
    // packages/traceroot/src/traceroot.ts's own process.once('beforeExit', ...)
    // auto-flush convention) — this integration never builds or owns a
    // TracerProvider of its own, so it has no flush hook to register or roll
    // back here the way the old standalone package's private-provider path
    // did.

    const originalPrompt = proto.prompt;
    proto.prompt = function (this: AgentSessionInstance, text, options) {
      ensureSubscribed(this, tracer, resolved, subscribedSessions, sessionSpanState);

      // MID-STREAM STEER: verified directly against the real, installed
      // @earendil-works/pi-coding-agent@0.80.6 (dist/core/agent-session.js,
      // around its own
      // `if (this.isStreaming) { if (!options?.streamingBehavior) throw ...;
      // ... await this._queueSteer/_queueFollowUp(...); return; }` branch) —
      // when a run is ALREADY streaming and the caller passes
      // options.streamingBehavior ('steer' or 'followUp'), prompt() injects
      // the text into the CURRENTLY-running run's queue and returns early.
      // It does NOT start a new run and does NOT throw. This must be
      // detected and delegated straight through BEFORE any root management
      // below: opening a fresh root here (and running the OVERLAP SAFETY
      // sweep further down) would force-close the ACTIVE run's still-open
      // root mid-flight, orphaning its children and discarding whatever
      // output it eventually produces — beheading a trace that is still
      // legitimately in progress, not merely stale. This is the ONLY
      // early-return path that skips root management entirely: the
      // "/command" and extension-runner 'input'-hook early returns below
      // happen on a fresh, non-streaming call, so their trivial, childless
      // root (boundary policy 2) is correct and unaffected by this check.
      const isQueueOnlySteer = this.isStreaming === true && !!options?.streamingBehavior;
      if (isQueueOnlySteer) {
        return originalPrompt.call(this, text, options);
      }

      // Guaranteed non-undefined: ensureSubscribed() either found an existing
      // entry or attachSpanListener() just created one for `this` session.
      const state = sessionSpanState.get(this) as SessionSpanState;

      // OVERLAP SAFETY: a previous prompt() window's root is still open here.
      // This is deliberately NOT the mid-stream steer case above — that is
      // detected and returned before this point is ever reached. The real
      // SDK's isStreaming guard only throws when the caller omits
      // streamingBehavior; this sweep is a last-resort safety net for a
      // genuinely-new, non-streaming prompt() call that races a still-open
      // prior window (e.g. a caller-side bug, or an SDK internal state
      // transition not verified to be covered here). Force-close that stale
      // window (root included) rather than silently overwriting
      // state.rootSpan and leaking it unended forever (a span that never has
      // .end() called on it is never exported at all).
      if (state.rootSpan) {
        sweepDanglingSpans(state, { includeRoot: true });
        // Matches every other sweep site's cleanup (agent_start, turn_end,
        // agent_end): sweepDanglingSpans() only clears state.llmSpan, not
        // the parenting context that goes with it. Left stale, a
        // tool_execution_start racing in right after this sweep would
        // wrongly parent under the now-force-closed LLM span's context
        // instead of falling back to the fresh root about to be opened
        // below.
        state.llmCtx = undefined;
      }

      // parentCtx = context.active(), matching both the pre-existing root
      // parent here and claude-agent-sdk.ts's own query() span — lets a host
      // that wraps prompt() in its own span nest this trace under it.
      const parentCtx = context.active();
      const rootSpan = openRootSpan(tracer, parentCtx, {
        text: typeof text === 'string' ? text : undefined,
        sessionId: this.sessionId,
        captureContent: resolved.captureContent,
      });
      state.rootSpan = rootSpan;
      state.rootCtx = trace.setSpan(parentCtx, rootSpan);
      state.rootForceClosedBySweep = false;
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
        // claude-agent-sdk.ts parity (see its own endInFlight(), called from
        // wrapQuery's finish()): a rejection (or an early settle racing a
        // crashed attempt) can leave a tool/LLM span from the in-flight
        // attempt never closed by its own normal event — agent_end's own
        // defensive sweep never got a chance to run for that attempt.
        // Force-close those BEFORE ending the root so they still export
        // instead of being silently dropped forever (a span that never has
        // .end() called on it is never exported at all). includeRoot is
        // deliberately omitted: finalizeRootSpan below is THIS window's own,
        // correct root close.
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
      // still sees the original resolution/rejection unmodified — mirrors
      // the pre-existing pattern of using `result.catch(() => ...)` and
      // still returning `result` unchanged. Re-throwing from onReject is not
      // needed: attaching this handler is enough to mark `result`'s
      // rejection as handled for Node's unhandledRejection detection, and
      // the caller's own await/.then on the returned `result` still observes
      // the real rejection.
      //
      // Four DECIDED boundary policies:
      //  1. A run that bypasses prompt() entirely (e.g.
      //     sendCustomMessage({triggerTurn:true})) never opens a root at
      //     all — see agent_start below — so it is unaffected by this block.
      //  2. An early-return prompt() call handled synchronously within a
      //     single prompt() invocation (a handled "/command", an
      //     extension-runner 'input' hook that fully handles the call)
      //     resolves without agent_start ever firing: onResolve below still
      //     fires, so the root closes OK with no children — a trivial,
      //     childless span.
      //  3. A rejection (or the sync-throw path above) always finalizes the
      //     root ERROR, with the rejection reason recorded as an exception.
      //  4. A queue-only mid-stream steer/followUp call (isStreaming===true
      //     AND options.streamingBehavior set) is detected and returned
      //     BEFORE this point in the function is ever reached at all — see
      //     isQueueOnlySteer above — so it never opens a root, never reaches
      //     this finalize() closure, and never touches the ACTIVE run's own
      //     still-open root.
      // Policies 2 and 3 are both implemented by this single finalize() call
      // reached via either the try/catch above or the .then below.
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
    // steer()/followUp() never open a root span themselves — only
    // proto.prompt does. A run triggered purely by steer()/followUp() (no
    // enclosing prompt() call ever made) is therefore a BYPASS run under the
    // rootless boundary policy: see agent_start below. Only patched when
    // present as a function — defensive, like the dispose() patch below, so
    // a minimal/partial double never disables prompt instrumentation over a
    // missing, unrelated method.
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
        // agent_start but before agent_end, or even after agent_end but
        // before this window's prompt() promise has settled — this session's
        // SessionSpanState (if it was ever subscribed) can still have an
        // open rootSpan/llmSpan/toolSpans that will now never see their
        // normal close. Force-close them here first, exactly like
        // agent_start's own dangling-span sweep, so they still export
        // instead of leaking silently (see closeDanglingSpan in spans.ts on
        // why a never-.end()ed span is never exported) — then delegate to
        // the real dispose().
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
        // this window's prompt() promise already settled and closed
        // everything normally) hits nothing but the `if (state)` check
        // above, so this call into the real dispose() is unchanged: same
        // arguments, same return value, same timing as before this patch
        // existed.
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
    rootForceClosedBySweep: false,
  };
  // Reachable from AgentSession.prototype.dispose (patched once, in
  // instrumentPiCodingAgent() above) so a mid-run dispose() can force-close
  // whatever this session's listener callback below left open.
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
      //
      // sweepDanglingSpans() force-closes the tool spans and the LLM span
      // unconditionally — it never needs to gate on anything first. That
      // matters because a stray tool_execution_start OR a stray
      // message_start (with no matching message_end) can arrive from a
      // prior attempt with no clean handoff, leaving an orphaned entry.
      // Without sweeping each one unconditionally, that orphan would either
      // stay open forever (tool span) or have its reference silently
      // overwritten below without ever calling .end() on it (LLM span) — see
      // closeDanglingSpan in spans.ts on why a never-.end()ed span is never
      // exported.
      sweepDanglingSpans(state);
      state.llmSpan = undefined;
      state.llmCtx = undefined;
      // Boundary policy 1 (rootless bypass): if state.rootSpan is undefined
      // here, this run never went through the wrapped prompt() call at all
      // (e.g. sendCustomMessage({triggerTurn:true}), or a steer()/followUp()-
      // only session with no enclosing prompt()). Do NOT synthesize a root
      // for it — its LLM/tool children fall back to ROOT_CONTEXT below
      // (message_start / tool_execution_start's own fallback) and form a
      // parentless mini-trace instead of a fabricated, input-less AGENT span.
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
      closeDanglingSpan(state.llmSpan);
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
      // deliberately left untouched — turn_end isn't session end (nor even
      // attempt end).
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
      closeDanglingSpan(existing);
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
      // Defensive cleanup for any tool/LLM span this attempt left dangling,
      // mirroring turn_end's identical sweep — normally both are already
      // empty by the time agent_end fires. The root span is NOT part of this
      // sweep: agent_end no longer owns closing the root at all (that is
      // now proto.prompt's job, on the enclosing promise settling) — it only
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
      } else if (state.rootForceClosedBySweep) {
        // This attempt's root span is gone because a reentrant dispose() (a
        // host's own earlier-registered agent_end listener disposing the
        // session synchronously — see session-dispose.test.ts) force-closed
        // it before this handler got to run for the same event. This is the
        // ONLY reachable cause: proto.prompt's own OVERLAP SAFETY sweep
        // never sets this flag (see SessionSpanState.rootForceClosedBySweep
        // above), and proto.prompt always clears it back to false the
        // instant it opens a new window's root — so a second prompt() call
        // racing this one can never be the source of a true value seen here.
        // The real stamp can no longer happen (the root span is already
        // ended), so the exported AGENT span is a FORCE_CLOSED one missing
        // this attempt's output. Surface that rather than silently dropping
        // the completion data.
        console.warn(
          "[traceroot-pi] agent_end arrived after this run's root span was already force-closed " +
            'by a reentrant dispose(); the exported AGENT span is missing its final output/retry ' +
            'attributes.',
        );
      }
      // Observable retry attempts, superseding the old per-attempt
      // traceroot.pi.will_retry flag: stamped onto the root as
      // traceroot.pi.retry_count once the enclosing prompt() call's own
      // promise settles (see finalizeRootSpan in spans.ts).
      if (event.willRetry) {
        state.retryCount += 1;
      }
      break;
    }
    default:
      break;
  }
}
