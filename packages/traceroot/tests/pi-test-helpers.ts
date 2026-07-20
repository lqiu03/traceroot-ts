/**
 * Shared fixtures for packages/traceroot/tests/pi-*.test.ts.
 *
 * Moved from packages/pi/tests/test-helpers.ts as part of folding the
 * standalone @traceroot-ai/pi package's behavioral test suite into
 * @traceroot-ai/traceroot's own in-tree pi instrumentation
 * (packages/traceroot/src/pi.ts). The old rig injected a private span
 * exporter into pi's config (a `_spanExporter` field that no longer
 * exists on PiInstrumentationConfig — the in-tree integration never owns
 * its own TracerProvider). This version instead registers a REAL global
 * OTel provider per rig and lets instrumentPiCodingAgent() re-resolve the
 * tracer through the global `trace` facade, exactly like core wiring does
 * in production.
 *
 * ── prompt()'s returned promise settles on the FINAL agent_end, not eagerly ──
 * The root AGENT span is now anchored on the wrapped prompt() call's own
 * promise window (see src/pi.ts's module header) — verified
 * against the real SDK, prompt() awaits its whole internal retry/compaction/
 * follow-up loop, so every attempt's agent_end fires before prompt() itself
 * resolves. FakeAgentSession mirrors that here: prompt()'s returned promise
 * does NOT resolve just because it was called — it stays pending until
 * emit() observes an agent_end whose willRetry is not true (the attempt that
 * ends the whole call, not a mid-loop continuation), and rejects only via the
 * synchronous shouldReject path (a pre-flight validation failure, exactly
 * like the real SDK's early-return validation) or the explicit
 * rejectPrompt() escape hatch below (an async-path failure, e.g. an
 * unexpected internal error the agent loop surfaces as a rejection instead
 * of an agent_end).
 *
 * Because of this, tests MUST NOT `await session.prompt(text)` before
 * emitting the events that make up that call's run — doing so would
 * deadlock (or, for an early-return call, would need an explicit
 * resolvePrompt()/rejectPrompt() call with no agent_start ever emitted). The
 * established pattern across this suite is:
 *
 *   const done = session.prompt('text');   // NOT awaited yet
 *   session.emit({ type: 'agent_start' });
 *   ...
 *   session.emit({ type: 'agent_end', ..., willRetry: false });
 *   await done;                            // now resolves; root is finalized
 *
 * `await done` is required before asserting on `capture.spans` for the root
 * span: instrumentation.ts's proto.prompt finalizes (stamps retry_count,
 * sets status, ends) the root span in a `.then()` callback registered on the
 * SAME promise BEFORE it is returned to the caller, so that callback is
 * guaranteed to run (and, via SimpleSpanProcessor, export the ended span)
 * strictly before `await done`'s own continuation resumes.
 */
import { trace } from '@opentelemetry/api';
import type { ExportResult } from '@opentelemetry/core';
import { ExportResultCode } from '@opentelemetry/core';
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import { SimpleSpanProcessor } from '@opentelemetry/sdk-trace-base';
import type { ReadableSpan, SpanExporter } from '@opentelemetry/sdk-trace-base';
import {
  instrumentPiCodingAgent,
  type AgentEvent,
  type AssistantMessage,
  type PiInstrumentationConfig,
  type PromptOptions,
} from '../src/pi';

/**
 * Capturing exporter — wired into a real, freshly-registered global
 * TracerProvider per rig (see makeRig below) rather than the deleted
 * private-exporter injection path, to avoid OTLP network calls while still
 * exercising the exact same tracer-acquisition path production uses.
 */
export class CapturingExporter implements SpanExporter {
  readonly spans: ReadableSpan[] = [];
  export(spans: ReadableSpan[], resultCallback: (result: ExportResult) => void): void {
    this.spans.push(...spans);
    resultCallback({ code: ExportResultCode.SUCCESS });
  }
  async shutdown(): Promise<void> {}
}

/**
 * Returns a fresh FakeAgentSession class — never a shared module-level
 * class, since instrumentPiCodingAgent patches AgentSession.prototype
 * directly, so reusing one class across rigs/tests would stack multiple
 * wrap layers onto the same prototype method. Call this once per rig/test.
 *
 * @param shouldReject - optional predicate; when it returns true for the
 *   text passed to prompt(), prompt()'s RETURNED PROMISE rejects instead of
 *   resolving, before ever creating a pending settle-function entry. Because
 *   this prompt() is declared `async`, a `throw` inside it is captured into
 *   a REJECTED PROMISE, not a genuine synchronous throw — callers must
 *   `await`/`.catch()` it, exactly like any other async-function throw; the
 *   call to `session.prompt(text)` itself never throws. (A real synchronous
 *   throw — where the call to `session.prompt(...)` itself throws, before
 *   ever returning a promise at all — is a materially different case,
 *   exercising a different catch branch in instrumentation.ts's proto.prompt;
 *   see the `instrumentation edge cases` suite's dedicated synchronous-throw
 *   fake, which overrides prompt() with a plain non-async function instead of
 *   using this predicate.) Used to reproduce a prompt() call that fails
 *   validation before agent_start ever fires — the real SDK's own
 *   early-return validation path.
 */
export function makeFakeSessionClass(shouldReject?: (text: string) => boolean) {
  return class FakeAgentSession {
    sessionId = 'sess-1';
    disposed = false;
    // Mutable mirror of the real SDK's `AgentSession.prototype.isStreaming`
    // getter (see types.ts's AgentSessionInstance doc comment) — false/idle
    // by default. Tests that need to simulate a call arriving while a run is
    // already active (instrumentation.ts's isQueueOnlySteer check) set this
    // to true directly on the instance before calling prompt() again.
    isStreaming = false;
    private listeners: Array<(event: AgentEvent) => void> = [];
    // The currently in-flight prompt() call's own settle functions, or
    // undefined when no prompt() call is awaiting its final agent_end (or an
    // explicit resolvePrompt()/rejectPrompt()). See this file's module
    // header for why prompt() does not resolve merely by being called.
    private pending: { resolve: () => void; reject: (err: unknown) => void } | undefined;

    async prompt(text: string, options?: PromptOptions): Promise<void> {
      if (shouldReject?.(text)) {
        throw new Error(`validation failed for: ${text}`);
      }
      // Mirrors the real SDK's own queue-and-return shape (verified in
      // instrumentation.ts's isQueueOnlySteer comment): while a run is
      // already streaming, a caller-supplied streamingBehavior means this
      // call queues into the ACTIVE run and resolves immediately instead of
      // starting a new one. Deliberately does NOT touch `this.pending` here
      // — that belongs to whichever earlier prompt() call is still awaiting
      // the active run's own final agent_end; overwriting it would steal
      // that call's settle functions out from under it.
      if (this.isStreaming && options?.streamingBehavior) {
        return;
      }
      return new Promise<void>((resolve, reject) => {
        this.pending = { resolve, reject };
      });
    }
    subscribe(listener: (event: AgentEvent) => void): () => void {
      this.listeners.push(listener);
      return () => {
        this.listeners = this.listeners.filter((l) => l !== listener);
      };
    }
    emit(event: AgentEvent): void {
      // Every registered listener (instrumentation's own, plus any
      // test-installed one) runs FIRST, synchronously — exactly mirroring
      // the real SDK, where every attempt's agent_end handler completes
      // before prompt()'s own promise resolves. Only after that do we check
      // whether this event is the one that ends the whole prompt() call.
      for (const listener of this.listeners) listener(event);
      if (event.type === 'agent_end' && !event.willRetry && this.pending) {
        const { resolve } = this.pending;
        this.pending = undefined;
        resolve();
      }
    }
    // Manually settles the current prompt() call's promise as a SUCCESS with
    // no agent_end ever having fired — the async-path mirror of an
    // early-return prompt() call (a handled "/command", a queue-only
    // steer/followUp early return): the real SDK resolves without ever
    // reaching _runAgentPrompt, so no agent_start/agent_end follows.
    resolvePrompt(): void {
      if (!this.pending) return;
      const { resolve } = this.pending;
      this.pending = undefined;
      resolve();
    }
    // Manually rejects the current prompt() call's promise — the async-path
    // mirror of an unexpected internal error the agent loop surfaces as a
    // rejection rather than an agent_end. Distinct from the constructor's
    // shouldReject, which models a SYNCHRONOUS pre-flight validation failure
    // before the agent loop (and this pending promise) ever exists.
    rejectPrompt(err: unknown): void {
      if (!this.pending) return;
      const { reject } = this.pending;
      this.pending = undefined;
      reject(err);
    }
    // dispose() mirrors the real, verified SDK mechanism (see the
    // `session dispose` suite's header): it reassigns the session's
    // internal listener array to a fresh empty one rather than calling each
    // stored unsubscribe() closure. Included unconditionally (not gated
    // behind an options flag) because every other caller of this factory
    // never invokes .dispose(), so its presence is inert for them — only
    // the `session dispose` suite exercises it.
    dispose(): void {
      this.disposed = true;
      this.listeners = [];
    }
  };
}

/**
 * Convenience wrapper around makeFakeSessionClass() for the common case: a
 * fresh CapturingExporter wired into a fresh, freshly-registered global
 * TracerProvider, plus a fresh FakeAgentSession class, already wired
 * through instrumentPiCodingAgent(). Extra config (e.g. captureContent,
 * captureToolIo) passes straight through.
 *
 * trace.disable() runs FIRST, unconditionally, on every call — this is
 * LOAD-BEARING for per-test isolation. The in-tree instrumentation never
 * builds its own TracerProvider; it always re-resolves the tracer through
 * the OTel API's global `trace` facade (see
 * packages/traceroot/src/pi.ts's createReresolvingTracer).
 * Without disabling the previous rig's registration first, a later
 * provider.register() call would layer a second global provider on top
 * of / behind the first (OTel's global registration is last-write-wins but
 * does not automatically detach the prior provider's own state), and — far
 * more importantly — a fresh FakeAgentSession class per rig only prevents
 * *prototype*-level double-wrapping; it does nothing about the *tracer*
 * each already-instrumented session resolves at span-open time. Disabling
 * first guarantees each rig's spans land only in that rig's own
 * CapturingExporter.
 */
export function makeRig(config: PiInstrumentationConfig = {}): {
  capture: CapturingExporter;
  Session: ReturnType<typeof makeFakeSessionClass>;
} {
  trace.disable();
  const capture = new CapturingExporter();
  const provider = new NodeTracerProvider();
  // SimpleSpanProcessor exports synchronously on span.end() (no batching
  // delay), matching every test in this suite's assumption that a span is
  // already present in capture.spans immediately after the event that
  // closes it.
  provider.addSpanProcessor(new SimpleSpanProcessor(capture));
  provider.register();
  const Session = makeFakeSessionClass();
  const sdk = { AgentSession: Session };
  instrumentPiCodingAgent(sdk, config);
  return { capture, Session };
}

/**
 * Minimal placeholder usage/cost numbers — most tests only care that the
 * fields exist on the span, not their exact value. Tests that assert on
 * specific usage/cost figures (e.g. instrumentation.test.ts,
 * spans-truncation.test.ts) must pass an explicit `usage` override rather
 * than relying on this default, so the numbers they depend on stay visible
 * at the call site instead of drifting silently between files.
 */
export function assistantMessage(overrides: Partial<AssistantMessage> = {}): AssistantMessage {
  return {
    role: 'assistant',
    content: [{ type: 'text', text: 'done' }],
    api: 'anthropic-messages',
    provider: 'anthropic',
    model: 'claude-sonnet-5',
    usage: {
      input: 1,
      output: 1,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 2,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
    stopReason: 'stop',
    timestamp: 0,
    ...overrides,
  } as AssistantMessage;
}

export function attrs(span: ReadableSpan): Record<string, unknown> {
  return span.attributes as Record<string, unknown>;
}
