/**
 * Shared fixtures for packages/traceroot/tests/pi-*.test.ts.
 *
 * Moved from packages/pi/tests/test-helpers.ts as part of folding the
 * standalone @traceroot-ai/pi package's behavioral test suite into
 * @traceroot-ai/traceroot's own in-tree pi instrumentation
 * (packages/traceroot/src/pi/*). The old rig injected a private span
 * exporter into pi's config (a `_spanExporter` field that no longer
 * exists on PiInstrumentationConfig — the in-tree integration never owns
 * its own TracerProvider). This version instead registers a REAL global
 * OTel provider per rig and lets instrumentPiCodingAgent() re-resolve the
 * tracer through the global `trace` facade, exactly like core wiring does
 * in production.
 */
import { trace } from '@opentelemetry/api';
import type { ExportResult } from '@opentelemetry/core';
import { ExportResultCode } from '@opentelemetry/core';
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import { SimpleSpanProcessor } from '@opentelemetry/sdk-trace-base';
import type { ReadableSpan, SpanExporter } from '@opentelemetry/sdk-trace-base';
import type { PiInstrumentationConfig } from '../src/pi/config';
import { instrumentPiCodingAgent } from '../src/pi/instrumentation';
import type { AgentEvent, AssistantMessage } from '../src/pi/types';

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
 *   text passed to prompt(), prompt() rejects instead of resolving. Used to
 *   reproduce a prompt() call that fails validation before agent_start
 *   ever fires.
 */
export function makeFakeSessionClass(shouldReject?: (text: string) => boolean) {
  return class FakeAgentSession {
    sessionId = 'sess-1';
    disposed = false;
    private listeners: Array<(event: AgentEvent) => void> = [];
    async prompt(text: string, _options?: unknown): Promise<void> {
      if (shouldReject?.(text)) {
        throw new Error(`validation failed for: ${text}`);
      }
    }
    subscribe(listener: (event: AgentEvent) => void): () => void {
      this.listeners.push(listener);
      return () => {
        this.listeners = this.listeners.filter((l) => l !== listener);
      };
    }
    emit(event: AgentEvent): void {
      for (const listener of this.listeners) listener(event);
    }
    // dispose() mirrors the real, verified SDK mechanism (see
    // session-dispose.test.ts's module header): it reassigns the session's
    // internal listener array to a fresh empty one rather than calling each
    // stored unsubscribe() closure. Included unconditionally (not gated
    // behind an options flag) because every other caller of this factory
    // never invokes .dispose(), so its presence is inert for them — only
    // session-dispose.test.ts exercises it.
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
 * packages/traceroot/src/pi/instrumentation.ts's createReresolvingTracer).
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
