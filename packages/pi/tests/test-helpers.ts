/**
 * Shared fixtures for packages/pi/tests/*.test.ts.
 *
 * Extracted from ~14 near-identical per-file copies of the same
 * CapturingExporter / FakeAgentSession / assistantMessage() / attrs()
 * fixtures — see the code-review finding this file resolves. A single
 * source of truth here means an OTel SpanExporter/ReadableSpan shape change
 * on an SDK bump, or a new AgentSessionInstance method, only has to be
 * hand-applied once, and default fixture values (e.g. usage/cost numbers)
 * can no longer silently drift between files the way they had.
 */
import type { ExportResult } from '@opentelemetry/core';
import { ExportResultCode } from '@opentelemetry/core';
import type { ReadableSpan, SpanExporter } from '@opentelemetry/sdk-trace-base';
import type { PiInstrumentationConfig } from '../src/config';
import { instrumentPiCodingAgent } from '../src/instrumentation';
import type { AgentEvent, AssistantMessage } from '../src/types';

/**
 * Capturing exporter — injected via _spanExporter to avoid OTLP network calls,
 * matching packages/mastra/tests/exporter-path.test.ts's convention.
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
 * fresh CapturingExporter plus a fresh FakeAgentSession class, already
 * wired through instrumentPiCodingAgent(). Extra config (e.g.
 * captureContent, captureToolIo) passes straight through.
 */
export function makeRig(config: Omit<PiInstrumentationConfig, 'apiKey' | '_spanExporter'> = {}): {
  capture: CapturingExporter;
  Session: ReturnType<typeof makeFakeSessionClass>;
} {
  const capture = new CapturingExporter();
  const Session = makeFakeSessionClass();
  const sdk = { AgentSession: Session };
  // instrumentPiCodingAgent() always runs in private mode here (a rig-local
  // apiKey/_spanExporter, never a shared global provider), so every call
  // registers its own process.once('beforeExit', flushOnExit) hook (see
  // instrumentation.ts) and never removes it on a normal, successful
  // install -- only a mid-setup failure rolls it back. That is a deliberate
  // per-process-lifetime design for real hosts (each independently
  // instrumented SDK object owns its own private OTLP pipeline and needs its
  // own flush hook), but this rig is called fresh, in private mode, from
  // every single test across ~10 files in this directory -- so without
  // cleanup here, the process accumulates one live 'beforeExit' listener per
  // test for the rest of the file's run, tripping Node's
  // MaxListenersExceededWarning well before a consolidated file (e.g.
  // spans-truncation.test.ts's 11 tests) finishes. No test in this package
  // depends on that listener ever firing: everything here observes spans via
  // CapturingExporter's own export() callback (invoked by the
  // BatchSpanProcessor on span end), never via the beforeExit flush path
  // (that path is exercised directly, with its own explicit cleanup, only by
  // provider-shared-mode-behavior.test.ts). It is therefore safe to strip
  // whatever this call added immediately after setup rather than leaving it
  // live for the remainder of the process. Diffing
  // process.listeners('beforeExit') before/after -- rather than assuming
  // exactly one listener was added -- matches this package's own established
  // idiom (see provider-shared-mode-behavior.test.ts) and stays correct even
  // if a future config combination changes how many hooks a single call
  // registers.
  const beforeExitListenersBeforeSetup = new Set(process.listeners('beforeExit'));
  instrumentPiCodingAgent(sdk, { apiKey: 'test-key', _spanExporter: capture, ...config });
  for (const listener of process.listeners('beforeExit')) {
    if (!beforeExitListenersBeforeSetup.has(listener)) {
      process.removeListener('beforeExit', listener);
    }
  }
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
