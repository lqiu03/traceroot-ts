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
  instrumentPiCodingAgent(sdk, { apiKey: 'test-key', _spanExporter: capture, ...config });
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
