/**
 * A prompt() call that rejects (or throws) BEFORE its corresponding
 * agent_start ever fires must not leave a stale entry in the pendingInput
 * FIFO queue — see instrumentation-edge-cases.test.ts's existing
 * "a rejected prompt() ... never creates a dangling root span" test, which
 * proves the rejected call produces no span but never checks whether its
 * queued text leaks forward into a LATER, unrelated, successful prompt()
 * call on the same session.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import type { ExportResult } from '@opentelemetry/core';
import { ExportResultCode } from '@opentelemetry/core';
import type { ReadableSpan, SpanExporter } from '@opentelemetry/sdk-trace-base';
import { instrumentPiCodingAgent } from '../src/instrumentation';
import type { AgentEvent, AssistantMessage } from '../src/types';

class CapturingExporter implements SpanExporter {
  readonly spans: ReadableSpan[] = [];
  export(spans: ReadableSpan[], resultCallback: (result: ExportResult) => void): void {
    this.spans.push(...spans);
    resultCallback({ code: ExportResultCode.SUCCESS });
  }
  async shutdown(): Promise<void> {}
}

function makeFakeSessionClass(shouldReject: (text: string) => boolean) {
  return class FakeAgentSession {
    sessionId = 'sess-1';
    private listeners: Array<(event: AgentEvent) => void> = [];
    async prompt(text: string, _options?: unknown): Promise<void> {
      if (shouldReject(text)) {
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

function assistantMessage(overrides: Partial<AssistantMessage> = {}): AssistantMessage {
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

function attrs(span: ReadableSpan): Record<string, unknown> {
  return span.attributes as Record<string, unknown>;
}

test('a prompt() call that rejects before agent_start does not leak its stale text into a LATER successful prompt() root span', async () => {
  const capture = new CapturingExporter();
  const Session = makeFakeSessionClass((text) => text === 'bad text that fails validation');
  const sdk = { AgentSession: Session };
  instrumentPiCodingAgent(sdk, { apiKey: 'k', _spanExporter: capture });
  const session = new Session();

  // First call: rejects before agent_start ever fires (e.g. a validation
  // failure inside Pi's own prompt() implementation).
  await assert.rejects(() => session.prompt('bad text that fails validation'));
  assert.equal(capture.spans.length, 0, 'the rejected call must not itself produce a span');

  // Second call: a genuine, unrelated, successful prompt on the SAME
  // session instance. This is the realistic case — a host app just retries
  // with different input, or a user submits a new message after a failed one.
  await session.prompt('good text that should actually run');
  session.emit({ type: 'agent_start' });
  session.emit({
    type: 'agent_end',
    messages: [assistantMessage({ content: [{ type: 'text', text: 'reply' }] })],
    willRetry: false,
  });

  const rootSpans = capture.spans.filter((s) => attrs(s)['openinference.span.kind'] === 'AGENT');
  assert.equal(rootSpans.length, 1, 'exactly one real run happened');
  assert.equal(
    attrs(rootSpans[0]!)['input.value'],
    'good text that should actually run',
    'the successful run must be attributed with ITS OWN prompt text, not the stale text left ' +
      'behind by the earlier rejected prompt() call that never reached agent_start',
  );
});
