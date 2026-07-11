/**
 * Verifies AgentSession.dispose()'s real, confirmed contract (see
 * instrumentation.ts's module header and types.ts's AgentSessionInstance
 * doc comment for the full verified call chain, read directly out of the
 * real, installed @earendil-works/pi-coding-agent@0.80.6 dist/core/
 * agent-session.js): dispose() clears every listener registered via
 * subscribe() — including instrumentPiCodingAgent()'s own — by reassigning
 * the session's internal listener array to a fresh empty one, with no
 * per-listener unsubscribe() call required from either side.
 *
 * FakeAgentSession here intentionally mirrors that exact mechanism (push on
 * subscribe, splice via the returned closure, reassign to [] on dispose) so
 * this test exercises the same shape the real SDK does, not a simplified
 * stand-in that would pass for the wrong reason.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import type { ExportResult } from '@opentelemetry/core';
import { ExportResultCode } from '@opentelemetry/core';
import type { ReadableSpan, SpanExporter } from '@opentelemetry/sdk-trace-base';
import { instrumentPiCodingAgent } from '../src/instrumentation';
import type { AgentEvent, AssistantMessage } from '../src/types';

// Copied locally per-file, matching every other tests/*.test.ts in this package.
class CapturingExporter implements SpanExporter {
  readonly spans: ReadableSpan[] = [];
  export(spans: ReadableSpan[], resultCallback: (result: ExportResult) => void): void {
    this.spans.push(...spans);
    resultCallback({ code: ExportResultCode.SUCCESS });
  }
  async shutdown(): Promise<void> {}
}

// Fresh class per rig, not a shared module-level class — instrumentPiCodingAgent
// patches AgentSession.prototype directly, so reusing one class across tests
// would stack multiple wrap layers onto the same prototype method.
//
// dispose() below reassigns `this.listeners` to a brand-new array rather
// than calling each stored unsubscribe closure — that is the actual,
// verified real-SDK mechanism (this._eventListeners = []), not a
// simplification.
function makeFakeSessionClass() {
  return class FakeAgentSession {
    sessionId = 'sess-1';
    disposed = false;
    private listeners: Array<(event: AgentEvent) => void> = [];
    async prompt(_text: string, _options?: unknown): Promise<void> {}
    subscribe(listener: (event: AgentEvent) => void): () => void {
      this.listeners.push(listener);
      return () => {
        this.listeners = this.listeners.filter((l) => l !== listener);
      };
    }
    emit(event: AgentEvent): void {
      for (const listener of this.listeners) listener(event);
    }
    dispose(): void {
      this.disposed = true;
      this.listeners = [];
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

test('session.dispose() does not throw and requires no extra cleanup call from instrumentPiCodingAgent()', async () => {
  const capture = new CapturingExporter();
  const Session = makeFakeSessionClass();
  const sdk = { AgentSession: Session };
  instrumentPiCodingAgent(sdk, { apiKey: 'k', _spanExporter: capture });
  const session = new Session();

  // Drive one full, cleanly-closed turn so instrumentPiCodingAgent()'s
  // subscribe() listener is actually registered and has produced a span,
  // matching real usage instead of disposing an untouched session.
  await session.prompt('do something');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
  assert.equal(capture.spans.length, 1, 'the normal run must still produce its span');

  assert.doesNotThrow(() => {
    session.dispose();
  }, 'dispose() must be safe to call even though instrumentPiCodingAgent() never captured or called the subscribe() unsubscribe function itself');
  assert.equal(session.disposed, true);

  // instrumentPiCodingAgent() never stored or invoked the unsubscribe
  // function subscribe() returned — it relies entirely on dispose() clearing
  // the SDK's own listener array. Firing more events post-dispose (as a
  // buggy or unusual host might) must produce no further spans, proving
  // instrumentation.ts needs no dispose-time hook of its own: the host
  // session's own dispose() is sufficient to stop delivery on its own.
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
  assert.equal(
    capture.spans.length,
    1,
    'no new spans may appear after dispose() — the listener must no longer be reachable',
  );
});

test('dispose() on a session that never had prompt() called (no traceroot-pi subscription registered yet) is still safe', async () => {
  const capture = new CapturingExporter();
  const Session = makeFakeSessionClass();
  const sdk = { AgentSession: Session };
  instrumentPiCodingAgent(sdk, { apiKey: 'k', _spanExporter: capture });
  const session = new Session();

  assert.doesNotThrow(() => {
    session.dispose();
  });
  assert.equal(session.disposed, true);
  assert.equal(capture.spans.length, 0);
});
