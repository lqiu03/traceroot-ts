/**
 * A prompt() call that resolves successfully but never reaches agent_start
 * must not leave a stale entry in the pendingInput FIFO queue either — not
 * just the throw/reject case already covered by
 * rejected-prompt-fifo-leak.test.ts.
 *
 * Verified against the real, installed @earendil-works/pi-coding-agent@0.80.6
 * (dist/core/agent-session.js:812-824): when `this.isStreaming` is true and
 * `options.streamingBehavior` is set ('steer' or 'followUp'), prompt() calls
 * `_queueSteer`/`_queueFollowUp` — which inject the message into the
 * CURRENTLY-running Agent loop, not a new run — and returns. It never reaches
 * `_runAgentPrompt`, the only call site that leads to a fresh agent_start.
 * The promise resolves, not rejects.
 *
 * FakeAgentSession below faithfully mirrors that gate: `isStreaming` is a
 * real, mutable property (flipped by the test the same way the real SDK's
 * own isStreaming lifecycle would — set on a fresh run, cleared once
 * agent_end fires), and prompt() only "runs" (i.e. becomes the kind of call
 * that will eventually produce agent_start) when NOT already streaming, or
 * when streaming without a streamingBehavior option.
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

function makeFakeSessionClass() {
  return class FakeAgentSession {
    sessionId = 'sess-1';
    // Mutable, mirroring the real SDK's `get isStreaming()` — true while a
    // run is in flight (from the moment prompt() starts a fresh run until
    // the test emits agent_end and clears it back to false).
    isStreaming = false;
    private listeners: Array<(event: AgentEvent) => void> = [];

    async prompt(
      _text: string,
      options?: { streamingBehavior?: 'steer' | 'followUp' },
    ): Promise<void> {
      // Faithful mirror of the verified real gate (agent-session.js:812-824):
      // while streaming, a call with streamingBehavior set queues into the
      // ALREADY-running loop and resolves without ever starting a new run —
      // no agent_start will ever follow this particular call.
      if (this.isStreaming && options?.streamingBehavior) {
        return;
      }
      // A genuinely new run: the real SDK flips isStreaming on before the
      // agent loop's own agent_start event fires.
      this.isStreaming = true;
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

test("a steer()/followUp() call queued in during an active run (resolves, no agent_start) does not corrupt the NEXT run's input.value", async () => {
  const capture = new CapturingExporter();
  const Session = makeFakeSessionClass();
  const sdk = { AgentSession: Session };
  instrumentPiCodingAgent(sdk, { apiKey: 'k', _spanExporter: capture });
  const session = new Session();

  // Run 1 starts normally.
  await session.prompt('run 1 text');
  session.emit({ type: 'agent_start' });

  // While run 1 is still active, a steered-in follow-up resolves
  // successfully with NO new agent_start — the SDK's documented interactive
  // steering/follow-up pattern.
  await session.prompt('steered-in text', { streamingBehavior: 'steer' });

  // Run 1 ends.
  session.emit({
    type: 'agent_end',
    messages: [assistantMessage({ content: [{ type: 'text', text: 'reply 1' }] })],
    willRetry: false,
  });
  session.isStreaming = false;

  // A genuinely new run 2 starts on the same session.
  await session.prompt('run 2 text');
  session.emit({ type: 'agent_start' });
  session.emit({
    type: 'agent_end',
    messages: [assistantMessage({ content: [{ type: 'text', text: 'reply 2' }] })],
    willRetry: false,
  });

  const rootSpans = capture.spans.filter((s) => attrs(s)['openinference.span.kind'] === 'AGENT');
  assert.equal(rootSpans.length, 2, 'run 1 and run 2 each got exactly one root span');
  assert.equal(attrs(rootSpans[0]!)['input.value'], 'run 1 text');
  assert.equal(
    attrs(rootSpans[1]!)['input.value'],
    'run 2 text',
    'run 2 must be attributed with ITS OWN prompt text, not the stale text left behind by the ' +
      'earlier steer()/followUp() queue-in call that resolved without ever reaching agent_start',
  );
});
