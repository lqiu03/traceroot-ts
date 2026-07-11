/**
 * Lens: a third independent pass looking for an asymmetry between how
 * agent_start's dangling-span sweep treats state.toolSpans versus
 * state.llmSpan.
 *
 * agent_start's tool-span sweep is deliberately gated on
 * `state.toolSpans.size` alone, independent of `state.rootSpan` — the
 * in-source comment explains this is because a stray tool_execution_start
 * can arrive after a prior run's agent_end already cleared rootSpan (e.g.
 * an async tool callback resolving late), leaving an orphaned tool span
 * with no rootSpan to gate the sweep on.
 *
 * The exact same scenario can happen to state.llmSpan: a stray
 * message_start (assistant role, no matching message_end) can arrive after
 * agent_end already cleared rootSpan, opening an llmSpan parented under
 * ROOT_CONTEXT. But the llmSpan close is nested inside `if (state.rootSpan)`,
 * which is false in this scenario — so the orphaned llmSpan is never
 * force-closed. Its reference is then unconditionally overwritten
 * (`state.llmSpan = undefined`) without ever calling .end() on it, so the
 * span is silently dropped: OTel never records/exports a span that never
 * had .end() called on it.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import type { ExportResult } from '@opentelemetry/core';
import { ExportResultCode } from '@opentelemetry/core';
import type { ReadableSpan, SpanExporter } from '@opentelemetry/sdk-trace-base';
import { instrumentPiCodingAgent } from '../src/instrumentation';
import type { AgentEvent, AssistantMessage } from '../src/types';

// Copied locally per-file, matching every other tests/*.test.ts in this
// package — no shared module-level exporter/session state across files.
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
function makeRig() {
  const capture = new CapturingExporter();

  class FakeAgentSession {
    sessionId = 'sess-1';
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
  }

  const sdk = { AgentSession: FakeAgentSession };
  instrumentPiCodingAgent(sdk, { apiKey: 'test-key', _spanExporter: capture });

  return { capture, Session: FakeAgentSession };
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

test('a stray message_start firing after a clean agent_end (no matching message_end) is force-closed on the next agent_start instead of being silently dropped', async () => {
  const { capture, Session } = makeRig();
  const session = new Session();

  await session.prompt('run 1 finishes cleanly with no LLM turn at all');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });

  assert.equal(capture.spans.length, 1, 'only run 1s root span has exported so far');

  // A straggler assistant message_start arrives after agent_end already
  // tore the run down (e.g. a late stream event) — with rootCtx cleared,
  // it opens an LLM span parented under ROOT_CONTEXT. Its message_end never
  // arrives (the stream is already abandoned).
  session.emit({
    type: 'message_start',
    message: assistantMessage({ model: 'stray-orphaned-model' }),
  });

  // Run 2 starts fresh. This is the moment the orphaned llmSpan from the
  // straggler above must be force-closed and exported — mirroring exactly
  // how a stray tool_execution_start in the same position is already swept.
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'message_start', message: assistantMessage({ model: 'run-2-model' }) });
  session.emit({ type: 'message_end', message: assistantMessage({ model: 'run-2-model' }) });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });

  const strayLlmSpan = capture.spans.find(
    (s) => attrs(s)['gen_ai.request.model'] === 'stray-orphaned-model',
  );
  assert.ok(
    strayLlmSpan,
    'the orphaned LLM span from the straggler message_start must still be force-closed and ' +
      'exported, not silently dropped forever (a span that never has .end() called on it is ' +
      'never recorded/exported at all)',
  );
  assert.equal(
    attrs(strayLlmSpan!)['traceroot.pi.force_closed'],
    true,
    'it must be marked force_closed, distinguishing it from a normally-closed span',
  );

  const run2LlmSpan = capture.spans.find((s) => attrs(s)['gen_ai.request.model'] === 'run-2-model');
  assert.ok(run2LlmSpan, 'run 2 must still produce its own normal LLM span');
});
