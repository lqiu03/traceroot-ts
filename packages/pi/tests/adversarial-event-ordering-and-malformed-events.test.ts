/**
 * Lens: event-ordering-and-malformed-events.
 *
 * Probes out-of-order / malformed AgentEvent sequences that a real Pi run
 * should never produce in the happy path, but that a buggy or aborted agent
 * loop plausibly could: a turn that ends without its message ever closing,
 * a tool call fired with no assistant message in front of it, a stray
 * agent_end with nothing open, two agent_starts back to back with no
 * agent_end between them, and events that arrive after a run has already
 * been torn down.
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

test('turn_end firing without a preceding message_end force-closes the LLM span instead of leaking it into the next turn', async () => {
  const { capture, Session } = makeRig();
  const session = new Session();

  await session.prompt('a stream error truncates turn 1 before message_end fires');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'message_start', message: assistantMessage({ model: 'turn-1-model' }) });
  // No message_end for turn 1 — simulates a stream/abort cutting the turn
  // short. turn_end still fires because the agent loop always emits it.
  session.emit({ type: 'turn_end', message: assistantMessage(), toolResults: [] });
  // Turn 2 proceeds normally.
  session.emit({ type: 'message_start', message: assistantMessage({ model: 'turn-2-model' }) });
  session.emit({ type: 'message_end', message: assistantMessage({ model: 'turn-2-model' }) });
  session.emit({ type: 'turn_end', message: assistantMessage(), toolResults: [] });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });

  const llmSpans = capture.spans.filter((s) => attrs(s)['openinference.span.kind'] === 'LLM');
  assert.equal(
    llmSpans.length,
    2,
    'turn 1s LLM span must still be exported (force-closed at turn_end), not silently dropped when turn 2s message_start overwrites the state reference',
  );
  const turn1Span = llmSpans.find((s) => attrs(s)['gen_ai.request.model'] === 'turn-1-model');
  assert.ok(turn1Span, 'the force-closed turn-1 span must retain its original attributes');
  assert.equal(
    attrs(turn1Span!)['traceroot.pi.force_closed'],
    true,
    'must be marked as abnormally closed',
  );
});

test('tool_execution_start with no prior message_start/message_end at all parents directly under the root span', async () => {
  const { capture, Session } = makeRig();
  const session = new Session();

  await session.prompt('agent invokes a tool with no assistant message event in front of it');
  session.emit({ type: 'agent_start' });
  session.emit({
    type: 'tool_execution_start',
    toolCallId: 't1',
    toolName: 'bash',
    args: { command: 'pwd' },
  });
  session.emit({
    type: 'tool_execution_end',
    toolCallId: 't1',
    toolName: 'bash',
    result: { content: [] },
    isError: false,
  });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });

  assert.equal(capture.spans.length, 2, 'root + tool span only — no LLM span was ever opened');
  const rootSpan = capture.spans.find((s) => attrs(s)['openinference.span.kind'] === 'AGENT');
  const toolSpan = capture.spans.find((s) => attrs(s)['gen_ai.tool.call.id'] === 't1');
  assert.ok(rootSpan);
  assert.ok(toolSpan);
  assert.equal(
    toolSpan!.parentSpanId,
    rootSpan!.spanContext().spanId,
    'with no LLM span open, the tool span must fall back to parenting directly under root, not be orphaned',
  );
});

test('agent_end with zero prior events (no agent_start ever fired) does not throw and produces no spans', async () => {
  const { capture, Session } = makeRig();
  const session = new Session();

  await session.prompt('nothing has happened yet on this session');
  assert.doesNotThrow(() => {
    session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
  });

  assert.equal(capture.spans.length, 0, 'there was no root span to close, so nothing is exported');
});

test('agent_start firing twice with no intervening agent_end force-closes the abandoned run instead of leaking its spans or its context into the new run', async () => {
  const { capture, Session } = makeRig();
  const session = new Session();

  await session.prompt('run 1 gets abandoned mid-flight (e.g. loop restart), run 2 starts cold');
  session.emit({ type: 'agent_start' }); // run 1
  session.emit({ type: 'message_start', message: assistantMessage({ model: 'run-1-model' }) });
  // Run 1 never reaches message_end/turn_end/agent_end — the loop is
  // restarted and immediately emits a second agent_start.
  session.emit({ type: 'agent_start' }); // run 2, no agent_end for run 1 in between
  session.emit({
    type: 'tool_execution_start',
    toolCallId: 'run2-tool',
    toolName: 'bash',
    args: { command: 'echo run2' },
  });
  session.emit({
    type: 'tool_execution_end',
    toolCallId: 'run2-tool',
    toolName: 'bash',
    result: {},
    isError: false,
  });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });

  const rootSpans = capture.spans.filter((s) => attrs(s)['openinference.span.kind'] === 'AGENT');
  assert.equal(
    rootSpans.length,
    2,
    'run 1s abandoned root span must be force-closed and exported, not leaked forever unclosed',
  );

  const run1LlmSpan = capture.spans.find((s) => attrs(s)['gen_ai.request.model'] === 'run-1-model');
  assert.ok(run1LlmSpan, 'run 1s dangling LLM span must also be force-closed, not leaked');

  const run2Tool = capture.spans.find((s) => attrs(s)['gen_ai.tool.call.id'] === 'run2-tool');
  assert.ok(run2Tool);
  // Identify run 1's (force-closed) root unambiguously: it's whichever root
  // span is the parent of run 1's LLM span.
  const run1Root = rootSpans.find((s) => s.spanContext().spanId === run1LlmSpan!.parentSpanId);
  assert.ok(run1Root, 'run 1s LLM span must be parented under run 1s (force-closed) root span');
  const run2Root = rootSpans.find((s) => s !== run1Root);
  assert.ok(run2Root);
  // The critical assertion: run 2's tool call must parent under run 2's own
  // root span. If run 1's stale llmCtx were not cleared when run 2 started,
  // this tool span would incorrectly parent under run 1's already-abandoned
  // LLM span — a stale-context leak forward into the new run.
  assert.equal(
    run2Tool!.parentSpanId,
    run2Root!.spanContext().spanId,
    "run 2's tool span must parent under run 2's root, never under run 1's stale/abandoned context",
  );
  assert.notEqual(
    run2Tool!.spanContext().traceId,
    run1Root!.spanContext().traceId,
    "run 2's tool span must live in a different trace than the abandoned run 1",
  );
});

test('a stray tool_execution_start firing after agent_end does not corrupt the next runs span tree', async () => {
  const { capture, Session } = makeRig();
  const session = new Session();

  await session.prompt('run 1 finishes cleanly');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
  // run 1's root span is the only span exported so far — capture its id
  // before run 2 starts so we can unambiguously tell the two roots apart
  // (both carry identical attributes otherwise).
  assert.equal(capture.spans.length, 1);
  const run1RootSpanId = capture.spans[0]!.spanContext().spanId;

  // A straggler event arrives after agent_end — e.g. an async tool runner's
  // callback that resolves after the agent loop already tore the run down.
  assert.doesNotThrow(() => {
    session.emit({
      type: 'tool_execution_start',
      toolCallId: 'straggler',
      toolName: 'bash',
      args: { command: 'echo late' },
    });
  });

  // Run 2 starts fresh.
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'message_start', message: assistantMessage() });
  session.emit({ type: 'message_end', message: assistantMessage() });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });

  const run2LlmSpan = capture.spans.find((s) => attrs(s)['openinference.span.kind'] === 'LLM');
  assert.ok(run2LlmSpan, 'run 2 must still produce a normal LLM span');
  const run2Root = capture.spans.find(
    (s) =>
      attrs(s)['openinference.span.kind'] === 'AGENT' && s.spanContext().spanId !== run1RootSpanId,
  );
  assert.ok(run2Root, 'run 2 must still produce its own root span, distinct from run 1s');
  assert.equal(
    run2LlmSpan!.parentSpanId,
    run2Root!.spanContext().spanId,
    "the straggler event must not have attached itself as run 2's parent context",
  );

  // The straggler tool span is eventually swept up (force-closed) by run 2's
  // agent_end cleanup since it was never explicitly ended — but it must not
  // be mistaken for a child of either run's root: it opened while no root
  // context was active, so it has no parent at all.
  const stragglerSpan = capture.spans.find((s) => attrs(s)['gen_ai.tool.call.id'] === 'straggler');
  assert.ok(stragglerSpan, 'the straggler span is still force-closed eventually, not dropped');
  assert.notEqual(stragglerSpan!.parentSpanId, run2Root!.spanContext().spanId);
  assert.notEqual(stragglerSpan!.parentSpanId, run1RootSpanId);
});
