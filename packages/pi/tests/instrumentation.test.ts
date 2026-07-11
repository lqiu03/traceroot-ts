import assert from 'node:assert/strict';
import { test } from 'node:test';
import type { ExportResult } from '@opentelemetry/core';
import { ExportResultCode } from '@opentelemetry/core';
import type { ReadableSpan, SpanExporter } from '@opentelemetry/sdk-trace-base';
import { instrumentPiCodingAgent } from '../src/instrumentation';
import type { AgentEvent, AssistantMessage } from '../src/types';

// Capturing exporter — injected via _spanExporter to avoid OTLP network calls,
// matching packages/mastra/tests/exporter-path.test.ts's convention.
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
function makeRig(config: { captureContent?: boolean; captureToolIo?: boolean } = {}) {
  const capture = new CapturingExporter();

  class FakeAgentSession {
    sessionId = 'sess-1';
    private listeners: Array<(event: AgentEvent) => void> = [];
    async prompt(_text: string, _options?: unknown): Promise<void> {
      // Real prompt() runs the agent loop; tests drive the resulting events manually.
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
  }

  const sdk = { AgentSession: FakeAgentSession };
  instrumentPiCodingAgent(sdk, { apiKey: 'test-key', _spanExporter: capture, ...config });

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
      input: 100,
      output: 20,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 120,
      cost: { input: 0.001, output: 0.0006, cacheRead: 0, cacheWrite: 0, total: 0.0016 },
    },
    stopReason: 'stop',
    timestamp: 0,
    ...overrides,
  } as AssistantMessage;
}

function attrs(span: ReadableSpan): Record<string, unknown> {
  return span.attributes as Record<string, unknown>;
}

test('a full turn with one tool call produces a correctly nested AGENT -> LLM -> TOOL span tree', async () => {
  const { capture, Session } = makeRig();
  const session = new Session();

  await session.prompt('list files in /tmp');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'turn_start' });
  session.emit({ type: 'message_start', message: assistantMessage() });
  session.emit({
    type: 'message_end',
    message: assistantMessage({
      content: [
        { type: 'text', text: "I'll list the files now." },
        { type: 'toolCall', id: 't1', name: 'bash' },
      ],
    }),
  });
  session.emit({
    type: 'tool_execution_start',
    toolCallId: 't1',
    toolName: 'bash',
    args: { command: 'ls /tmp' },
  });
  session.emit({
    type: 'tool_execution_end',
    toolCallId: 't1',
    toolName: 'bash',
    result: { content: [{ type: 'text', text: 'a.txt' }] },
    isError: false,
  });
  session.emit({
    type: 'turn_end',
    message: assistantMessage(),
    toolResults: [],
  });
  session.emit({
    type: 'agent_end',
    messages: [assistantMessage({ content: [{ type: 'text', text: 'listed the files' }] })],
    willRetry: false,
  });

  assert.equal(capture.spans.length, 3, 'expected exactly root, LLM, and tool spans');

  const [llmSpan, toolSpan, rootSpan] = capture.spans;

  assert.equal(rootSpan.name, 'AgentSession.prompt');
  assert.equal(attrs(rootSpan)['openinference.span.kind'], 'AGENT');
  assert.equal(attrs(rootSpan)['session.id'], 'sess-1');
  assert.equal(attrs(rootSpan)['traceroot.sdk.name'], 'traceroot-pi');
  assert.equal(attrs(rootSpan)['input.value'], 'list files in /tmp');
  assert.equal(attrs(rootSpan)['output.value'], 'listed the files');
  assert.equal(attrs(rootSpan)['traceroot.pi.will_retry'], false);

  assert.equal(attrs(llmSpan)['openinference.span.kind'], 'LLM');
  assert.equal(attrs(llmSpan)['gen_ai.system'], 'anthropic');
  assert.equal(attrs(llmSpan)['gen_ai.request.model'], 'claude-sonnet-5');
  assert.equal(attrs(llmSpan)['gen_ai.usage.input_tokens'], 100);
  assert.equal(attrs(llmSpan)['gen_ai.usage.output_tokens'], 20);
  assert.equal(attrs(llmSpan)['traceroot.pi.cost.total'], 0.0016);
  assert.equal(
    attrs(llmSpan)['output.value'],
    "I'll list the files now.",
    'captureContent:true (the default) must populate output.value on the LLM span too, not just the root span',
  );
  assert.equal(
    llmSpan.parentSpanId,
    rootSpan.spanContext().spanId,
    'LLM span must be a child of the root span',
  );

  assert.equal(toolSpan.name, 'bash: ls /tmp');
  assert.equal(attrs(toolSpan)['openinference.span.kind'], 'TOOL');
  assert.equal(attrs(toolSpan)['gen_ai.tool.name'], 'bash');
  assert.equal(attrs(toolSpan)['gen_ai.tool.call.id'], 't1');
  assert.equal(
    toolSpan.parentSpanId,
    llmSpan.spanContext().spanId,
    'tool span must be a child of the LLM span that requested it, not a sibling under root',
  );
});

test('two concurrent tool calls in one turn each get their own correctly-keyed span', async () => {
  const { capture, Session } = makeRig();
  const session = new Session();

  await session.prompt('do two things');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'message_start', message: assistantMessage() });
  session.emit({ type: 'message_end', message: assistantMessage() });
  session.emit({
    type: 'tool_execution_start',
    toolCallId: 'a',
    toolName: 'read',
    args: { path: '/x.txt' },
  });
  session.emit({
    type: 'tool_execution_start',
    toolCallId: 'b',
    toolName: 'read',
    args: { path: '/y.txt' },
  });
  session.emit({
    type: 'tool_execution_end',
    toolCallId: 'b',
    toolName: 'read',
    result: {},
    isError: false,
  });
  session.emit({
    type: 'tool_execution_end',
    toolCallId: 'a',
    toolName: 'read',
    result: {},
    isError: false,
  });
  session.emit({ type: 'turn_end', message: assistantMessage(), toolResults: [] });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });

  const toolSpans = capture.spans.filter(
    (s) => (s.attributes as Record<string, unknown>)['gen_ai.tool.call.id'],
  );
  assert.equal(toolSpans.length, 2);
  const ids = toolSpans
    .map((s) => (s.attributes as Record<string, unknown>)['gen_ai.tool.call.id'])
    .sort();
  assert.deepEqual(ids, ['a', 'b']);
  assert.equal(toolSpans[0]!.name, 'read: y.txt', 'b ended first, so it should export first');
  assert.equal(toolSpans[1]!.name, 'read: x.txt');
});

test('a failed LLM turn (stopReason error) marks the LLM span as ERROR', async () => {
  const { capture, Session } = makeRig();
  const session = new Session();

  await session.prompt('trigger a provider error');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'message_start', message: assistantMessage() });
  session.emit({
    type: 'message_end',
    message: assistantMessage({ stopReason: 'error', errorMessage: 'rate limited' }),
  });
  session.emit({ type: 'turn_end', message: assistantMessage(), toolResults: [] });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });

  const llmSpan = capture.spans.find(
    (s) => (s.attributes as Record<string, unknown>)['openinference.span.kind'] === 'LLM',
  );
  assert.ok(llmSpan);
  assert.equal(llmSpan!.status.code, 2 /* SpanStatusCode.ERROR */);
  assert.equal(llmSpan!.status.message, 'rate limited');
});

test('a failed tool call (isError) marks the TOOL span as ERROR', async () => {
  const { capture, Session } = makeRig();
  const session = new Session();

  await session.prompt('run a failing command');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'message_start', message: assistantMessage() });
  session.emit({ type: 'message_end', message: assistantMessage() });
  session.emit({
    type: 'tool_execution_start',
    toolCallId: 't1',
    toolName: 'bash',
    args: { command: 'false' },
  });
  session.emit({
    type: 'tool_execution_end',
    toolCallId: 't1',
    toolName: 'bash',
    result: { content: [{ type: 'text', text: 'exit 1' }] },
    isError: true,
  });
  session.emit({ type: 'turn_end', message: assistantMessage(), toolResults: [] });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });

  const toolSpan = capture.spans.find(
    (s) => (s.attributes as Record<string, unknown>)['gen_ai.tool.call.id'] === 't1',
  );
  assert.ok(toolSpan);
  assert.equal(toolSpan!.status.code, 2 /* SpanStatusCode.ERROR */);
});

test('captureContent: false suppresses input.value/output.value but keeps other attributes', async () => {
  const { capture, Session } = makeRig({ captureContent: false });
  const session = new Session();

  await session.prompt('sensitive prompt text');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'message_start', message: assistantMessage() });
  session.emit({
    type: 'message_end',
    message: assistantMessage({ content: [{ type: 'text', text: 'sensitive llm reply' }] }),
  });
  session.emit({ type: 'turn_end', message: assistantMessage(), toolResults: [] });
  session.emit({
    type: 'agent_end',
    messages: [assistantMessage({ content: [{ type: 'text', text: 'sensitive reply' }] })],
    willRetry: false,
  });

  const [llmSpan, rootSpan] = capture.spans;
  assert.equal(attrs(rootSpan!)['input.value'], undefined);
  assert.equal(attrs(rootSpan!)['output.value'], undefined);
  assert.equal(attrs(rootSpan!)['session.id'], 'sess-1');

  assert.ok(llmSpan, 'expected an LLM span to have been captured');
  assert.equal(attrs(llmSpan!)['openinference.span.kind'], 'LLM');
  assert.equal(
    attrs(llmSpan!)['output.value'],
    undefined,
    'captureContent:false must suppress output.value on the LLM span too, not just the root span',
  );
});

test('captureToolIo: false suppresses tool input.value/output.value but keeps the tool name', async () => {
  const { capture, Session } = makeRig({ captureToolIo: false });
  const session = new Session();

  await session.prompt('run a tool');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'message_start', message: assistantMessage() });
  session.emit({ type: 'message_end', message: assistantMessage() });
  // Use a path-based tool call, not a bash command: describeToolCallSpan's own
  // contract (see span-name.test.ts) is that a bash command up to 60 chars
  // appears verbatim in the span NAME by design — that tradeoff is unrelated
  // to captureToolIo, which only gates the separate input.value/output.value
  // attributes tested below. A path arg, by contrast, is unconditionally
  // reduced to its basename, which is what this test asserts.
  session.emit({
    type: 'tool_execution_start',
    toolCallId: 't1',
    toolName: 'read',
    args: { path: '/Users/alice/secret-project/notes.txt' },
  });
  session.emit({
    type: 'tool_execution_end',
    toolCallId: 't1',
    toolName: 'read',
    result: { secret: 'leaked?' },
    isError: false,
  });
  session.emit({ type: 'turn_end', message: assistantMessage(), toolResults: [] });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });

  const toolSpan = capture.spans.find(
    (s) => (s.attributes as Record<string, unknown>)['gen_ai.tool.call.id'] === 't1',
  );
  assert.ok(toolSpan);
  assert.equal(attrs(toolSpan!)['input.value'], undefined);
  assert.equal(attrs(toolSpan!)['output.value'], undefined);
  assert.equal(attrs(toolSpan!)['gen_ai.tool.name'], 'read');
  assert.equal(toolSpan!.name, 'read: notes.txt');
  assert.ok(
    !toolSpan!.name.includes('secret-project'),
    'the full path must never appear in the span name',
  );
});
