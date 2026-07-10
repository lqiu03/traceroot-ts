import assert from 'node:assert/strict';
import { test } from 'node:test';
import type { ExportResult } from '@opentelemetry/core';
import { ExportResultCode } from '@opentelemetry/core';
import type { ReadableSpan, SpanExporter } from '@opentelemetry/sdk-trace-base';
import { instrumentPiCodingAgent } from '../src/instrumentation';
import type { AgentEvent, AssistantMessage } from '../src/types';

// Copied locally — no shared state across test files, matching
// packages/mastra/tests/exporter-adversarial.test.ts's explicit convention.
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

test('instrumenting the same sdk object twice does not double-wrap prompt', async () => {
  const capture = new CapturingExporter();
  const Session = makeFakeSessionClass();
  const sdk = { AgentSession: Session };

  instrumentPiCodingAgent(sdk, { apiKey: 'k', _spanExporter: capture });
  const wrappedOnce = Session.prototype.prompt;
  instrumentPiCodingAgent(sdk, { apiKey: 'k', _spanExporter: capture });
  const wrappedTwice = Session.prototype.prompt;

  assert.equal(
    wrappedOnce,
    wrappedTwice,
    'a second instrumentPiCodingAgent() call must be a no-op',
  );

  const session = new Session();
  await session.prompt('hi');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });

  assert.equal(
    capture.spans.length,
    1,
    'exactly one root span, not two — proves subscribe() was not registered twice',
  );
});

test('missing API key returns the sdk unmodified and never throws', () => {
  const Session = makeFakeSessionClass();
  const originalPrompt = Session.prototype.prompt;
  const sdk = { AgentSession: Session };

  const result = instrumentPiCodingAgent(sdk, {});

  assert.equal(result, sdk);
  assert.equal(
    Session.prototype.prompt,
    originalPrompt,
    'prompt must be untouched when there is no API key',
  );
});

test('a module missing AgentSession.prototype.prompt/subscribe degrades to a no-op, never throws', () => {
  const brokenSdk = { AgentSession: { prototype: {} } };
  assert.doesNotThrow(() => {
    const result = instrumentPiCodingAgent(brokenSdk, { apiKey: 'k' });
    assert.equal(result, brokenSdk);
  });

  assert.doesNotThrow(() => {
    instrumentPiCodingAgent(undefined, { apiKey: 'k' });
  });
  assert.doesNotThrow(() => {
    instrumentPiCodingAgent({}, { apiKey: 'k' });
  });
});

test('two different session instances on the same instrumented sdk keep fully independent span trees', async () => {
  const capture = new CapturingExporter();
  const Session = makeFakeSessionClass();
  const sdk = { AgentSession: Session };
  instrumentPiCodingAgent(sdk, { apiKey: 'k', _spanExporter: capture });

  const sessionA = new Session();
  const sessionB = new Session();
  (sessionA as { sessionId: string }).sessionId = 'session-a';
  (sessionB as { sessionId: string }).sessionId = 'session-b';

  // Interleaved on purpose: A starts, B starts, A's tool call runs, B's turn ends, A ends.
  await sessionA.prompt('task A');
  sessionA.emit({ type: 'agent_start' });
  await sessionB.prompt('task B');
  sessionB.emit({ type: 'agent_start' });
  sessionA.emit({ type: 'message_start', message: assistantMessage() });
  sessionA.emit({ type: 'message_end', message: assistantMessage() });
  sessionA.emit({ type: 'tool_execution_start', toolCallId: 'a-tool', toolName: 'bash', args: {} });
  sessionA.emit({
    type: 'tool_execution_end',
    toolCallId: 'a-tool',
    toolName: 'bash',
    result: {},
    isError: false,
  });
  sessionB.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
  sessionA.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });

  assert.equal(capture.spans.length, 4, 'A: root+LLM+tool (3), B: root (1)');

  const bRoot = capture.spans.find((s) => attrs(s)['session.id'] === 'session-b');
  const aRoot = capture.spans.find((s) => attrs(s)['session.id'] === 'session-a');
  assert.ok(bRoot);
  assert.ok(aRoot);
  // Only the root span carries session.id and only tool spans carry
  // gen_ai.tool.call.id — the LLM span carries neither, so "everything
  // that isn't B's root" is the correct way to count A's 3 spans.
  const aRelated = capture.spans.filter((s) => s !== bRoot);
  assert.equal(aRelated.length, 3, 'A: root + LLM + tool span');
  // B's root span must not be a parent/child of anything in A's tree.
  assert.notEqual(bRoot!.spanContext().traceId, aRoot!.spanContext().traceId);
});

test('tool_execution_end for an unknown toolCallId (no matching start) is ignored, not a crash', async () => {
  const capture = new CapturingExporter();
  const Session = makeFakeSessionClass();
  const sdk = { AgentSession: Session };
  instrumentPiCodingAgent(sdk, { apiKey: 'k', _spanExporter: capture });
  const session = new Session();

  await session.prompt('hi');
  session.emit({ type: 'agent_start' });
  assert.doesNotThrow(() => {
    session.emit({
      type: 'tool_execution_end',
      toolCallId: 'never-started',
      toolName: 'bash',
      result: {},
      isError: false,
    });
  });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });

  const toolSpans = capture.spans.filter((s) => attrs(s)['gen_ai.tool.name']);
  assert.equal(toolSpans.length, 0);
});

test('agent_end while a tool span is still open force-closes it instead of leaking', async () => {
  const capture = new CapturingExporter();
  const Session = makeFakeSessionClass();
  const sdk = { AgentSession: Session };
  instrumentPiCodingAgent(sdk, { apiKey: 'k', _spanExporter: capture });
  const session = new Session();

  await session.prompt('hi');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'message_start', message: assistantMessage() });
  session.emit({ type: 'message_end', message: assistantMessage() });
  session.emit({
    type: 'tool_execution_start',
    toolCallId: 'dangling',
    toolName: 'bash',
    args: {},
  });
  // No tool_execution_end — simulates an aborted run mid-tool-call.
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });

  assert.equal(capture.spans.length, 3, 'root, LLM, and the force-closed dangling tool span');
  const dangling = capture.spans.find((s) => attrs(s)['gen_ai.tool.name'] === 'bash');
  assert.ok(dangling);
  assert.equal(
    attrs(dangling!)['traceroot.pi.force_closed'],
    true,
    'must be marked as abnormally closed, not indistinguishable from a clean tool span',
  );
});

test('message_start/message_end for non-assistant roles never opens an LLM span', async () => {
  const capture = new CapturingExporter();
  const Session = makeFakeSessionClass();
  const sdk = { AgentSession: Session };
  instrumentPiCodingAgent(sdk, { apiKey: 'k', _spanExporter: capture });
  const session = new Session();

  await session.prompt('hi');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'message_start', message: { role: 'user', content: 'hi', timestamp: 0 } });
  session.emit({ type: 'message_end', message: { role: 'user', content: 'hi', timestamp: 0 } });
  session.emit({
    type: 'message_start',
    message: {
      role: 'toolResult',
      toolCallId: 'x',
      toolName: 'bash',
      content: [],
      isError: false,
      timestamp: 0,
    },
  });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });

  assert.equal(
    capture.spans.length,
    1,
    'only the root span — no LLM span for user/toolResult messages',
  );
});

test('a handler throw inside span-building is caught and never propagates to session.emit()', async () => {
  const capture = new CapturingExporter();
  const Session = makeFakeSessionClass();
  const sdk = { AgentSession: Session };
  instrumentPiCodingAgent(sdk, { apiKey: 'k', _spanExporter: capture });
  const session = new Session();

  await session.prompt('hi');
  session.emit({ type: 'agent_start' });
  // A message_end with `role: 'assistant'` but a malformed/missing `usage`
  // field must not crash the listener — attribute setters must tolerate it.
  assert.doesNotThrow(() => {
    session.emit({
      type: 'message_end',
      // @ts-expect-error intentionally malformed to test resilience
      message: { role: 'assistant', content: [], stopReason: 'stop', timestamp: 0 },
    });
  });
  assert.doesNotThrow(() => {
    session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
  });
});

test('tool args/result containing a circular reference do not crash span creation', async () => {
  const capture = new CapturingExporter();
  const Session = makeFakeSessionClass();
  const sdk = { AgentSession: Session };
  instrumentPiCodingAgent(sdk, { apiKey: 'k', _spanExporter: capture });
  const session = new Session();

  const circular: Record<string, unknown> = {};
  circular.self = circular;

  await session.prompt('hi');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'message_start', message: assistantMessage() });
  session.emit({ type: 'message_end', message: assistantMessage() });
  assert.doesNotThrow(() => {
    session.emit({
      type: 'tool_execution_start',
      toolCallId: 't1',
      toolName: 'bash',
      args: circular,
    });
  });
  assert.doesNotThrow(() => {
    session.emit({
      type: 'tool_execution_end',
      toolCallId: 't1',
      toolName: 'bash',
      result: circular,
      isError: false,
    });
  });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });

  const toolSpan = capture.spans.find((s) => attrs(s)['gen_ai.tool.call.id'] === 't1');
  assert.ok(toolSpan);
  assert.equal(
    attrs(toolSpan!)['input.value'],
    undefined,
    'circular args must be skipped, not crash or emit garbage',
  );
});

test('a rejected prompt() (validation failure before agent_start) never creates a dangling root span', async () => {
  const capture = new CapturingExporter();
  const Session = makeFakeSessionClass();
  Session.prototype.prompt = async function (): Promise<void> {
    throw new Error('no model selected');
  };
  const sdk = { AgentSession: Session };
  instrumentPiCodingAgent(sdk, { apiKey: 'k', _spanExporter: capture });
  const session = new Session();

  await assert.rejects(() => session.prompt('hi'), /no model selected/);
  // agent_start never fires for a run that failed validation before starting.
  assert.equal(capture.spans.length, 0);
});

test('willRetry: true still closes the root span (one span per attempt), recording the attribute', async () => {
  const capture = new CapturingExporter();
  const Session = makeFakeSessionClass();
  const sdk = { AgentSession: Session };
  instrumentPiCodingAgent(sdk, { apiKey: 'k', _spanExporter: capture });
  const session = new Session();

  await session.prompt('hi');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: true });
  // A retry re-enters the loop and fires a fresh agent_start/agent_end pair.
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });

  assert.equal(
    capture.spans.length,
    2,
    'each attempt gets its own root span, both properly closed',
  );
  assert.equal(attrs(capture.spans[0]!)['traceroot.pi.will_retry'], true);
  assert.equal(attrs(capture.spans[1]!)['traceroot.pi.will_retry'], false);
});
