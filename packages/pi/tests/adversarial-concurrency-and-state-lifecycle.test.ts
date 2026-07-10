/**
 * Lens: concurrency-and-state-lifecycle.
 *
 * Probes src/instrumentation.ts's SessionSpanState lifecycle under scenarios
 * distinct from the other adversarial-*.test.ts files (verified by reading
 * every existing test file fully before writing this one): a duplicate
 * tool_execution_start for a toolCallId that is already open (no intervening
 * end), a genuine SECOND session.prompt() call on the same session instance
 * after the first run's agent_end already fired cleanly (as opposed to the
 * existing willRetry test, which replays agent_start/agent_end twice under a
 * single prompt() call and never re-enters proto.prompt itself), and whether
 * instrumentPiCodingAgent()/resolveConfig() snapshot the caller's config
 * object instead of holding a live reference that could change later.
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
function makeFakeSessionClass() {
  return class FakeAgentSession {
    sessionId = 'sess-1';
    private listeners: Array<(event: AgentEvent) => void> = [];
    subscribeCallCount = 0;
    async prompt(_text: string, _options?: unknown): Promise<void> {}
    subscribe(listener: (event: AgentEvent) => void): () => void {
      this.subscribeCallCount += 1;
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

test('a duplicate tool_execution_start for a toolCallId that is already open force-closes the first span instead of silently overwriting the Map and leaking it', async () => {
  const capture = new CapturingExporter();
  const Session = makeFakeSessionClass();
  const sdk = { AgentSession: Session };
  instrumentPiCodingAgent(sdk, { apiKey: 'k', _spanExporter: capture });
  const session = new Session();

  await session.prompt('a buggy tool runner fires two starts for the same call id');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'message_start', message: assistantMessage() });
  session.emit({ type: 'message_end', message: assistantMessage() });
  session.emit({
    type: 'tool_execution_start',
    toolCallId: 't1',
    toolName: 'bash',
    args: { command: 'first' },
  });
  // Duplicate start for the SAME toolCallId, no tool_execution_end in
  // between — e.g. a retried dispatch that reused the id instead of minting
  // a fresh one. The Map's own .set() semantics would otherwise silently
  // drop the reference to the first (still-open, unended) span, and since
  // spans only export on end(), that first span would never be seen again.
  assert.doesNotThrow(() => {
    session.emit({
      type: 'tool_execution_start',
      toolCallId: 't1',
      toolName: 'write',
      args: { command: 'second' },
    });
  });
  session.emit({
    type: 'tool_execution_end',
    toolCallId: 't1',
    toolName: 'write',
    result: {},
    isError: false,
  });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });

  const toolSpans = capture.spans.filter((s) => attrs(s)['gen_ai.tool.call.id'] === 't1');
  assert.equal(
    toolSpans.length,
    2,
    'both the abandoned first span and the properly-closed second span must be exported — the ' +
      'first must never silently vanish just because its Map slot was overwritten',
  );
  const firstSpan = toolSpans.find((s) => attrs(s)['gen_ai.tool.name'] === 'bash');
  const secondSpan = toolSpans.find((s) => attrs(s)['gen_ai.tool.name'] === 'write');
  assert.ok(firstSpan, 'the first (bash) span must still have been force-closed and exported');
  assert.ok(secondSpan, 'the second (write) span must close normally via tool_execution_end');
  assert.equal(
    attrs(firstSpan!)['traceroot.pi.force_closed'],
    true,
    'the abandoned first span must be marked as abnormally closed',
  );
  assert.equal(
    attrs(secondSpan!)['traceroot.pi.force_closed'],
    undefined,
    'the normally-closed second span must NOT be marked force-closed',
  );
  assert.notEqual(
    firstSpan!.spanContext().spanId,
    secondSpan!.spanContext().spanId,
    'the two spans sharing a toolCallId must still be genuinely distinct span objects',
  );
});

test('a genuine second session.prompt() call after the first runs agent_end already fired cleanly produces a fully separate span tree, reusing the same subscribe() listener rather than re-subscribing', async () => {
  const capture = new CapturingExporter();
  const Session = makeFakeSessionClass();
  const sdk = { AgentSession: Session };
  instrumentPiCodingAgent(sdk, { apiKey: 'k', _spanExporter: capture });
  const session = new Session();

  // Run 1: a full, clean turn with a tool call, then agent_end.
  await session.prompt('first task');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'message_start', message: assistantMessage({ model: 'run-1-model' }) });
  session.emit({ type: 'message_end', message: assistantMessage({ model: 'run-1-model' }) });
  session.emit({
    type: 'tool_execution_start',
    toolCallId: 'call-1',
    toolName: 'bash',
    args: { command: 'echo run1' },
  });
  session.emit({
    type: 'tool_execution_end',
    toolCallId: 'call-1',
    toolName: 'bash',
    result: {},
    isError: false,
  });
  session.emit({ type: 'turn_end', message: assistantMessage(), toolResults: [] });
  session.emit({
    type: 'agent_end',
    messages: [assistantMessage({ content: [{ type: 'text', text: 'run1 done' }] })],
    willRetry: false,
  });

  assert.equal(capture.spans.length, 3, 'run 1: root + LLM + tool span');

  // Run 2: a genuine second call to prompt() on the SAME session instance
  // (e.g. the user sends a second chat message), well after run 1's
  // agent_end already tore its state down. Deliberately reuses run 1's
  // toolCallId ("call-1") to prove a stale Map entry from run 1 cannot
  // bleed into run 2 — agent_end already cleared state.toolSpans.
  await session.prompt('second task');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'message_start', message: assistantMessage({ model: 'run-2-model' }) });
  session.emit({ type: 'message_end', message: assistantMessage({ model: 'run-2-model' }) });
  session.emit({
    type: 'tool_execution_start',
    toolCallId: 'call-1',
    toolName: 'bash',
    args: { command: 'echo run2' },
  });
  session.emit({
    type: 'tool_execution_end',
    toolCallId: 'call-1',
    toolName: 'bash',
    result: {},
    isError: false,
  });
  session.emit({ type: 'turn_end', message: assistantMessage(), toolResults: [] });
  session.emit({
    type: 'agent_end',
    messages: [assistantMessage({ content: [{ type: 'text', text: 'run2 done' }] })],
    willRetry: false,
  });

  assert.equal(
    session.subscribeCallCount,
    1,
    'a second real prompt() call on the same session instance must not re-subscribe — the ' +
      'subscribedSessions WeakSet guard must hold across repeated prompt() calls, not just ' +
      'across repeated instrumentPiCodingAgent() calls',
  );

  assert.equal(capture.spans.length, 6, 'run 1 (3) + run 2 (3), none dropped or duplicated');

  const rootSpans = capture.spans.filter((s) => attrs(s)['openinference.span.kind'] === 'AGENT');
  assert.equal(rootSpans.length, 2);
  const run1Root = rootSpans.find((s) => attrs(s)['input.value'] === 'first task');
  const run2Root = rootSpans.find((s) => attrs(s)['input.value'] === 'second task');
  assert.ok(run1Root, "run 1's root span must carry run 1's own prompt text");
  assert.ok(
    run2Root,
    "run 2's root span must carry run 2's own prompt text, not a stale copy of run 1's",
  );
  assert.equal(attrs(run1Root!)['output.value'], 'run1 done');
  assert.equal(attrs(run2Root!)['output.value'], 'run2 done');
  assert.notEqual(
    run1Root!.spanContext().traceId,
    run2Root!.spanContext().traceId,
    'the two runs must live in genuinely separate traces',
  );

  const toolSpans = capture.spans.filter((s) => attrs(s)['gen_ai.tool.call.id'] === 'call-1');
  assert.equal(toolSpans.length, 2, 'each run gets its own span despite reusing the toolCallId');
  const run1Tool = toolSpans.find((s) => s.name.includes('run1'));
  const run2Tool = toolSpans.find((s) => s.name.includes('run2'));
  assert.ok(run1Tool);
  assert.ok(run2Tool);
  // ReadableSpan has no top-level traceId field — it lives under
  // spanContext().traceId, matching how root-span trace ids are read above.
  assert.notEqual(
    run1Tool!.spanContext().traceId,
    run2Tool!.spanContext().traceId,
    "run 2's reused-id tool span must not be attached to run 1's trace",
  );
  assert.equal(
    run1Tool!.spanContext().traceId,
    run1Root!.spanContext().traceId,
    "run 1's tool span must belong to run 1's own trace",
  );
  assert.equal(
    run2Tool!.spanContext().traceId,
    run2Root!.spanContext().traceId,
    "run 2's tool span must belong to run 2's own trace, not leak forward from run 1's now-stale state",
  );
});

test('instrumentPiCodingAgent() snapshots the config object at call time — mutating apiKey/captureContent/_spanExporter on the caller-owned object after the call returns has no effect on already-instrumented behavior', async () => {
  const originalCapture = new CapturingExporter();
  const mutatedCapture = new CapturingExporter();
  const Session = makeFakeSessionClass();
  const sdk = { AgentSession: Session };

  // A plain mutable object, exactly as a caller might build and later reuse
  // or mutate it (e.g. a shared config object edited elsewhere in the host app).
  const config = {
    apiKey: 'original-key',
    captureContent: true,
    _spanExporter: originalCapture,
  };

  instrumentPiCodingAgent(sdk, config);

  // Mutate every field AFTER instrumentPiCodingAgent() has already returned.
  // resolveConfig() must have copied primitive values out (not retained a
  // live reference to `config`) and must have captured the exporter object
  // that was live at call time, not one assigned to the field afterward.
  config.apiKey = 'mutated-key';
  config.captureContent = false;
  config._spanExporter = mutatedCapture;

  const session = new Session();
  await session.prompt('sensitive prompt text');
  session.emit({ type: 'agent_start' });
  session.emit({
    type: 'agent_end',
    messages: [assistantMessage({ content: [{ type: 'text', text: 'sensitive reply' }] })],
    willRetry: false,
  });

  assert.equal(
    mutatedCapture.spans.length,
    0,
    'the exporter assigned to config._spanExporter AFTER the call must never receive spans — ' +
      'the pipeline was already built from the exporter that was live at call time',
  );
  assert.equal(
    originalCapture.spans.length,
    1,
    'the exporter that was live at call time must still receive the span',
  );

  const [rootSpan] = originalCapture.spans;
  assert.equal(
    attrs(rootSpan!)['input.value'],
    'sensitive prompt text',
    'captureContent must still resolve to its call-time value (true), not the post-call ' +
      'mutation to false — resolveConfig() must copy primitives by value, not hold a live ' +
      'reference to the caller-owned config object',
  );
  assert.equal(attrs(rootSpan!)['output.value'], 'sensitive reply');
});
