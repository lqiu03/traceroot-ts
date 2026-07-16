import assert from 'node:assert/strict';
import { test } from 'node:test';
import type { AssistantMessage } from '../src/pi/types';
import { assistantMessage as baseAssistantMessage, attrs, makeRig } from './pi-test-helpers';

// instrumentation.test.ts asserts on specific usage/cost numbers (see the LLM
// span assertions below), so it overrides test-helpers.ts's minimal
// placeholder usage with realistic values here — the one place a reader
// needs to look to find them, rather than a second copy of assistantMessage()
// with different numbers baked in silently (see test-helpers.ts's docstring).
function assistantMessage(overrides: Partial<AssistantMessage> = {}): AssistantMessage {
  return baseAssistantMessage({
    usage: {
      input: 100,
      output: 20,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 120,
      cost: { input: 0.001, output: 0.0006, cacheRead: 0, cacheWrite: 0, total: 0.0016 },
    },
    ...overrides,
  });
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
  // pi no longer self-stamps traceroot.sdk.name; core's TraceRootSpanProcessor
  // owns it uniformly (matching the Claude Agent SDK integration). This rig
  // wires no such processor, so the attribute is absent here.
  assert.equal(attrs(rootSpan)['traceroot.sdk.name'], undefined);
  assert.equal(attrs(rootSpan)['input.value'], 'list files in /tmp');
  assert.equal(attrs(rootSpan)['output.value'], 'listed the files');
  assert.equal(attrs(rootSpan)['traceroot.pi.will_retry'], false);

  assert.equal(attrs(llmSpan)['openinference.span.kind'], 'LLM');
  assert.equal(attrs(llmSpan)['gen_ai.system'], 'anthropic');
  assert.equal(attrs(llmSpan)['gen_ai.request.model'], 'claude-sonnet-5');
  assert.equal(attrs(llmSpan)['gen_ai.usage.input_tokens'], 100);
  assert.equal(attrs(llmSpan)['gen_ai.usage.output_tokens'], 20);
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

test('captureContent:false suppresses input.value on the ROOT span for an empty-string prompt exactly as for an absent one, while captureContent:true legitimately records the empty string', async () => {
  async function rootInputValue(
    captureContent: boolean,
    promptText: string | undefined,
  ): Promise<{ hasKey: boolean; value: unknown }> {
    const { capture, Session } = makeRig({ captureContent });
    const session = new Session();
    // Calling prompt() with a non-string bypasses the pendingInput.set() typeof
    // guard in instrumentation.ts, simulating a caller whose prompt text is
    // genuinely absent (as opposed to the empty-string case below) while still
    // registering the subscribe() listener that instrumentPiCodingAgent wires
    // up inside the wrapped prompt() call itself.
    await session.prompt(promptText as unknown as string);
    session.emit({ type: 'agent_start' });
    session.emit({ type: 'agent_end', messages: [], willRetry: false });
    const root = capture.spans[0]!;
    return {
      hasKey: Object.prototype.hasOwnProperty.call(attrs(root), 'input.value'),
      value: attrs(root)['input.value'],
    };
  }

  const falseEmpty = await rootInputValue(false, '');
  assert.equal(falseEmpty.hasKey, false, 'captureContent:false must omit input.value for ""');
  assert.equal(falseEmpty.value, undefined);

  const falseUndefined = await rootInputValue(false, undefined);
  assert.equal(
    falseUndefined.hasKey,
    false,
    'captureContent:false must omit input.value for undefined',
  );
  assert.equal(falseUndefined.value, undefined);

  // Contrast: with captureContent on, "" is a real (if uninformative) value
  // and must be distinguishable from "no prompt text at all".
  const trueEmpty = await rootInputValue(true, '');
  assert.equal(trueEmpty.hasKey, true, 'captureContent:true legitimately sets input.value to ""');
  assert.equal(trueEmpty.value, '');

  const trueUndefined = await rootInputValue(true, undefined);
  assert.equal(
    trueUndefined.hasKey,
    false,
    'no prompt text at all must leave input.value unset even when captureContent is on',
  );
  assert.equal(trueUndefined.value, undefined);
});
