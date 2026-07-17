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

  // The root AGENT span is now anchored on prompt()'s own promise window
  // (see pi-test-helpers.ts's module header) — NOT awaited yet, so the
  // events below drive the run while the root is still open.
  const done = session.prompt('list files in /tmp');
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
  await done;

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
  // traceroot.pi.will_retry is gone (removed along with agent_end owning the
  // close); traceroot.pi.retry_count replaces it, now stamped once when the
  // enclosing prompt() call settles. No retry happened in this run, so it is 0.
  assert.equal(attrs(rootSpan)['traceroot.pi.retry_count'], 0);

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

  const done = session.prompt('do two things');
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
  await done;

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

  const done = session.prompt('trigger a provider error');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'message_start', message: assistantMessage() });
  session.emit({
    type: 'message_end',
    message: assistantMessage({ stopReason: 'error', errorMessage: 'rate limited' }),
  });
  session.emit({ type: 'turn_end', message: assistantMessage(), toolResults: [] });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
  await done;

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

  const done = session.prompt('run a failing command');
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
  await done;

  const toolSpan = capture.spans.find(
    (s) => (s.attributes as Record<string, unknown>)['gen_ai.tool.call.id'] === 't1',
  );
  assert.ok(toolSpan);
  assert.equal(toolSpan!.status.code, 2 /* SpanStatusCode.ERROR */);
});

test('captureContent: false suppresses input.value/output.value but keeps other attributes', async () => {
  const { capture, Session } = makeRig({ captureContent: false });
  const session = new Session();

  const done = session.prompt('sensitive prompt text');
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
  await done;

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

  const done = session.prompt('run a tool');
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
  await done;

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
    // Calling prompt() with a non-string bypasses proto.prompt's own
    // `typeof text === 'string'` guard, simulating a caller whose prompt
    // text is genuinely absent (as opposed to the empty-string case below)
    // while still registering the subscribe() listener that
    // instrumentPiCodingAgent wires up inside the wrapped prompt() call itself.
    const done = session.prompt(promptText as unknown as string);
    session.emit({ type: 'agent_start' });
    session.emit({ type: 'agent_end', messages: [], willRetry: false });
    await done;
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

// Boundary policy 2 (the prompt()-anchored root model): an early-return
// prompt() call — a handled "/command", a queue-only steer/followUp — resolves
// without ever reaching pi's internal run loop, so no agent_start/agent_end
// follows. The wrapped prompt() must still open a root on entry and finalize it
// OK on settle, yielding exactly one childless root. Migrated here from the
// deleted pi-test-helpers.test.ts (its F2 case): this is the suite's only
// span-asserting coverage of the no-events early-return path, so it belongs
// with the instrumentation behavior it exercises, not among the fixture tests.
test('an early-return prompt() call (resolved with no agent_start/agent_end) exports exactly one childless, OK-status AGENT root span', async () => {
  const { capture, Session } = makeRig();
  const session = new Session();

  const done = session.prompt('a handled slash command');
  session.resolvePrompt();
  await assert.doesNotReject(() => done);

  assert.equal(
    capture.spans.length,
    1,
    'exactly one span: the root, with ZERO child spans (no agent_start ever fired to open any)',
  );
  const rootSpan = capture.spans[0]!;
  assert.equal(rootSpan.name, 'AgentSession.prompt');
  assert.equal(attrs(rootSpan)['openinference.span.kind'], 'AGENT');
  assert.equal(
    rootSpan.status.code,
    1 /* SpanStatusCode.OK */,
    'finalize() explicitly stamps OK on a resolved call (a successfully-resolved ' +
      'promise, even with zero children, is not merely "unset")',
  );
  assert.equal(
    rootSpan.parentSpanId,
    undefined,
    'the root itself has no parent (it is the trace root)',
  );
});
