/**
 * Lens: span-lifecycle-event-ordering.
 *
 * Probes out-of-order / malformed AgentEvent sequences that a real Pi run
 * should never produce in the happy path, but that a buggy or aborted agent
 * loop plausibly could — and specifically what happens to in-flight state
 * (state.llmSpan, state.toolSpans, rootCtx) when a second "open" event
 * arrives before the first was ever closed, or when an event arrives with
 * no matching state open at all.
 *
 * Covers:
 *  - a second message_start firing back-to-back with no message_end/turn_end
 *    between them, WITHIN the same run (originally
 *    confirmed-bugfix-regressions.test.ts's Bug 2);
 *  - a stray message_start firing AFTER a clean agent_end already tore the
 *    run down, with no message_end ever following it, discovered only when
 *    the NEXT run's agent_start fires (originally
 *    agent-start-orphaned-llm-span.test.ts);
 *  - turn_end firing without a preceding message_end;
 *  - tool_execution_start with no prior message_start/message_end at all;
 *  - agent_end with zero prior events (no agent_start ever fired);
 *  - agent_start firing twice with no intervening agent_end;
 *  - a stray tool_execution_start firing after agent_end;
 *  - a duplicate tool_execution_start for a toolCallId that is already open
 *    (originally adversarial-concurrency-and-state-lifecycle.test.ts's first
 *    test) — the same duplicate-open-event concern as the message_start
 *    cases above, but exercised against the toolSpans Map instead of the
 *    single state.llmSpan reference, so it is kept here rather than dropped
 *    as a redundant case.
 *  (the last five originally lived in
 *  adversarial-event-ordering-and-malformed-events.test.ts)
 *
 * All must force-close the abandoned span/state instead of silently
 * overwriting it and dropping it (a span that never has .end() called on it
 * is never recorded/exported by OTel at all).
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { assistantMessage, attrs, makeRig } from './test-helpers';

test('a second message_start with no intervening message_end/turn_end force-closes the abandoned first LLM span instead of silently dropping it', async () => {
  const { capture, Session } = makeRig();
  const session = new Session();

  await session.prompt('two message_start events fire back to back, no message_end between them');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'message_start', message: assistantMessage({ model: 'first-model' }) });
  // No message_end for the first message — a second message_start fires
  // directly (e.g. a buggy provider stream that restarts mid-response).
  assert.doesNotThrow(() => {
    session.emit({ type: 'message_start', message: assistantMessage({ model: 'second-model' }) });
  });
  session.emit({ type: 'message_end', message: assistantMessage({ model: 'second-model' }) });
  session.emit({ type: 'turn_end', message: assistantMessage(), toolResults: [] });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });

  const llmSpans = capture.spans.filter((s) => attrs(s)['openinference.span.kind'] === 'LLM');
  assert.equal(
    llmSpans.length,
    2,
    'both the abandoned first LLM span and the properly-closed second must be exported — the ' +
      'first must never silently vanish just because state.llmSpan was overwritten',
  );
  const firstSpan = llmSpans.find((s) => attrs(s)['gen_ai.request.model'] === 'first-model');
  const secondSpan = llmSpans.find((s) => attrs(s)['gen_ai.request.model'] === 'second-model');
  assert.ok(
    firstSpan,
    'the abandoned first LLM span must still have been force-closed and exported',
  );
  assert.ok(secondSpan, 'the second LLM span must close normally via message_end');
  assert.equal(
    attrs(firstSpan!)['traceroot.pi.force_closed'],
    true,
    'the abandoned first LLM span must be marked as abnormally closed',
  );
  assert.equal(
    attrs(secondSpan!)['traceroot.pi.force_closed'],
    undefined,
    'the normally-closed second LLM span must NOT be marked force-closed',
  );
});

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

test('a duplicate tool_execution_start for a toolCallId that is already open force-closes the first span instead of silently overwriting the Map and leaking it', async () => {
  const { capture, Session } = makeRig();
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

test('a message_end (assistant) whose message_start never arrived is ignored — no crash, no phantom LLM span, and the run still completes normally', async () => {
  const { capture, Session } = makeRig();
  const session = new Session();

  await session.prompt('a stream resumes mid-message: only the end event is ever delivered');
  session.emit({ type: 'agent_start' });
  // No message_start — the close event arrives with nothing open. The handler
  // must take the `if (state.llmSpan)` no-op path rather than throwing or
  // fabricating an LLM span from close-time data alone.
  session.emit({ type: 'message_end', message: assistantMessage({ model: 'orphan-end-model' }) });
  // The rest of the run proceeds normally and must be unaffected.
  session.emit({ type: 'message_start', message: assistantMessage({ model: 'real-model' }) });
  session.emit({ type: 'message_end', message: assistantMessage({ model: 'real-model' }) });
  session.emit({ type: 'turn_end', message: assistantMessage(), toolResults: [] });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });

  const llmSpans = capture.spans.filter((s) => attrs(s)['openinference.span.kind'] === 'LLM');
  assert.equal(
    llmSpans.length,
    1,
    'only the real message_start/message_end pair produces an LLM span — the orphaned message_end produces nothing',
  );
  assert.equal(attrs(llmSpans[0])['gen_ai.request.model'], 'real-model');
  const rootSpans = capture.spans.filter((s) => attrs(s)['openinference.span.kind'] === 'AGENT');
  assert.equal(rootSpans.length, 1, 'the run itself still closes normally');
  assert.notEqual(
    attrs(rootSpans[0])['traceroot.pi.force_closed'],
    true,
    'the root span closed via the normal agent_end path, not a force-close sweep',
  );
});
