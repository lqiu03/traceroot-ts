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
 *  - a stray message_start/tool_execution_start firing AFTER a clean
 *    prompt() call already settled and tore the run down, with no matching
 *    close event ever following it, discovered only when the NEXT prompt()
 *    call's own agent_start fires (originally
 *    agent-start-orphaned-llm-span.test.ts);
 *  - turn_end firing without a preceding message_end;
 *  - tool_execution_start with no prior message_start/message_end at all;
 *  - agent_end with zero prior events (no agent_start ever fired, but a root
 *    already exists — opened by prompt() itself under the new model);
 *  - agent_start firing twice with no intervening agent_end, no new prompt()
 *    call between them — the retry/compaction continuation shape;
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
import { assistantMessage, attrs, makeRig } from './pi-test-helpers';

test('a second message_start with no intervening message_end/turn_end force-closes the abandoned first LLM span instead of silently dropping it', async () => {
  const { capture, Session } = makeRig();
  const session = new Session();

  const done = session.prompt(
    'two message_start events fire back to back, no message_end between them',
  );
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
  await done;

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

test('a stray message_start firing after a clean prompt() call already settled (no matching message_end) is force-closed on the NEXT prompt() call’s agent_start instead of being silently dropped', async () => {
  const { capture, Session } = makeRig();
  const session = new Session();

  // Reworked for the new model: run 1 must be a COMPLETE, AWAITED prompt()
  // call (root opened and closed) before the stray event fires, so the
  // stray message_start genuinely has no root open — under the old
  // agent_start-anchored model, agent_end alone cleared rootCtx; under the
  // new model the root only clears once THIS prompt() call's own promise
  // settles.
  const done1 = session.prompt('run 1 finishes cleanly with no LLM turn at all');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
  await done1;

  assert.equal(capture.spans.length, 1, 'only run 1s root span has exported so far');

  // A straggler assistant message_start arrives after run 1s prompt() call
  // already settled and tore the run down (e.g. a late stream event) — with
  // rootCtx cleared, it opens an LLM span parented under ROOT_CONTEXT. Its
  // message_end never arrives (the stream is already abandoned).
  session.emit({
    type: 'message_start',
    message: assistantMessage({ model: 'stray-orphaned-model' }),
  });

  // Run 2 starts fresh via a genuinely NEW prompt() call — agent_start alone
  // no longer fabricates a root under the new model (see instrumentation.ts's
  // rootless-bypass boundary policy), so "run 2 starts fresh" now requires
  // its own prompt() call, not just another agent_start. This is the moment
  // the orphaned llmSpan from the straggler above must be force-closed and
  // exported — mirroring exactly how a stray tool_execution_start in the
  // same position is already swept.
  const done2 = session.prompt('run 2 prompt text');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'message_start', message: assistantMessage({ model: 'run-2-model' }) });
  session.emit({ type: 'message_end', message: assistantMessage({ model: 'run-2-model' }) });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
  await done2;

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

  const done = session.prompt('a stream error truncates turn 1 before message_end fires');
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
  await done;

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

  const done = session.prompt(
    'agent invokes a tool with no assistant message event in front of it',
  );
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
  await done;

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

  const done = session.prompt('a buggy tool runner fires two starts for the same call id');
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
  await done;

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

test('agent_end with zero prior agent_start events still stamps the root that prompt() already opened, and does not throw', async () => {
  // FLIPPED for the new model: under the old agent_start-anchored root, a
  // session with no agent_start ever fired had no root to close, so this
  // event produced zero spans. Under the new prompt()-anchored root, the
  // wrapped prompt() call itself already opened a root the instant it was
  // called — agent_end simply stamps output onto that still-open root
  // (agent_start is not required first; it never was a strict precondition
  // for the handler itself, only for the LLM/tool span machinery).
  const { capture, Session } = makeRig();
  const session = new Session();

  const done = session.prompt('nothing has happened yet on this session');
  assert.doesNotThrow(() => {
    session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
  });
  await done;

  const rootSpans = capture.spans.filter((s) => attrs(s)['openinference.span.kind'] === 'AGENT');
  assert.equal(
    rootSpans.length,
    1,
    'the root prompt() opened is still stamped and exported once its promise settles',
  );
});

test('agent_start firing twice with no intervening agent_end, and no new prompt() call between them, shares ONE root and does not leak a stale LLM context into the continuation', async () => {
  // FLIPPED for the new model: under the old agent_start-anchored root, this
  // exact event shape (two agent_starts, no agent_end, no new prompt() call)
  // was indistinguishable from "run 1 abandoned, run 2 started cold" — so
  // agent_start force-closed run 1's root and opened a genuinely NEW one for
  // run 2. Under the new model this shape IS the retry/compaction
  // continuation (see instrumentation.ts's module header): both attempts
  // belong to the SAME prompt() call and must share the SAME still-open
  // root, with only the dangling LLM/TOOL spans from the abandoned first
  // attempt swept — never the root itself.
  const { capture, Session } = makeRig();
  const session = new Session();

  const done = session.prompt(
    'attempt one gets abandoned mid-flight (e.g. loop restart), attempt two starts cold',
  );
  session.emit({ type: 'agent_start' }); // attempt one
  session.emit({ type: 'message_start', message: assistantMessage({ model: 'attempt-1-model' }) });
  // Attempt one never reaches message_end/turn_end/agent_end — the loop is
  // restarted and immediately emits a second agent_start (no new prompt()).
  session.emit({ type: 'agent_start' }); // attempt two, no agent_end for attempt one in between
  session.emit({
    type: 'tool_execution_start',
    toolCallId: 'attempt2-tool',
    toolName: 'bash',
    args: { command: 'echo attempt2' },
  });
  session.emit({
    type: 'tool_execution_end',
    toolCallId: 'attempt2-tool',
    toolName: 'bash',
    result: {},
    isError: false,
  });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
  await done;

  const rootSpans = capture.spans.filter((s) => attrs(s)['openinference.span.kind'] === 'AGENT');
  assert.equal(
    rootSpans.length,
    1,
    'both attempts share exactly ONE root span — agent_start must never force-close it under the ' +
      'new model, since it belongs to the whole prompt() window, not to one attempt',
  );

  const attempt1LlmSpan = capture.spans.find(
    (s) => attrs(s)['gen_ai.request.model'] === 'attempt-1-model',
  );
  assert.ok(
    attempt1LlmSpan,
    'attempt ones dangling LLM span must still be force-closed, not leaked',
  );
  assert.equal(attrs(attempt1LlmSpan!)['traceroot.pi.force_closed'], true);

  const attempt2Tool = capture.spans.find(
    (s) => attrs(s)['gen_ai.tool.call.id'] === 'attempt2-tool',
  );
  assert.ok(attempt2Tool);
  // The critical assertion under the new model: attempt two's tool span must
  // parent under the SAME shared root as attempt one's abandoned LLM span —
  // proving the continuation correctly reuses rootCtx rather than losing it.
  assert.equal(
    attempt2Tool!.parentSpanId,
    rootSpans[0]!.spanContext().spanId,
    'attempt twos tool span must parent under the one shared root',
  );
  assert.equal(
    attempt1LlmSpan!.spanContext().traceId,
    attempt2Tool!.spanContext().traceId,
    'both attempts must live in the SAME trace — the whole point of anchoring the root on ' +
      'prompt() rather than on each individual attempt',
  );
});

test('a stray tool_execution_start firing after a clean prompt() call already settled does not corrupt the NEXT prompt() call’s span tree', async () => {
  const { capture, Session } = makeRig();
  const session = new Session();

  const done1 = session.prompt('run 1 finishes cleanly');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
  await done1;
  // run 1's root span is the only span exported so far — capture its id
  // before run 2 starts so we can unambiguously tell the two roots apart
  // (both carry identical attributes otherwise).
  assert.equal(capture.spans.length, 1);
  const run1RootSpanId = capture.spans[0]!.spanContext().spanId;

  // A straggler event arrives after run 1's prompt() call already settled —
  // e.g. an async tool runner's callback that resolves after the agent loop
  // already tore the run down.
  assert.doesNotThrow(() => {
    session.emit({
      type: 'tool_execution_start',
      toolCallId: 'straggler',
      toolName: 'bash',
      args: { command: 'echo late' },
    });
  });

  // Run 2 starts fresh via a genuinely NEW prompt() call (see the earlier
  // stray-message_start test's comment on why agent_start alone no longer
  // suffices under the new model).
  const done2 = session.prompt('run 2 prompt text');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'message_start', message: assistantMessage() });
  session.emit({ type: 'message_end', message: assistantMessage() });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
  await done2;

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
  // agent_start cleanup since it was never explicitly ended — but it must not
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

  const done = session.prompt('a stream resumes mid-message: only the end event is ever delivered');
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
  await done;

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
    'the root span closed via the normal prompt()-settle path, not a force-close sweep',
  );
});
