/**
 * Behavioral guard that the "force-close every open tool span, then the LLM
 * span, then (sometimes) the root span" dangling-span sweep actually runs at
 * every call site that needs it in packages/traceroot/src/pi/instrumentation.ts
 * — agent_start, turn_end, agent_end, proto.prompt's overlap-safety check, and
 * dispose().
 *
 * This finding was originally guarded by a source-text regex (assert the sweep
 * helper is defined once and called N times). That caught the structural
 * duplication risk but broke on harmless refactors and never exercised the
 * actual behavior. These tests instead drive each call site into a real
 * dangling-span scenario through the public API and assert the observable
 * force-close result, mirroring session-dispose.test.ts's own dispose()
 * pattern — so a real future regression (a site quietly dropping its sweep) is
 * caught by behavior, and a benign rename/extraction of the helper is not.
 *
 * Sweep scope, under the prompt()-anchored root model (see
 * instrumentation.ts's module header):
 *   - agent_start & turn_end: sweep only the dangling LLM/TOOL spans. The
 *     root is NEVER force-closed by agent_start anymore — it belongs to the
 *     enclosing prompt() call's whole promise window (which may span several
 *     agent_start/agent_end attempts via retry/compaction/follow-up), not to
 *     any one attempt, so agent_start must leave it open across a
 *     continuation.
 *   - agent_end: sweeps only the dangling LLM/TOOL spans; the root is
 *     STAMPED (its output overwritten with this attempt's), never force-
 *     closed and never even ended here.
 *   - proto.prompt's OVERLAP SAFETY check & dispose(): these are now the
 *     ONLY two places that ever force-close a still-open ROOT — a second
 *     prompt() call arriving while a previous window's root is still open,
 *     or the session being torn down mid-run.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { assistantMessage, attrs, makeRig } from './pi-test-helpers';

test('agent_start sweeps a dangling LLM + TOOL span from a crashed prior ATTEMPT, but leaves the still-open root untouched', async () => {
  // Flipped from the pre-fix model, where a second agent_start with no
  // intervening agent_end force-closed the PREVIOUS run's root (agent_start
  // used to own opening/closing roots). Under the new model this exact event
  // sequence — two agent_starts with no agent_end between them, and no new
  // prompt() call in between either — is precisely the retry/compaction
  // continuation shape (see instrumentation.ts's module header): the root
  // must stay open and become the shared parent for BOTH attempts' spans,
  // not be force-closed and orphan the second attempt onto a fresh trace.
  const { capture, Session } = makeRig();
  const session = new Session();

  const done = session.prompt('attempt one crashes mid-flight, the loop restarts it');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'message_start', message: assistantMessage({ model: 'attempt1-llm' }) });
  session.emit({
    type: 'tool_execution_start',
    toolCallId: 'attempt1-tool',
    toolName: 'read_file',
    args: { path: '/tmp/one' },
  });
  assert.equal(capture.spans.length, 0, 'nothing exports while the root is still open');

  // A fresh agent_start (no agent_end for attempt one, no new prompt() call)
  // must sweep attempt one's dangling LLM/TOOL spans, but the root itself —
  // still the SAME prompt() window's root — must stay open.
  session.emit({ type: 'agent_start' });

  assert.equal(
    capture.spans.length,
    2,
    'agent_start must force-close attempt one’s LLM and TOOL spans only, not the still-open root ' +
      `(found ${capture.spans.length})`,
  );
  const llmSpan = capture.spans.find((s) => attrs(s)['gen_ai.request.model'] === 'attempt1-llm');
  const toolSpan = capture.spans.find((s) => attrs(s)['gen_ai.tool.call.id'] === 'attempt1-tool');
  assert.ok(llmSpan && toolSpan, 'attempt one’s LLM and TOOL spans must be exported');
  for (const span of [llmSpan!, toolSpan!]) {
    assert.equal(
      attrs(span)['traceroot.pi.force_closed'],
      true,
      `${span.name} must be marked force_closed by agent_start’s sweep`,
    );
  }
  assert.equal(
    capture.spans.find((s) => s.name === 'AgentSession.prompt'),
    undefined,
    'the root span must still be open — agent_start never force-closes it under the new model',
  );

  // The second attempt completes normally, sharing the ONE still-open root.
  session.emit({
    type: 'agent_end',
    messages: [assistantMessage({ content: [{ type: 'text', text: 'recovered' }] })],
    willRetry: false,
  });
  await done;

  const rootSpans = capture.spans.filter((s) => attrs(s)['openinference.span.kind'] === 'AGENT');
  assert.equal(rootSpans.length, 1, 'exactly one root span for the whole prompt() call');
  assert.equal(attrs(rootSpans[0]!)['output.value'], 'recovered');
});

test('turn_end force-closes a dangling LLM + TOOL span but leaves the root span open', async () => {
  const { capture, Session } = makeRig();
  const session = new Session();

  const done = session.prompt('run with a stream error mid-turn');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'message_start', message: assistantMessage({ model: 'turn-llm' }) });
  session.emit({
    type: 'tool_execution_start',
    toolCallId: 'turn-tool',
    toolName: 'read_file',
    args: { path: '/tmp/turn' },
  });

  // turn_end with neither message_end nor tool_execution_end having arrived:
  // the LLM and TOOL spans are dangling. The root span is NOT swept here.
  session.emit({ type: 'turn_end', message: assistantMessage(), toolResults: [] });

  assert.equal(
    capture.spans.length,
    2,
    'turn_end must force-close exactly the dangling LLM + TOOL spans, not the root span (found ' +
      `${capture.spans.length})`,
  );
  const llmSpan = capture.spans.find((s) => attrs(s)['gen_ai.request.model'] === 'turn-llm');
  const toolSpan = capture.spans.find((s) => attrs(s)['gen_ai.tool.call.id'] === 'turn-tool');
  assert.ok(llmSpan && toolSpan, 'the LLM and TOOL spans must be force-closed and exported');
  assert.equal(attrs(llmSpan!)['traceroot.pi.force_closed'], true);
  assert.equal(attrs(toolSpan!)['traceroot.pi.force_closed'], true);
  assert.equal(
    capture.spans.find((s) => s.name === 'AgentSession.prompt'),
    undefined,
    'the root span must still be open (turn_end is not session end), so it must not export yet',
  );

  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
  await done;
});

test('agent_end sweeps a dangling LLM + TOOL span while stamping (not force-closing, not even ending) the still-open root', async () => {
  // Flipped from the pre-fix model, where agent_end owned closing the root
  // normally (via closeRootSpan). Under the new model agent_end never ends
  // the root at all — it only stamps output.value onto whatever root is
  // currently open (see instrumentation.ts's module header) — so the root
  // stays open (unexported) until the enclosing prompt() call's own promise
  // settles.
  const { capture, Session } = makeRig();
  const session = new Session();

  const done = session.prompt('run that ends with tool/LLM spans still dangling');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'message_start', message: assistantMessage({ model: 'end-llm' }) });
  session.emit({
    type: 'tool_execution_start',
    toolCallId: 'end-tool',
    toolName: 'read_file',
    args: { path: '/tmp/end' },
  });

  // agent_end with the LLM + TOOL spans never having seen their own close:
  // agent_end's defensive sweep force-closes those two, then stamps output
  // onto the root — WITHOUT ending it.
  session.emit({
    type: 'agent_end',
    messages: [assistantMessage({ content: [{ type: 'text', text: 'stamped output' }] })],
    willRetry: false,
  });

  assert.equal(capture.spans.length, 2, 'only the LLM and TOOL spans export so far');
  const llmSpan = capture.spans.find((s) => attrs(s)['gen_ai.request.model'] === 'end-llm');
  const toolSpan = capture.spans.find((s) => attrs(s)['gen_ai.tool.call.id'] === 'end-tool');
  assert.ok(llmSpan && toolSpan);
  assert.equal(
    attrs(llmSpan!)['traceroot.pi.force_closed'],
    true,
    'the dangling LLM span must be force-closed by agent_end’s sweep',
  );
  assert.equal(
    attrs(toolSpan!)['traceroot.pi.force_closed'],
    true,
    'the dangling TOOL span must be force-closed by agent_end’s sweep',
  );
  assert.equal(
    capture.spans.find((s) => s.name === 'AgentSession.prompt'),
    undefined,
    'the root span must still be open — agent_end only stamps it, never ends it',
  );

  await done;

  const rootSpan = capture.spans.find((s) => s.name === 'AgentSession.prompt');
  assert.ok(rootSpan, 'the root span exports once prompt() settles');
  assert.equal(attrs(rootSpan!)['output.value'], 'stamped output');
  assert.notEqual(
    attrs(rootSpan!)['traceroot.pi.force_closed'],
    true,
    'the root span must be closed normally when prompt() settles, not force-closed',
  );
});

test('dispose() mid-run force-closes the root + LLM + TOOL spans', async () => {
  const { capture, Session } = makeRig();
  const session = new Session();

  session.prompt('run torn down mid-flight');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'message_start', message: assistantMessage({ model: 'dispose-llm' }) });
  session.emit({
    type: 'tool_execution_start',
    toolCallId: 'dispose-tool',
    toolName: 'read_file',
    args: { path: '/tmp/dispose' },
  });
  assert.equal(capture.spans.length, 0, 'nothing exports while the run is still open');

  // dispose() before agent_end sweeps everything still open, root included.
  session.dispose();

  assert.equal(
    capture.spans.length,
    3,
    'dispose() must force-close the root, LLM, and TOOL spans (found ' + `${capture.spans.length})`,
  );
  const rootSpan = capture.spans.find((s) => s.name === 'AgentSession.prompt');
  const llmSpan = capture.spans.find((s) => attrs(s)['gen_ai.request.model'] === 'dispose-llm');
  const toolSpan = capture.spans.find((s) => attrs(s)['gen_ai.tool.call.id'] === 'dispose-tool');
  assert.ok(rootSpan && llmSpan && toolSpan);
  for (const span of [rootSpan!, llmSpan!, toolSpan!]) {
    assert.equal(
      attrs(span)['traceroot.pi.force_closed'],
      true,
      `${span.name} must be marked force_closed by dispose()’s sweep`,
    );
  }
});

// NEW under the prompt()-anchored model: the root is now also swept at
// OVERLAP time — a second prompt() call arriving while a previous window's
// root is still open (rare; the real SDK's own isStreaming guard throws for
// most overlaps, but this is not verified to cover every path) force-closes
// the stale window instead of silently overwriting state.rootSpan and
// leaking it unended forever.
test('a second prompt() call while the first window’s root is still open force-closes the first root (and its dangling LLM/TOOL), then opens a fresh root for the second', async () => {
  const { capture, Session } = makeRig();
  const session = new Session();

  const done1 = session.prompt('first window never got its agent_end');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'message_start', message: assistantMessage({ model: 'overlap-llm' }) });
  assert.equal(capture.spans.length, 0, 'nothing exports while the first root is still open');

  // A second prompt() call arrives before the first window's root ever
  // closed — the OVERLAP SAFETY sweep in proto.prompt must force-close the
  // first root (and its dangling LLM span) before opening a fresh one.
  const done2 = session.prompt('second window');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
  await done2;
  // The first window's promise never settles on its own (its agent_end never
  // fired) — this test only needs to observe the sweep's effect on spans, so
  // it deliberately leaves done1 unawaited/unresolved.
  void done1;

  const rootSpans = capture.spans.filter((s) => attrs(s)['openinference.span.kind'] === 'AGENT');
  assert.equal(rootSpans.length, 2, 'both the swept first root and the second root export');
  const firstRoot = rootSpans.find(
    (s) => attrs(s)['input.value'] === 'first window never got its agent_end',
  );
  const secondRoot = rootSpans.find((s) => attrs(s)['input.value'] === 'second window');
  assert.ok(firstRoot, 'the first (overlapped) root must still export');
  assert.ok(secondRoot, 'the second root must export normally');
  assert.equal(
    attrs(firstRoot!)['traceroot.pi.force_closed'],
    true,
    'the first root must be marked force_closed by the overlap sweep',
  );
  assert.notEqual(
    attrs(secondRoot!)['traceroot.pi.force_closed'],
    true,
    'the second root must close normally via its own prompt() settle',
  );
  const overlapLlm = capture.spans.find((s) => attrs(s)['gen_ai.request.model'] === 'overlap-llm');
  assert.ok(overlapLlm, 'the first window’s dangling LLM span must also be swept, not leaked');
  assert.equal(attrs(overlapLlm!)['traceroot.pi.force_closed'], true);
  assert.notEqual(
    firstRoot!.spanContext().traceId,
    secondRoot!.spanContext().traceId,
    'the two overlapping windows must live in genuinely separate traces',
  );
});

// F1 follow-up: a mid-stream steer/followUp call must NOT be treated as an
// overlap. Verified against the real, installed
// @earendil-works/pi-coding-agent@0.80.6 (dist/core/agent-session.js, its own
// `if (this.isStreaming) { if (!options?.streamingBehavior) throw ...; ...
// return; }` branch): when isStreaming is true AND the caller passes
// streamingBehavior, prompt() queues into the ACTIVE run and returns early —
// it never starts a new run and never throws. Before this fix, proto.prompt
// had no way to distinguish that shape from a genuinely-new overlapping
// prompt() call, so it force-closed the still-open ACTIVE root (and its
// live LLM span) out from under the run that was still legitimately in
// progress. This test drives exactly that scenario and asserts the active
// trace survives intact.
test('a mid-stream steer (isStreaming===true, streamingBehavior set) never opens a fresh root or sweeps the active run’s still-open root — the active trace stays intact', async () => {
  const { capture, Session } = makeRig();
  const session = new Session();

  const done = session.prompt('first task, still running');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'message_start', message: assistantMessage({ model: 'active-llm' }) });

  // The run is now actively streaming (mirrors the real SDK's own
  // isStreaming getter). A mid-stream steer call in this state must be
  // detected and delegated straight through — no new root, no overlap sweep.
  session.isStreaming = true;
  await session.prompt('steer text', { streamingBehavior: 'steer' });

  assert.equal(
    capture.spans.length,
    0,
    'the mid-stream steer call must not force-close or export anything — the active run’s root ' +
      'and LLM span are still genuinely open',
  );

  // The active run continues and finishes normally afterward.
  session.isStreaming = false;
  session.emit({
    type: 'message_end',
    message: assistantMessage({
      model: 'active-llm',
      content: [{ type: 'text', text: 'steered result' }],
    }),
  });
  session.emit({ type: 'turn_end', message: assistantMessage(), toolResults: [] });
  session.emit({
    type: 'agent_end',
    messages: [assistantMessage({ content: [{ type: 'text', text: 'steered result' }] })],
    willRetry: false,
  });
  await done;

  const rootSpans = capture.spans.filter((s) => attrs(s)['openinference.span.kind'] === 'AGENT');
  assert.equal(
    rootSpans.length,
    1,
    'exactly one intact root for the whole run — the steer call never opened a second one',
  );
  const root = rootSpans[0]!;
  assert.notEqual(
    attrs(root)['traceroot.pi.force_closed'],
    true,
    'the active root must NOT be force-closed by the mid-stream steer',
  );
  assert.equal(attrs(root)['output.value'], 'steered result');

  const llmSpan = capture.spans.find((s) => attrs(s)['openinference.span.kind'] === 'LLM');
  assert.ok(llmSpan, 'the active run’s LLM span must still export normally');
  assert.notEqual(
    attrs(llmSpan!)['traceroot.pi.force_closed'],
    true,
    'the active LLM span must not have been swept by the steer call',
  );
  assert.equal(
    llmSpan!.parentSpanId,
    root.spanContext().spanId,
    'the LLM span must remain a child of the single intact root',
  );
});

// F3 follow-up: claude-agent-sdk.ts parity (see its own endInFlight(),
// called from wrapQuery's finish()). Before this fix, finalize (the settle
// path that ends the root when prompt()'s own promise resolves/rejects) only
// ended the root span — a mid-run rejection with a tool/LLM span still open
// left that span never .end()ed, and a span that never has .end() called on
// it is never exported at all: silently dropped, not merely "left open".
test('a prompt() call that REJECTS mid-run force-closes a still-open tool span before finalizing the root as ERROR (claude-agent-sdk.ts endInFlight parity)', async () => {
  const { capture, Session } = makeRig();
  const session = new Session();

  const done = session.prompt('a run whose internal loop rejects while a tool call is still open');
  session.emit({ type: 'agent_start' });
  session.emit({
    type: 'tool_execution_start',
    toolCallId: 'reject-tool',
    toolName: 'bash',
    args: { command: 'sleep 999' },
  });
  assert.equal(capture.spans.length, 0, 'nothing exports while the run is still open');

  session.rejectPrompt(new Error('internal agent loop failure'));
  await assert.rejects(() => done, /internal agent loop failure/);

  assert.equal(
    capture.spans.length,
    2,
    'the still-open tool span AND the root must both export once the rejection settles ' +
      `(found ${capture.spans.length})`,
  );
  const rootSpan = capture.spans.find((s) => s.name === 'AgentSession.prompt');
  const toolSpan = capture.spans.find((s) => attrs(s)['gen_ai.tool.call.id'] === 'reject-tool');
  assert.ok(rootSpan);
  assert.ok(toolSpan, 'the dangling tool span must still export, not be dropped forever');
  assert.equal(
    attrs(toolSpan!)['traceroot.pi.force_closed'],
    true,
    'the tool span must be force-closed by finalize’s pre-close sweep',
  );
  assert.equal(
    toolSpan!.parentSpanId,
    rootSpan!.spanContext().spanId,
    'the force-closed tool span must still be parented under the root',
  );

  assert.equal(rootSpan!.status.code, 2 /* SpanStatusCode.ERROR */);
  assert.equal(
    rootSpan!.events.some((e) => e.name === 'exception'),
    true,
    'the rejection reason must be recorded as an exception on the root span',
  );
});
