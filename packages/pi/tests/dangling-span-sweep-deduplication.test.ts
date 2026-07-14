/**
 * Behavioral guard that the "force-close every open tool span, then the LLM
 * span, then (sometimes) the root span" dangling-span sweep actually runs at
 * all 4 call sites in packages/pi/src/instrumentation.ts — agent_start,
 * turn_end, agent_end, and dispose().
 *
 * This finding was originally guarded by a source-text regex (assert the sweep
 * helper is defined once and called 4 times). That caught the structural
 * duplication risk but broke on harmless refactors and never exercised the
 * actual behavior. These tests instead drive each of the 4 call sites into a
 * real dangling-span scenario through the public API and assert the observable
 * force-close result, mirroring session-dispose.test.ts's own dispose()
 * pattern — so a real future regression (a site quietly dropping its sweep) is
 * caught by behavior, and a benign rename/extraction of the helper is not.
 *
 * Sweep scope differs by site, and each test pins that down:
 *   - agent_start & dispose(): sweep the root span too (includeRoot) — a new
 *     run starting, or the session being torn down, ends the old root.
 *   - turn_end & agent_end: sweep only the dangling LLM/TOOL spans; the root
 *     span is NOT force-closed (turn_end isn't session end; agent_end closes
 *     the root normally, not as a force-close).
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { assistantMessage, attrs, makeRig } from './test-helpers';

test('agent_start force-closes a previous abandoned run (root + LLM + TOOL) when a fresh run starts', async () => {
  const { capture, Session } = makeRig();
  const session = new Session();

  // Run one, left mid-flight: root + LLM + TOOL all open, and agent_end never
  // fires for it.
  await session.prompt('run one');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'message_start', message: assistantMessage({ model: 'run1-llm' }) });
  session.emit({
    type: 'tool_execution_start',
    toolCallId: 'run1-tool',
    toolName: 'read_file',
    args: { path: '/tmp/one' },
  });
  assert.equal(capture.spans.length, 0, 'nothing exports while run one is still open');

  // A fresh agent_start (run two, no new prompt() needed — e.g. the loop
  // crashed and restarted) must sweep run one's dangling spans.
  session.emit({ type: 'agent_start' });

  assert.equal(
    capture.spans.length,
    3,
    'agent_start must force-close run one’s root, LLM, and TOOL spans (found ' +
      `${capture.spans.length})`,
  );
  const rootSpan = capture.spans.find((s) => s.name === 'AgentSession.prompt');
  const llmSpan = capture.spans.find((s) => attrs(s)['gen_ai.request.model'] === 'run1-llm');
  const toolSpan = capture.spans.find((s) => attrs(s)['gen_ai.tool.call.id'] === 'run1-tool');
  assert.ok(rootSpan && llmSpan && toolSpan, 'all three of run one’s spans must be exported');
  for (const span of [rootSpan!, llmSpan!, toolSpan!]) {
    assert.equal(
      attrs(span)['traceroot.pi.force_closed'],
      true,
      `${span.name} must be marked force_closed by agent_start’s sweep`,
    );
  }
});

test('turn_end force-closes a dangling LLM + TOOL span but leaves the root span open', async () => {
  const { capture, Session } = makeRig();
  const session = new Session();

  await session.prompt('run with a stream error mid-turn');
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
  session.emit({ type: 'turn_end' });

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
});

test('agent_end sweeps a dangling LLM + TOOL span while closing the root span normally (not force-closed)', async () => {
  const { capture, Session } = makeRig();
  const session = new Session();

  await session.prompt('run that ends with tool/LLM spans still dangling');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'message_start', message: assistantMessage({ model: 'end-llm' }) });
  session.emit({
    type: 'tool_execution_start',
    toolCallId: 'end-tool',
    toolName: 'read_file',
    args: { path: '/tmp/end' },
  });

  // agent_end with the LLM + TOOL spans never having seen their own close:
  // agent_end's defensive sweep force-closes those two, then closes the root
  // span through its normal closeRootSpan() path.
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });

  assert.equal(capture.spans.length, 3, 'the root, LLM, and TOOL spans must all export');
  const rootSpan = capture.spans.find((s) => s.name === 'AgentSession.prompt');
  const llmSpan = capture.spans.find((s) => attrs(s)['gen_ai.request.model'] === 'end-llm');
  const toolSpan = capture.spans.find((s) => attrs(s)['gen_ai.tool.call.id'] === 'end-tool');
  assert.ok(rootSpan && llmSpan && toolSpan);
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
  assert.notEqual(
    attrs(rootSpan!)['traceroot.pi.force_closed'],
    true,
    'the root span must be closed normally by agent_end, not force-closed by the sweep',
  );
});

test('dispose() mid-run force-closes the root + LLM + TOOL spans', async () => {
  const { capture, Session } = makeRig();
  const session = new Session();

  await session.prompt('run torn down mid-flight');
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
