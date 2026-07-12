/**
 * agent_end's re-queue of pendingInputText must not be gated on
 * `event.willRetry` alone.
 *
 * Verified against the real, installed @earendil-works/pi-coding-agent@0.80.6
 * (dist/core/agent-session.js): AgentSession._runAgentPrompt() runs
 * `while (await this._handlePostAgentRun()) { await this.agent.continue(); }`,
 * and `_handlePostAgentRun()` returns true for THREE independent reasons —
 * (1) a retryable error (the only one reflected in `event.willRetry`, which
 * `_willRetryAfterAgentEnd` computes purely from `_isRetryableError` on the
 * last assistant message — and `_isRetryableError` explicitly EXCLUDES
 * context-overflow: "Context overflow is handled by compaction, not retry"),
 * (2) `_checkCompaction(msg)` returning true — an ordinary *successful*
 * response whose context just crossed the compaction threshold, a routine,
 * default-enabled path in any long session, not an error — and (3)
 * `this.agent.hasQueuedMessages()`, i.e. an extension queued a follow-up
 * during its own agent_end handler. `agent.continue()` emits a fresh
 * `agent_start` unconditionally in every case, with willRetry never having
 * been true for (2) or (3).
 *
 * Before the fix: instrumentation.ts's agent_end case only re-queued
 * state.pendingInputText when `event.willRetry` was true, so a
 * compaction-driven (or extension-queued-follow-up-driven) continuation
 * silently discarded the input text instead of handing it to the
 * immediately-following phantom agent_start — leaving that run's root span
 * with input.value === undefined even though it is semantically still the
 * same in-flight session.prompt() call.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { assistantMessage, attrs, CapturingExporter, makeFakeSessionClass } from './test-helpers';
import { instrumentPiCodingAgent } from '../src/instrumentation';

test('an agent_end continuation with willRetry: false (auto-compaction or an extension-queued follow-up) still reuses the same input text on its phantom agent_start, which fires with no new prompt() call in between', async () => {
  const capture = new CapturingExporter();
  const Session = makeFakeSessionClass();
  const sdk = { AgentSession: Session };
  instrumentPiCodingAgent(sdk, { apiKey: 'test-key', _spanExporter: capture });
  const session = new Session();

  await session.prompt('a long task that crosses the compaction threshold');
  session.emit({ type: 'agent_start' });
  // The turn succeeds normally and crosses the auto-compaction threshold (or
  // an extension queues a follow-up from its agent_end handler) — either
  // way _handlePostAgentRun() returns true for a reason that has nothing to
  // do with the retry heuristic, so willRetry stays false even though
  // agent.continue() is about to fire a fresh agent_start.
  session.emit({
    type: 'agent_end',
    messages: [assistantMessage({ content: [{ type: 'text', text: 'partial progress' }] })],
    willRetry: false,
  });
  // The phantom continuation: agent_start fires again with NO new prompt()
  // call in between.
  session.emit({ type: 'agent_start' });
  session.emit({
    type: 'agent_end',
    messages: [assistantMessage({ content: [{ type: 'text', text: 'final result' }] })],
    willRetry: false,
  });

  const rootSpans = capture.spans.filter((s) => attrs(s)['openinference.span.kind'] === 'AGENT');
  assert.equal(rootSpans.length, 2, 'each agent_start gets its own root span');
  assert.equal(
    attrs(rootSpans[0]!)['input.value'],
    'a long task that crosses the compaction threshold',
    'the first attempt must carry the original prompt text',
  );
  assert.equal(
    attrs(rootSpans[1]!)['input.value'],
    'a long task that crosses the compaction threshold',
    'the continuation root span must still carry the SAME input text even though willRetry was ' +
      'false — the old code only re-queued pendingInputText when event.willRetry was true, so a ' +
      'compaction/extension-follow-up-driven continuation lost it entirely (input.value would be ' +
      'undefined here under the pre-fix code)',
  );
  assert.equal(attrs(rootSpans[1]!)['output.value'], 'final result');

  // A genuine follow-up prompt() call after the continuation finishes must
  // still get its own fresh text — proving the reuse does not leak forward
  // and make a later, unrelated prompt() replay the old text again.
  await session.prompt('a brand new followup prompt');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });

  const allRootSpans = capture.spans.filter((s) => attrs(s)['openinference.span.kind'] === 'AGENT');
  assert.equal(allRootSpans.length, 3);
  assert.equal(
    attrs(allRootSpans[2]!)['input.value'],
    'a brand new followup prompt',
    'after the continuation finished, a genuine new prompt() call must not accidentally replay ' +
      'the old reused text a third time',
  );
});
