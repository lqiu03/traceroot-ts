/**
 * Regression tests for three bugs (numbered 6-8 in the review that found
 * them) confirmed against src/instrumentation.ts, each structured to fail
 * against the pre-fix code and pass against the fix. This file originally
 * held nine such tests with no coherent subject; the other six (bugs 1-5,
 * with bug 5 covering two tests) were redistributed to their proper
 * subject-matter files:
 *  - Bug 1 (beforeExit forceFlush) -> provider-shared-mode-behavior.test.ts
 *  - Bug 2 (second message_start force-closes the abandoned first LLM span)
 *    -> span-lifecycle-event-ordering.test.ts
 *  - Bug 3 (overlapping prompt() calls, FIFO) and Bug 4 (retried run reuses
 *    the same input text) both exercised the now-deleted per-session
 *    PromptQueue (prompt-queue.ts and its dedicated prompt-queue.test.ts are
 *    both removed — see this change's root span re-anchoring). Their real
 *    intent lives on in different, still-relevant forms: overlapping
 *    prompt() calls are now covered by
 *    dangling-span-sweep-deduplication.test.ts's overlap-safety test, and a
 *    retried run sharing one root/input.value is covered by
 *    instrumentation-edge-cases.test.ts's retry test.
 *  - Bug 5 (stray events never parent under an ambient active span, both
 *    variants) -> span-context-parenting.test.ts
 * Bugs 6-8 don't share a subject with any other file, so they stay here.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { hrTime, hrTimeToMilliseconds } from '@opentelemetry/core';
import { instrumentPiCodingAgent } from '../src/pi/instrumentation';
import { makeFakeSessionClass, makeRig, assistantMessage, attrs } from './pi-test-helpers';

// Bug 6 -------------------------------------------------------------------

test('agent_start force-closes an orphaned tool span left over from a stray event even when rootSpan was already undefined, instead of leaving it open through the entire next run', async () => {
  const { capture, Session } = makeRig();
  const session = new Session();

  const done1 = session.prompt('run 1 finishes cleanly');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
  // Reworked for the new model: run 1's prompt() call must actually SETTLE
  // (awaited here) before the stray event, so its root is genuinely gone —
  // under the old agent_start-anchored root, agent_end alone cleared
  // rootSpan; under the new prompt()-anchored root it only clears once this
  // call's own promise settles (see instrumentation.ts's module header).
  await done1;

  // Stray tool_execution_start after run 1 fully settled — rootSpan is
  // already undefined here, so the old "if (state.rootSpan)" gate would skip
  // sweeping this orphan at the NEXT agent_start entirely.
  session.emit({
    type: 'tool_execution_start',
    toolCallId: 'orphan',
    toolName: 'bash',
    args: { command: 'echo orphan' },
  });

  // Run 2 starts via a genuinely NEW prompt() call: agent_start alone no
  // longer fabricates a root under the new model's rootless-bypass boundary
  // policy (see instrumentation.ts's module header), so "run 2" must be a
  // real second prompt() call here, not just another bare agent_start.
  const done2 = session.prompt('run 2 prompt text');
  session.emit({ type: 'agent_start' });
  // A real ~30ms gap before run 2 does its own work. If the orphan is only
  // swept at run 2's agent_end (the bug), message_start/message_end/agent_end
  // all fire back to back AFTER this gap, so the orphan's endTime lands only
  // a fraction of a millisecond before/after run 2's LLM span opens — too
  // close to distinguish from clock jitter. Comparing against run 2's own
  // agent_start moment (captured via its root span's startTime, set BEFORE
  // the gap) instead gives a real ~30ms margin: under the fix, the orphan is
  // swept as part of THAT agent_start call, so its endTime must land at or
  // before run 2's root span opens, not ~30ms+ later.
  // Captured via OTel's own hrTime() — the same clock ReadableSpan
  // start/end times use — not process.hrtime(), which is a different,
  // arbitrary-origin monotonic clock and not directly comparable to it.
  const preGapTimestamp = hrTime();
  await new Promise((resolve) => setTimeout(resolve, 30));
  session.emit({ type: 'message_start', message: assistantMessage() });
  session.emit({ type: 'message_end', message: assistantMessage() });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
  await done2;

  const orphanSpan = capture.spans.find((s) => attrs(s)['gen_ai.tool.call.id'] === 'orphan');
  const run2Llm = capture.spans.find((s) => attrs(s)['openinference.span.kind'] === 'LLM');
  assert.ok(orphanSpan, 'the orphaned span must still be exported eventually');
  assert.ok(run2Llm);
  assert.equal(
    attrs(orphanSpan!)['traceroot.pi.force_closed'],
    true,
    'must be marked as abnormally closed',
  );
  const orphanClosedBeforeGapMs =
    hrTimeToMilliseconds(preGapTimestamp) - hrTimeToMilliseconds(orphanSpan!.endTime);
  // preGapTimestamp and orphanSpan.endTime are two independent hrTime() reads
  // taken microseconds apart (endTime is stamped inside the sweep during run
  // 2's agent_start; preGapTimestamp right after that emit returns). Under
  // full-suite parallel load their sub-millisecond rounding/jitter can make the
  // later-read value round marginally below the earlier one, so a zero-tolerance
  // `>= 0` compare flakes (observed once: "closed 0.051ms after the gap
  // started"). Allow a small tolerance far below the 30ms gap this
  // discriminates against, so the assertion still fails hard for the real bug
  // (orphan swept only at run 2's agent_end, ~30ms+ later) while never flaking
  // on clock-read jitter.
  const CLOCK_JITTER_TOLERANCE_MS = 5;
  assert.ok(
    orphanClosedBeforeGapMs >= -CLOCK_JITTER_TOLERANCE_MS,
    'the orphaned tool span must be force-closed at run 2s agent_start (before the 30ms gap), ' +
      `not left dangling open through the entirety of run 2 (closed ${-orphanClosedBeforeGapMs}ms ` +
      'after the gap started, which only happens if it waited for run 2s agent_end instead)',
  );
});

// Bug 7 -------------------------------------------------------------------

test('turn_end force-closes any tool spans still open at the end of the turn instead of leaving them dangling until agent_end', async () => {
  const { capture, Session } = makeRig();
  const session = new Session();

  const done = session.prompt('a tool call never gets its tool_execution_end before the turn ends');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'message_start', message: assistantMessage() });
  session.emit({ type: 'message_end', message: assistantMessage() });
  session.emit({
    type: 'tool_execution_start',
    toolCallId: 'never-closes',
    toolName: 'bash',
    args: { command: 'sleep 999' },
  });
  // turn_end fires with the tool call still open — no tool_execution_end.
  session.emit({ type: 'turn_end', message: assistantMessage(), toolResults: [] });

  const toolSpanAtTurnEnd = capture.spans.find(
    (s) => attrs(s)['gen_ai.tool.call.id'] === 'never-closes',
  );
  assert.ok(
    toolSpanAtTurnEnd,
    'the tool span must already be force-closed and exported by turn_end, not deferred to agent_end',
  );
  assert.equal(attrs(toolSpanAtTurnEnd!)['traceroot.pi.force_closed'], true);

  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
  await done;
  const toolSpans = capture.spans.filter((s) => attrs(s)['gen_ai.tool.call.id'] === 'never-closes');
  assert.equal(
    toolSpans.length,
    1,
    'the tool span must be exported exactly once — turn_end must also clear it from ' +
      'state.toolSpans so agent_end does not try to force-close it a second time',
  );
});

// Bug 8 -------------------------------------------------------------------

test('a second instrumentPiCodingAgent() call on an already-wrapped sdk logs a console.warn instead of silently ignoring its config', () => {
  const Session = makeFakeSessionClass();
  const sdk = { AgentSession: Session };

  instrumentPiCodingAgent(sdk, {});

  const warnings: unknown[][] = [];
  const originalWarn = console.warn;
  console.warn = (...args: unknown[]) => {
    warnings.push(args);
  };
  try {
    instrumentPiCodingAgent(sdk, { captureContent: false });
  } finally {
    console.warn = originalWarn;
  }

  assert.ok(
    warnings.some((args) => typeof args[0] === 'string' && /already (been )?called/.test(args[0])),
    'the second call must console.warn that instrumentation was already set up and this config ' +
      "is being ignored, matching the function's other two early-return paths",
  );
});
