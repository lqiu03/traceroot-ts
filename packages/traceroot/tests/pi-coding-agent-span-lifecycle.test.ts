import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { hrTime, hrTimeToMilliseconds } from '@opentelemetry/core';
import { assistantMessage, attrs, makeRig } from './pi-test-helpers';
import { context, trace, ROOT_CONTEXT, TraceFlags } from '@opentelemetry/api';
import type { Context, ContextManager, SpanContext } from '@opentelemetry/api';
import type { ExportResult } from '@opentelemetry/core';
import { ExportResultCode } from '@opentelemetry/core';
import { SimpleSpanProcessor } from '@opentelemetry/sdk-trace-base';
import type { ReadableSpan, SpanExporter } from '@opentelemetry/sdk-trace-base';
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import { instrumentPiCodingAgent } from '../src/pi/instrumentation';
import type { AgentEvent, AssistantMessage } from '../src/pi/types';

describe('span lifecycle event ordering', () => {
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
   *  - the same stray tool_execution_start scenario, but asserting it is
   *    force-closed AT the next window's agent_start rather than merely
   *    "eventually" (originally confirmed-bugfix-regressions.test.ts's Bug 6);
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
  it('a second message_start with no intervening message_end/turn_end force-closes the abandoned first LLM span instead of silently dropping it', async () => {
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

  it('a stray message_start firing after a clean prompt() call already settled (no matching message_end) is force-closed on the NEXT prompt() call’s agent_start instead of being silently dropped', async () => {
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

    const run2LlmSpan = capture.spans.find(
      (s) => attrs(s)['gen_ai.request.model'] === 'run-2-model',
    );
    assert.ok(run2LlmSpan, 'run 2 must still produce its own normal LLM span');
  });

  it('turn_end firing without a preceding message_end force-closes the LLM span instead of leaking it into the next turn', async () => {
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

  it('tool_execution_start with no prior message_start/message_end at all parents directly under the root span', async () => {
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

  it('a duplicate tool_execution_start for a toolCallId that is already open force-closes the first span instead of silently overwriting the Map and leaking it', async () => {
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

  it('agent_end with zero prior agent_start events still stamps the root that prompt() already opened, and does not throw', async () => {
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

  it('agent_start firing twice with no intervening agent_end, and no new prompt() call between them, shares ONE root and does not leak a stale LLM context into the continuation', async () => {
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
    session.emit({
      type: 'message_start',
      message: assistantMessage({ model: 'attempt-1-model' }),
    });
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

  it('a stray tool_execution_start firing after a clean prompt() call already settled does not corrupt the NEXT prompt() call’s span tree', async () => {
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
        attrs(s)['openinference.span.kind'] === 'AGENT' &&
        s.spanContext().spanId !== run1RootSpanId,
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
    const stragglerSpan = capture.spans.find(
      (s) => attrs(s)['gen_ai.tool.call.id'] === 'straggler',
    );
    assert.ok(stragglerSpan, 'the straggler span is still force-closed eventually, not dropped');
    assert.notEqual(stragglerSpan!.parentSpanId, run2Root!.spanContext().spanId);
    assert.notEqual(stragglerSpan!.parentSpanId, run1RootSpanId);
  });

  it('agent_start force-closes an orphaned tool span left over from a stray event even when rootSpan was already undefined, instead of leaving it open through the entire next run', async () => {
    const { capture, Session } = makeRig();
    const session = new Session();

    const done1 = session.prompt('run 1 finishes cleanly');
    session.emit({ type: 'agent_start' });
    session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
    // Run 1's prompt() call must actually SETTLE (awaited here) before the
    // stray event, so its root is genuinely gone — under the old
    // agent_start-anchored root, agent_end alone cleared rootSpan; under the
    // new prompt()-anchored root it only clears once this call's own promise
    // settles (see instrumentation.ts's module header).
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

  it('a message_end (assistant) whose message_start never arrived is ignored — no crash, no phantom LLM span, and the run still completes normally', async () => {
    const { capture, Session } = makeRig();
    const session = new Session();

    const done = session.prompt(
      'a stream resumes mid-message: only the end event is ever delivered',
    );
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
});

describe('span context parenting', () => {
  /**
   * Lens: span-context-parenting (formerly otel-context-and-span-parenting).
   *
   * Probes OTel Context/parent-span correctness across turn boundaries within
   * a single agent run: whether a turn-2 LLM span can accidentally inherit
   * turn-1's already-cleared llmCtx, whether a tool span opened during turn 1
   * stays correctly bound to turn 1's (immutable) parent Context even after
   * state has moved on to turn 2, whether a tool_execution_start with no
   * preceding message_start for its turn correctly falls back to the root
   * span instead of a stale ended LLM context, and whether span.updateName()
   * in closeLlmSpan actually changes what the exporter captures.
   *
   * Also folds in two tests from the former
   * confirmed-bugfix-regressions.test.ts grab-bag (its Bug 5):
   * whether a stray event with NO rootCtx/llmCtx at all (no agent_start ever
   * fired on the session) parents under whatever span happens to be ambiently
   * active in the host process's own OTel context, instead of correctly
   * starting a fresh standalone trace. Same subject — OTel Context/parent-span
   * correctness — just probing the "nothing is open at all" edge instead of
   * the "something else is open" edges the rest of this file covers.
   */
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
  function makeRig() {
    const capture = new CapturingExporter();

    class FakeAgentSession {
      sessionId = 'sess-1';
      private listeners: Array<(event: AgentEvent) => void> = [];
      // prompt()'s returned promise settles only once its final agent_end
      // fires (willRetry !== true) — mirrors the real SDK; see
      // pi-test-helpers.ts's module header for the full rationale.
      private pending: { resolve: () => void; reject: (err: unknown) => void } | undefined;
      async prompt(_text: string, _options?: unknown): Promise<void> {
        return new Promise<void>((resolve, reject) => {
          this.pending = { resolve, reject };
        });
      }
      // A standalone entry point distinct from prompt() — used by the two
      // "stray event with no root open at all" tests below to attach the span
      // listener WITHOUT opening a root span (steer()/followUp() never do;
      // only prompt() does — see instrumentation.ts's module header on the
      // rootless bypass boundary policy).
      async steer(_text: string): Promise<void> {}
      subscribe(listener: (event: AgentEvent) => void): () => void {
        this.listeners.push(listener);
        return () => {
          this.listeners = this.listeners.filter((l) => l !== listener);
        };
      }
      emit(event: AgentEvent): void {
        for (const listener of this.listeners) listener(event);
        if (event.type === 'agent_end' && !event.willRetry && this.pending) {
          const { resolve } = this.pending;
          this.pending = undefined;
          resolve();
        }
      }
    }

    const sdk = { AgentSession: FakeAgentSession };
    // Real global provider per rig, not a private exporter injection —
    // PiInstrumentationConfig no longer has an apiKey/_spanExporter escape
    // hatch (see packages/traceroot/src/pi/config.ts). trace.disable() first
    // clears any prior rig's registration so this file's tests stay isolated
    // from one another (see pi-test-helpers.ts's makeRig() for the full
    // rationale, mirrored here since this file keeps its own local rig).
    trace.disable();
    const provider = new NodeTracerProvider();
    provider.addSpanProcessor(new SimpleSpanProcessor(capture));
    provider.register();
    instrumentPiCodingAgent(sdk, {});

    return { capture, Session: FakeAgentSession };
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

  // A minimal, real, synchronous ContextManager — needed because
  // @opentelemetry/api's default NoopContextManager makes context.with() a
  // no-op and context.active() always return ROOT_CONTEXT, which would make
  // the ambient-context-contamination bug untestable (it would look "fixed"
  // even against the buggy code, since context.active() and ROOT_CONTEXT are
  // otherwise indistinguishable without a manager registered). Registering
  // this reproduces what a host app's own real OTel setup (e.g.
  // AsyncHooksContextManager) does.
  class StackContextManager implements ContextManager {
    private stack: Context[] = [ROOT_CONTEXT];
    active(): Context {
      return this.stack[this.stack.length - 1]!;
    }
    with<A extends unknown[], F extends (...args: A) => ReturnType<F>>(
      ctx: Context,
      fn: F,
      thisArg?: ThisParameterType<F>,
      ...args: A
    ): ReturnType<F> {
      this.stack.push(ctx);
      try {
        return fn.call(thisArg, ...args);
      } finally {
        this.stack.pop();
      }
    }
    bind<T>(_ctx: Context, target: T): T {
      return target;
    }
    enable(): this {
      return this;
    }
    disable(): this {
      return this;
    }
  }

  const AMBIENT_SPAN_CONTEXT: SpanContext = {
    traceId: 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
    spanId: 'aaaaaaaaaaaaaaaa',
    traceFlags: TraceFlags.SAMPLED,
  };

  it('two back-to-back turns in one agent run each get their own LLM span parented under the shared root, not under each other', async () => {
    const { capture, Session } = makeRig();
    const session = new Session();

    const done = session.prompt('do two turns');
    session.emit({ type: 'agent_start' });

    // Turn 1.
    session.emit({ type: 'turn_start' });
    session.emit({ type: 'message_start', message: assistantMessage({ model: 'turn-1-model' }) });
    session.emit({ type: 'message_end', message: assistantMessage({ model: 'turn-1-model' }) });
    session.emit({ type: 'turn_end', message: assistantMessage(), toolResults: [] });

    // Turn 2 — fires immediately after turn 1 fully closed (message_end AND
    // turn_end both already ran, so state.llmSpan/state.llmCtx are cleared).
    session.emit({ type: 'turn_start' });
    session.emit({ type: 'message_start', message: assistantMessage({ model: 'turn-2-model' }) });
    session.emit({
      type: 'tool_execution_start',
      toolCallId: 'turn2-tool',
      toolName: 'bash',
      args: { command: 'echo turn2' },
    });
    session.emit({
      type: 'tool_execution_end',
      toolCallId: 'turn2-tool',
      toolName: 'bash',
      result: {},
      isError: false,
    });
    session.emit({ type: 'message_end', message: assistantMessage({ model: 'turn-2-model' }) });
    session.emit({ type: 'turn_end', message: assistantMessage(), toolResults: [] });

    session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
    await done;

    const rootSpan = capture.spans.find((s) => attrs(s)['openinference.span.kind'] === 'AGENT');
    const llmSpans = capture.spans.filter((s) => attrs(s)['openinference.span.kind'] === 'LLM');
    assert.ok(rootSpan);
    assert.equal(llmSpans.length, 2, 'each turn gets its own LLM span');

    const turn1Llm = llmSpans.find((s) => attrs(s)['gen_ai.request.model'] === 'turn-1-model');
    const turn2Llm = llmSpans.find((s) => attrs(s)['gen_ai.request.model'] === 'turn-2-model');
    assert.ok(turn1Llm);
    assert.ok(turn2Llm);
    assert.notEqual(
      turn1Llm!.spanContext().spanId,
      turn2Llm!.spanContext().spanId,
      'turn 1 and turn 2 must be genuinely distinct spans',
    );

    // Both LLM spans must parent directly under the single shared root span —
    // turn 2's LLM span must NOT inherit turn 1's already-cleared llmCtx (it
    // has no reason to chain off turn 1 at all; the root is the only valid
    // parent for any turn's LLM span).
    assert.equal(turn1Llm!.parentSpanId, rootSpan!.spanContext().spanId);
    assert.equal(turn2Llm!.parentSpanId, rootSpan!.spanContext().spanId);

    // Turn 2's tool call (opened after turn 2's own message_start) must parent
    // under turn 2's LLM span, never under turn 1's already-closed LLM span.
    const turn2Tool = capture.spans.find((s) => attrs(s)['gen_ai.tool.call.id'] === 'turn2-tool');
    assert.ok(turn2Tool);
    assert.equal(
      turn2Tool!.parentSpanId,
      turn2Llm!.spanContext().spanId,
      "turn 2's tool call must parent under turn 2's LLM span",
    );
    assert.notEqual(
      turn2Tool!.parentSpanId,
      turn1Llm!.spanContext().spanId,
      "turn 2's tool call must not accidentally inherit turn 1's stale llmCtx",
    );
  });

  it('a tool span opened during turn 1 keeps its parent bound to turn 1s LLM span (OTel Context is captured immutably at open time) even after turn 2 has already opened its own LLM span before turn 1s tool call closes', async () => {
    const { capture, Session } = makeRig();
    const session = new Session();

    const done = session.prompt('turn 1s tool call resolves late, after turn 2 already started');
    session.emit({ type: 'agent_start' });

    // Turn 1: message_end already closed the LLM span content-wise, but
    // llmCtx is deliberately kept alive (see instrumentation.ts's own comment
    // on message_end) so a tool call fired in the grace window before
    // turn_end still parents under turn 1's LLM span.
    session.emit({ type: 'turn_start' });
    session.emit({ type: 'message_start', message: assistantMessage({ model: 'turn-1-model' }) });
    session.emit({ type: 'message_end', message: assistantMessage({ model: 'turn-1-model' }) });
    session.emit({
      type: 'tool_execution_start',
      toolCallId: 'turn1-tool',
      toolName: 'bash',
      args: { command: 'echo turn1' },
    });
    // turn_end now clears state.llmSpan/state.llmCtx entirely before turn 1's
    // tool call has been closed — simulating a tool whose completion event is
    // slow to arrive relative to the rest of the turn's lifecycle events.
    session.emit({ type: 'turn_end', message: assistantMessage(), toolResults: [] });

    // Turn 2 starts and opens its own LLM span — this overwrites
    // state.llmSpan/state.llmCtx to point at turn 2 entirely.
    session.emit({ type: 'turn_start' });
    session.emit({ type: 'message_start', message: assistantMessage({ model: 'turn-2-model' }) });

    // Only now does turn 1's tool call actually finish.
    session.emit({
      type: 'tool_execution_end',
      toolCallId: 'turn1-tool',
      toolName: 'bash',
      result: {},
      isError: false,
    });

    session.emit({ type: 'message_end', message: assistantMessage({ model: 'turn-2-model' }) });
    session.emit({ type: 'turn_end', message: assistantMessage(), toolResults: [] });
    session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
    await done;

    const llmSpans = capture.spans.filter((s) => attrs(s)['openinference.span.kind'] === 'LLM');
    const turn1Llm = llmSpans.find((s) => attrs(s)['gen_ai.request.model'] === 'turn-1-model');
    const turn2Llm = llmSpans.find((s) => attrs(s)['gen_ai.request.model'] === 'turn-2-model');
    assert.ok(turn1Llm);
    assert.ok(turn2Llm);

    const turn1Tool = capture.spans.find((s) => attrs(s)['gen_ai.tool.call.id'] === 'turn1-tool');
    assert.ok(turn1Tool);
    assert.equal(
      turn1Tool!.parentSpanId,
      turn1Llm!.spanContext().spanId,
      "turn 1's tool span must stay bound to turn 1's LLM span — the parent Context was captured at " +
        'tool_execution_start time and is immutable, so it must not silently move onto turn 2 just ' +
        'because state.llmCtx has since been reassigned',
    );
    assert.notEqual(
      turn1Tool!.parentSpanId,
      turn2Llm!.spanContext().spanId,
      "turn 1's tool span must never parent under turn 2's LLM span",
    );
  });

  it('tool_execution_start firing after turn_end cleared llmCtx but before the next turns message_start falls back to the root span, not a stale ended LLM context', async () => {
    const { capture, Session } = makeRig();
    const session = new Session();

    const done = session.prompt('a tool fires in the gap between two turns');
    session.emit({ type: 'agent_start' });

    // Turn 1 completes fully, including turn_end — state.llmSpan/state.llmCtx
    // are now both cleared back to undefined.
    session.emit({ type: 'turn_start' });
    session.emit({ type: 'message_start', message: assistantMessage({ model: 'turn-1-model' }) });
    session.emit({ type: 'message_end', message: assistantMessage({ model: 'turn-1-model' }) });
    session.emit({ type: 'turn_end', message: assistantMessage(), toolResults: [] });

    // A tool call fires in the gap before turn 2's message_start ever arrives
    // (e.g. a housekeeping/background tool the agent loop runs between turns).
    assert.doesNotThrow(() => {
      session.emit({
        type: 'tool_execution_start',
        toolCallId: 'gap-tool',
        toolName: 'bash',
        args: { command: 'echo gap' },
      });
      session.emit({
        type: 'tool_execution_end',
        toolCallId: 'gap-tool',
        toolName: 'bash',
        result: {},
        isError: false,
      });
    });

    session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
    await done;

    const rootSpan = capture.spans.find((s) => attrs(s)['openinference.span.kind'] === 'AGENT');
    const turn1Llm = capture.spans.find(
      (s) =>
        attrs(s)['openinference.span.kind'] === 'LLM' &&
        attrs(s)['gen_ai.request.model'] === 'turn-1-model',
    );
    const gapTool = capture.spans.find((s) => attrs(s)['gen_ai.tool.call.id'] === 'gap-tool');
    assert.ok(rootSpan);
    assert.ok(turn1Llm);
    assert.ok(gapTool);

    assert.equal(
      gapTool!.parentSpanId,
      rootSpan!.spanContext().spanId,
      'with no LLM span currently open, the gap tool call must fall back to the root span',
    );
    assert.notEqual(
      gapTool!.parentSpanId,
      turn1Llm!.spanContext().spanId,
      'the gap tool call must not parent under turn 1s already-ended LLM span just because it was the ' +
        'most recently active one',
    );
  });

  it('closeLlmSpan span.updateName() changes the name the exporter actually captures — the final response model wins over the initial request model', async () => {
    const { capture, Session } = makeRig();
    const session = new Session();

    const done = session.prompt('the provider renames the model between request and response');
    session.emit({ type: 'agent_start' });
    session.emit({ type: 'turn_start' });
    // openLlmSpan names the span from message_start's `model` field.
    session.emit({
      type: 'message_start',
      message: assistantMessage({ model: 'claude-sonnet-5-preview', provider: 'anthropic' }),
    });
    // closeLlmSpan renames it via span.updateName() using message_end's
    // `responseModel`, which the provider may resolve to something more
    // specific than the requested alias.
    session.emit({
      type: 'message_end',
      message: assistantMessage({
        model: 'claude-sonnet-5-preview',
        responseModel: 'claude-sonnet-5-20260315',
        provider: 'anthropic',
      }),
    });
    session.emit({ type: 'turn_end', message: assistantMessage(), toolResults: [] });
    session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
    await done;

    const llmSpan = capture.spans.find((s) => attrs(s)['openinference.span.kind'] === 'LLM');
    assert.ok(llmSpan);

    // The exporter only ever sees the span after it ends (SimpleSpanProcessor
    // exports onEnd), so the FINAL name post-updateName is what must show up —
    // never the transient initial name set at startSpan() time.
    assert.equal(
      llmSpan!.name,
      'claude-sonnet-5-20260315',
      'the exported span name must be the final (post-updateName) response model',
    );
    assert.notEqual(
      llmSpan!.name,
      'claude-sonnet-5-preview',
      'the exported span name must not be the initial (pre-updateName) request model',
    );

    // The request-model attribute is a separate concern from the span name
    // and must still independently reflect message_start's data, proving the
    // rename doesn't clobber the earlier-set attribute.
    assert.equal(attrs(llmSpan!)['gen_ai.request.model'], 'claude-sonnet-5-preview');
    assert.equal(attrs(llmSpan!)['gen_ai.response.model'], 'claude-sonnet-5-20260315');
  });

  it('a stray message_start with no rootCtx never parents under whatever span is ambiently active in the host process', async () => {
    const { capture, Session } = makeRig();
    const session = new Session();

    // Flipped from calling session.prompt() here: under the new prompt()-
    // anchored root model, EVERY prompt() call opens a root immediately at
    // entry (see instrumentation.ts's module header), so it can no longer
    // stand in for "no root context at all". steer() attaches the same span
    // listener (via ensureSubscribed) WITHOUT ever opening a root — exactly
    // the rootless-bypass boundary policy this test means to probe.
    await session.steer('a stray assistant message with no prompt() ever called');

    const manager = new StackContextManager();
    context.setGlobalContextManager(manager);
    try {
      const ambientCtx = trace.setSpanContext(context.active(), AMBIENT_SPAN_CONTEXT);
      context.with(ambientCtx, () => {
        session.emit({ type: 'message_start', message: assistantMessage() });
        session.emit({ type: 'message_end', message: assistantMessage() });
      });
    } finally {
      context.disable();
    }

    const llmSpan = capture.spans.find((s) => attrs(s)['openinference.span.kind'] === 'LLM');
    assert.ok(llmSpan, 'the stray message still produces an LLM span');
    assert.equal(
      llmSpan!.parentSpanId,
      undefined,
      'with no rootCtx, the LLM span must start a fresh standalone trace (ROOT_CONTEXT), not ' +
        'silently attach to whatever span the host process happens to have ambiently active',
    );
    assert.notEqual(
      llmSpan!.spanContext().traceId,
      AMBIENT_SPAN_CONTEXT.traceId,
      'the stray LLM span must not join the ambient hosts trace',
    );
  });

  it('a stray tool_execution_start with no llmCtx/rootCtx never parents under whatever span is ambiently active in the host process', async () => {
    const { capture, Session } = makeRig();
    const session = new Session();

    // See the previous test's comment: steer(), not prompt(), is now the way
    // to attach the listener without opening a root.
    await session.steer('a stray tool call with no prompt() ever called');

    const manager = new StackContextManager();
    context.setGlobalContextManager(manager);
    try {
      const ambientCtx = trace.setSpanContext(context.active(), AMBIENT_SPAN_CONTEXT);
      context.with(ambientCtx, () => {
        session.emit({
          type: 'tool_execution_start',
          toolCallId: 'stray',
          toolName: 'bash',
          args: { command: 'echo stray' },
        });
        session.emit({
          type: 'tool_execution_end',
          toolCallId: 'stray',
          toolName: 'bash',
          result: {},
          isError: false,
        });
      });
    } finally {
      context.disable();
    }

    const toolSpan = capture.spans.find((s) => attrs(s)['gen_ai.tool.call.id'] === 'stray');
    assert.ok(toolSpan, 'the stray tool call still produces a tool span');
    assert.equal(
      toolSpan!.parentSpanId,
      undefined,
      'with no llmCtx/rootCtx, the tool span must start a fresh standalone trace, not silently ' +
        'attach to whatever span the host process happens to have ambiently active',
    );
    assert.notEqual(
      toolSpan!.spanContext().traceId,
      AMBIENT_SPAN_CONTEXT.traceId,
      'the stray tool span must not join the ambient hosts trace',
    );
  });
});

describe('close-event idempotency and content safety', () => {
  /**
   * Probes scenarios for src/instrumentation.ts not covered elsewhere:
   *  - duplicate/late CLOSE events (tool_execution_end twice, message_end twice,
   *    a late tool_execution_end after a force-close sweep) — the rest of the
   *    suite covers duplicate OPEN events heavily but not duplicate/stale CLOSE
   *    ones;
   *  - 3+ concurrent tool calls with out-of-order ends (only 2 concurrent tested
   *    elsewhere);
   *  - cross-session isolation of the retry-attempt count;
   *  - agent_end firing twice for one attempt (idempotent output stamping,
   *    since agent_end no longer owns closing the root — see
   *    instrumentation.ts's module header on the prompt()-anchored model);
   *  - malformed message content (not an array) reaching stampRootOutput /
   *    closeLlmSpan while captureContent is on — regression coverage for a bug
   *    where the content-extraction step could throw before endSpanSafe() ran,
   *    silently dropping the span instead of exporting it.
   */
  it('tool_execution_end firing twice for the same toolCallId exports exactly one tool span and never crashes on the second (stale) close', async () => {
    const { capture, Session } = makeRig();
    const session = new Session();

    const done = session.prompt('a buggy tool runner emits two end events for one call id');
    session.emit({ type: 'agent_start' });
    session.emit({ type: 'message_start', message: assistantMessage() });
    session.emit({ type: 'message_end', message: assistantMessage() });
    session.emit({
      type: 'tool_execution_start',
      toolCallId: 't1',
      toolName: 'bash',
      args: { command: 'echo hi' },
    });
    session.emit({
      type: 'tool_execution_end',
      toolCallId: 't1',
      toolName: 'bash',
      result: { ok: true },
      isError: false,
    });
    // Second end for the same id — the Map entry is already gone, so this must
    // take the `if (span)` no-op path, never double-close or throw.
    assert.doesNotThrow(() => {
      session.emit({
        type: 'tool_execution_end',
        toolCallId: 't1',
        toolName: 'bash',
        result: { ok: true },
        isError: false,
      });
    });
    session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
    await done;

    const toolSpans = capture.spans.filter((s) => attrs(s)['gen_ai.tool.call.id'] === 't1');
    assert.equal(toolSpans.length, 1, 'exactly one tool span despite two end events');
    assert.equal(
      attrs(toolSpans[0]!)['traceroot.pi.force_closed'],
      undefined,
      'the tool span closed normally via the first end — it must not be marked force_closed',
    );
  });

  it('message_end (assistant) firing twice exports exactly one LLM span and the second (stale) close is a no-op', async () => {
    const { capture, Session } = makeRig();
    const session = new Session();

    const done = session.prompt('a duplicated message_end arrives for one assistant turn');
    session.emit({ type: 'agent_start' });
    session.emit({ type: 'message_start', message: assistantMessage({ model: 'm1' }) });
    session.emit({ type: 'message_end', message: assistantMessage({ model: 'm1' }) });
    assert.doesNotThrow(() => {
      session.emit({ type: 'message_end', message: assistantMessage({ model: 'm1' }) });
    });
    session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
    await done;

    const llmSpans = capture.spans.filter((s) => attrs(s)['openinference.span.kind'] === 'LLM');
    assert.equal(
      llmSpans.length,
      1,
      'one LLM span — the second message_end must not fabricate another',
    );
  });

  it('a late tool_execution_end arriving after agent_end already force-closed the dangling tool span is a harmless no-op (no double export, no crash)', async () => {
    const { capture, Session } = makeRig();
    const session = new Session();

    const done = session.prompt('a tool never closes before agent_end, then its end arrives late');
    session.emit({ type: 'agent_start' });
    session.emit({ type: 'message_start', message: assistantMessage() });
    session.emit({ type: 'message_end', message: assistantMessage() });
    session.emit({
      type: 'tool_execution_start',
      toolCallId: 'slow',
      toolName: 'bash',
      args: { command: 'sleep 999' },
    });
    // agent_end force-closes the dangling tool span (marks it force_closed).
    session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
    await done;
    // The real tool_execution_end lands late, after the sweep already cleared
    // state.toolSpans — must be a no-op, not a second export or a crash.
    assert.doesNotThrow(() => {
      session.emit({
        type: 'tool_execution_end',
        toolCallId: 'slow',
        toolName: 'bash',
        result: { done: true },
        isError: false,
      });
    });

    const toolSpans = capture.spans.filter((s) => attrs(s)['gen_ai.tool.call.id'] === 'slow');
    assert.equal(toolSpans.length, 1, 'the tool span was exported once (force-closed), not twice');
    assert.equal(attrs(toolSpans[0]!)['traceroot.pi.force_closed'], true);
  });

  it('three concurrent tool calls in one turn with out-of-order ends each get their own correctly-keyed span, all parented under the LLM span', async () => {
    const { capture, Session } = makeRig();
    const session = new Session();

    const done = session.prompt('three tools run concurrently and finish out of order');
    session.emit({ type: 'agent_start' });
    session.emit({ type: 'message_start', message: assistantMessage() });
    session.emit({ type: 'message_end', message: assistantMessage() });
    session.emit({
      type: 'tool_execution_start',
      toolCallId: 'a',
      toolName: 'bash',
      args: { i: 1 },
    });
    session.emit({
      type: 'tool_execution_start',
      toolCallId: 'b',
      toolName: 'read',
      args: { i: 2 },
    });
    session.emit({
      type: 'tool_execution_start',
      toolCallId: 'c',
      toolName: 'write',
      args: { i: 3 },
    });
    // Ends arrive out of registration order: b, then c, then a.
    session.emit({
      type: 'tool_execution_end',
      toolCallId: 'b',
      toolName: 'read',
      result: {},
      isError: false,
    });
    session.emit({
      type: 'tool_execution_end',
      toolCallId: 'c',
      toolName: 'write',
      result: {},
      isError: true,
    });
    session.emit({
      type: 'tool_execution_end',
      toolCallId: 'a',
      toolName: 'bash',
      result: {},
      isError: false,
    });
    session.emit({ type: 'turn_end', message: assistantMessage(), toolResults: [] });
    session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
    await done;

    const toolSpans = capture.spans.filter((s) => attrs(s)['openinference.span.kind'] === 'TOOL');
    assert.equal(toolSpans.length, 3, 'exactly three distinct tool spans');
    const byId = new Map(toolSpans.map((s) => [attrs(s)['gen_ai.tool.call.id'], s]));
    assert.equal(attrs(byId.get('a')!)['gen_ai.tool.name'], 'bash');
    assert.equal(attrs(byId.get('b')!)['gen_ai.tool.name'], 'read');
    assert.equal(attrs(byId.get('c')!)['gen_ai.tool.name'], 'write');
    // None force-closed — each got its own explicit end.
    for (const s of toolSpans) {
      assert.equal(attrs(s)['traceroot.pi.force_closed'], undefined);
    }
    // c was an error result — it must carry ERROR status; a and b must not.
    const llmSpan = capture.spans.find((s) => attrs(s)['openinference.span.kind'] === 'LLM');
    assert.ok(llmSpan);
    for (const s of toolSpans) {
      assert.equal(
        s.parentSpanId,
        llmSpan!.spanContext().spanId,
        'every concurrent tool span parents under the shared LLM span, not under a sibling tool',
      );
    }
  });

  // Rephrased from a pre-fix test of the now-deleted per-session PromptQueue's
  // willRetry reservation (that mechanism no longer exists — see
  // prompt-queue.ts's removal in this change). The underlying concern —
  // per-session state must never leak across sessions — still applies to its
  // replacement, SessionSpanState.retryCount: session A retrying must not
  // bleed its retry_count into an unrelated session B's own root span.
  it('the retry-attempt count is per-session: one session incrementing it must not make another session inherit it', async () => {
    const { capture, Session } = makeRig();
    const sessionA = new Session();
    const sessionB = new Session();
    (sessionA as { sessionId: string }).sessionId = 'A';
    (sessionB as { sessionId: string }).sessionId = 'B';

    // Session A retries once (its own root stays open across the continuation
    // — see instrumentation-edge-cases.test.ts's retry test).
    const doneA = sessionA.prompt('input-A');
    sessionA.emit({ type: 'agent_start' });
    sessionA.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: true });
    sessionA.emit({ type: 'agent_start' });
    sessionA.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
    await doneA;

    // Session B, a completely independent session, runs cleanly with no retry.
    const doneB = sessionB.prompt('input-B');
    sessionB.emit({ type: 'agent_start' });
    sessionB.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
    await doneB;

    const bRoot = capture.spans.find(
      (s) => attrs(s)['openinference.span.kind'] === 'AGENT' && attrs(s)['session.id'] === 'B',
    );
    assert.ok(bRoot);
    assert.equal(
      attrs(bRoot!)['input.value'],
      'input-B',
      "session B's root span must carry its own input, never session A's",
    );
    assert.equal(
      attrs(bRoot!)['traceroot.pi.retry_count'],
      0,
      "session B's retry_count must not inherit session A's retry",
    );
  });

  it('agent_end firing twice for one attempt before prompt() settles stamps output idempotently', async () => {
    // Under the pre-fix (agent_end-anchored) model, agent_end closed the root
    // itself, so a duplicate agent_end risked a double-close. Flipped here:
    // agent_end no longer closes the root at all (see instrumentation.ts's
    // module header) — it only stamps output onto the still-open root, so a
    // duplicate agent_end is now just a harmless repeated stamp. The root
    // still closes exactly once, when the enclosing prompt() call's own
    // promise settles.
    const { capture, Session } = makeRig();
    const session = new Session();

    const done = session.prompt('a duplicate agent_end fires for one run');
    session.emit({ type: 'agent_start' });
    session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
    assert.doesNotThrow(() => {
      session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
    });
    await done;

    const rootSpans = capture.spans.filter((s) => attrs(s)['openinference.span.kind'] === 'AGENT');
    assert.equal(rootSpans.length, 1, 'the root span must be closed and exported exactly once');
    assert.equal(
      attrs(rootSpans[0]!)['traceroot.pi.force_closed'],
      undefined,
      'the root span closed normally when prompt() settled',
    );
  });

  // ---------------------------------------------------------------------------
  // Bug-probing tests: malformed message content reaching the content-extraction
  // step while captureContent is on (the default). See report.
  // ---------------------------------------------------------------------------

  it('agent_end with a malformed final assistant message (content is not an array) does not throw, and the AGENT span still exports once prompt() settles', async () => {
    const { capture, Session } = makeRig();
    const session = new Session();

    const done = session.prompt('agent_end delivers a malformed final message');
    session.emit({ type: 'agent_start' });
    // role passes the `=== 'assistant'` narrowing, but content is not an array,
    // so spans.ts textOf()'s `message.content.filter(...)` throws. spans.ts's
    // stampRootOutput() now wraps that extraction so it can never crash the
    // event handler — and the span isn't even ended here (agent_end no longer
    // owns closing the root), so there is nothing to leak unended either way.
    assert.doesNotThrow(() => {
      session.emit({
        type: 'agent_end',
        // @ts-expect-error intentionally malformed content to probe robustness
        messages: [{ role: 'assistant', content: undefined, stopReason: 'stop', timestamp: 0 }],
        willRetry: false,
      });
    });
    await done;

    const rootSpans = capture.spans.filter((s) => attrs(s)['openinference.span.kind'] === 'AGENT');
    assert.equal(
      rootSpans.length,
      1,
      'the AGENT span must still be ended and exported once prompt() settles, even though output ' +
        'extraction threw while stamping it',
    );
  });

  it('message_end with a malformed assistant message (content is not an array) still ends and exports the LLM span instead of leaking it unended', async () => {
    const { capture, Session } = makeRig();
    const session = new Session();

    const done = session.prompt('message_end delivers a malformed assistant message');
    session.emit({ type: 'agent_start' });
    session.emit({ type: 'message_start', message: assistantMessage({ model: 'the-model' }) });
    // Malformed content on message_end: textOf() throws inside closeLlmSpan, but
    // that extraction is now wrapped in try/catch so endSpanSafe() still runs as
    // part of THIS event — not deferred to a later force-close sweep.
    assert.doesNotThrow(() => {
      session.emit({
        type: 'message_end',
        message: {
          role: 'assistant',
          // @ts-expect-error intentionally malformed content to probe robustness
          content: undefined,
          model: 'the-model',
          stopReason: 'stop',
          timestamp: 0,
        },
      });
    });
    // message_end is the LLM span's normal close, so it must not be force_closed
    // by this later turn_end — proving message_end itself already ended it above.
    session.emit({ type: 'turn_end', message: assistantMessage(), toolResults: [] });
    session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
    await done;

    const llmSpans = capture.spans.filter((s) => attrs(s)['openinference.span.kind'] === 'LLM');
    assert.equal(llmSpans.length, 1, 'the LLM span must still be exported');
    assert.equal(
      attrs(llmSpans[0]!)['traceroot.pi.force_closed'],
      undefined,
      "message_end is the LLM span's normal close — a malformed content payload must not " +
        'demote it to a force-closed span (which means span.end() was skipped in closeLlmSpan)',
    );
  });

  it('agent_end with an empty messages array (valid) does not throw, and the AGENT span exports with no output.value once prompt() settles', async () => {
    const { capture, Session } = makeRig();
    const session = new Session();

    const done = session.prompt('agent_end with no messages at all');
    session.emit({ type: 'agent_start' });
    assert.doesNotThrow(() => {
      session.emit({ type: 'agent_end', messages: [], willRetry: false });
    });
    await done;

    const rootSpans = capture.spans.filter((s) => attrs(s)['openinference.span.kind'] === 'AGENT');
    assert.equal(rootSpans.length, 1);
    assert.equal(
      attrs(rootSpans[0]!)['output.value'],
      undefined,
      'no assistant message -> no output',
    );
    assert.equal(attrs(rootSpans[0]!)['traceroot.pi.force_closed'], undefined, 'closed normally');
  });
});

describe('dangling-span sweep deduplication', () => {
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
  it('agent_start sweeps a dangling LLM + TOOL span from a crashed prior ATTEMPT, but leaves the still-open root untouched', async () => {
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

  it('turn_end force-closes a dangling LLM + TOOL span but leaves the root span open', async () => {
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

  // Originally confirmed-bugfix-regressions.test.ts's Bug 7: kept distinct from
  // the turn_end test above because it asserts a different invariant — that
  // turn_end must also CLEAR the tool span from state.toolSpans, so agent_end's
  // own defensive sweep does not try to force-close (and double-export) the
  // same span a second time.
  it('turn_end force-closes any tool spans still open at the end of the turn, and clears them so agent_end does not force-close them a second time', async () => {
    const { capture, Session } = makeRig();
    const session = new Session();

    const done = session.prompt(
      'a tool call never gets its tool_execution_end before the turn ends',
    );
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
    const toolSpans = capture.spans.filter(
      (s) => attrs(s)['gen_ai.tool.call.id'] === 'never-closes',
    );
    assert.equal(
      toolSpans.length,
      1,
      'the tool span must be exported exactly once — turn_end must also clear it from ' +
        'state.toolSpans so agent_end does not try to force-close it a second time',
    );
  });

  it('agent_end sweeps a dangling LLM + TOOL span while stamping (not force-closing, not even ending) the still-open root', async () => {
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

  it('dispose() mid-run force-closes the root + LLM + TOOL spans', async () => {
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
      'dispose() must force-close the root, LLM, and TOOL spans (found ' +
        `${capture.spans.length})`,
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
  it('a second prompt() call while the first window’s root is still open force-closes the first root (and its dangling LLM/TOOL), then opens a fresh root for the second', async () => {
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
    const overlapLlm = capture.spans.find(
      (s) => attrs(s)['gen_ai.request.model'] === 'overlap-llm',
    );
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
  it('a mid-stream steer (isStreaming===true, streamingBehavior set) never opens a fresh root or sweeps the active run’s still-open root — the active trace stays intact', async () => {
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
  it('a prompt() call that REJECTS mid-run force-closes a still-open tool span before finalizing the root as ERROR (claude-agent-sdk.ts endInFlight parity)', async () => {
    const { capture, Session } = makeRig();
    const session = new Session();

    const done = session.prompt(
      'a run whose internal loop rejects while a tool call is still open',
    );
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
});
