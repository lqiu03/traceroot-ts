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
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { assistantMessage, attrs, makeRig } from './pi-test-helpers';

test('tool_execution_end firing twice for the same toolCallId exports exactly one tool span and never crashes on the second (stale) close', async () => {
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

test('message_end (assistant) firing twice exports exactly one LLM span and the second (stale) close is a no-op', async () => {
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

test('a late tool_execution_end arriving after agent_end already force-closed the dangling tool span is a harmless no-op (no double export, no crash)', async () => {
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

test('three concurrent tool calls in one turn with out-of-order ends each get their own correctly-keyed span, all parented under the LLM span', async () => {
  const { capture, Session } = makeRig();
  const session = new Session();

  const done = session.prompt('three tools run concurrently and finish out of order');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'message_start', message: assistantMessage() });
  session.emit({ type: 'message_end', message: assistantMessage() });
  session.emit({ type: 'tool_execution_start', toolCallId: 'a', toolName: 'bash', args: { i: 1 } });
  session.emit({ type: 'tool_execution_start', toolCallId: 'b', toolName: 'read', args: { i: 2 } });
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
test('the retry-attempt count is per-session: one session incrementing it must not make another session inherit it', async () => {
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

test('agent_end firing twice for one attempt before prompt() settles stamps output idempotently and does not emit a spurious reentrant-dispose warning', async () => {
  // Under the pre-fix (agent_end-anchored) model, agent_end closed the root
  // itself, so a duplicate agent_end risked a double-close. Flipped here:
  // agent_end no longer closes the root at all (see instrumentation.ts's
  // module header) — it only stamps output onto the still-open root, so a
  // duplicate agent_end is now just a harmless repeated stamp. The root
  // still closes exactly once, when the enclosing prompt() call's own
  // promise settles.
  const { capture, Session } = makeRig();
  const session = new Session();

  const warnings: unknown[][] = [];
  const originalWarn = console.warn;
  console.warn = (...args: unknown[]) => {
    warnings.push(args);
  };
  try {
    const done = session.prompt('a duplicate agent_end fires for one run');
    session.emit({ type: 'agent_start' });
    session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
    assert.doesNotThrow(() => {
      session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
    });
    await done;
  } finally {
    console.warn = originalWarn;
  }

  const rootSpans = capture.spans.filter((s) => attrs(s)['openinference.span.kind'] === 'AGENT');
  assert.equal(rootSpans.length, 1, 'the root span must be closed and exported exactly once');
  assert.equal(
    attrs(rootSpans[0]!)['traceroot.pi.force_closed'],
    undefined,
    'the root span closed normally when prompt() settled',
  );
  assert.ok(
    !warnings.some((args) => typeof args[0] === 'string' && args[0].includes('reentrant dispose')),
    'a second agent_end (root span still open, never force-closed) must not be mistaken for a ' +
      'reentrant-dispose force-close',
  );
});

// ---------------------------------------------------------------------------
// Bug-probing tests: malformed message content reaching the content-extraction
// step while captureContent is on (the default). See report.
// ---------------------------------------------------------------------------

test('agent_end with a malformed final assistant message (content is not an array) does not throw, and the AGENT span still exports once prompt() settles', async () => {
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

test('message_end with a malformed assistant message (content is not an array) still ends and exports the LLM span instead of leaking it unended', async () => {
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

test('agent_end with an empty messages array (valid) does not throw, and the AGENT span exports with no output.value once prompt() settles', async () => {
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
