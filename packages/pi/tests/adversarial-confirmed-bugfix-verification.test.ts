/**
 * Lens: confirmed-bugfix-verification.
 *
 * Each test below reproduces the EXACT failure scenario for one of eight
 * independently-verified bugs found by two separate review passes against
 * src/instrumentation.ts (and, for the forceFlush test, src/provider.ts),
 * and asserts the fixed behavior. Written after the fixes landed, but each
 * test is structured to fail against the pre-fix code (verified by reading
 * the git diff), not just to pass against the current implementation.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { context, trace, ROOT_CONTEXT, TraceFlags } from '@opentelemetry/api';
import type { Context, ContextManager, SpanContext } from '@opentelemetry/api';
import type { ExportResult } from '@opentelemetry/core';
import { ExportResultCode, hrTime, hrTimeToMilliseconds } from '@opentelemetry/core';
import type { ReadableSpan, SpanExporter } from '@opentelemetry/sdk-trace-base';
import { instrumentPiCodingAgent } from '../src/instrumentation';
import type { AgentEvent, AssistantMessage } from '../src/types';

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
function makeFakeSessionClass() {
  return class FakeAgentSession {
    sessionId = 'sess-1';
    private listeners: Array<(event: AgentEvent) => void> = [];
    async prompt(_text: string, _options?: unknown): Promise<void> {}
    subscribe(listener: (event: AgentEvent) => void): () => void {
      this.listeners.push(listener);
      return () => {
        this.listeners = this.listeners.filter((l) => l !== listener);
      };
    }
    emit(event: AgentEvent): void {
      for (const listener of this.listeners) listener(event);
    }
  };
}

function makeRig() {
  const capture = new CapturingExporter();
  const Session = makeFakeSessionClass();
  const sdk = { AgentSession: Session };
  instrumentPiCodingAgent(sdk, { apiKey: 'test-key', _spanExporter: capture });
  return { capture, Session };
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

// Bug 1 -------------------------------------------------------------------

test('beforeExit hook calls forceFlush(), not shutdown() — a later run still exports after it fires', async () => {
  const capture = new CapturingExporter();
  const Session = makeFakeSessionClass();
  const sdk = { AgentSession: Session };

  const originalOnce = process.once.bind(process);
  let beforeExitListener: (() => void) | undefined;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (process as any).once = (event: string, listener: (...a: unknown[]) => void) => {
    if (event === 'beforeExit') beforeExitListener = listener as () => void;
    return originalOnce(event as never, listener as never);
  };
  try {
    instrumentPiCodingAgent(sdk, { apiKey: 'k', _spanExporter: capture });
  } finally {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (process as any).once = originalOnce;
  }

  assert.ok(beforeExitListener, 'instrumentPiCodingAgent() must register a beforeExit hook');

  const session = new Session();
  await session.prompt('first run');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
  assert.equal(capture.spans.length, 1, 'run 1 exports normally');

  // Simulate the event loop draining once, as it would in a real long-lived
  // host process after the first prompt's work settles.
  assert.doesNotThrow(() => beforeExitListener!());
  await new Promise((resolve) => setTimeout(resolve, 10));

  // Run 2, well after the beforeExit hook already fired. If the hook had
  // called shutdown() (the bug) instead of forceFlush(), the pipeline would
  // be permanently disabled and this span would never export — exactly the
  // "first drain of the event loop kills all future export" failure mode.
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });

  assert.equal(
    capture.spans.length,
    2,
    'a second run after beforeExit fired must still export — forceFlush() must not permanently ' +
      'disable the pipeline the way shutdown() would',
  );
});

// Bug 2 -------------------------------------------------------------------

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

// Bug 3 -------------------------------------------------------------------

test('two overlapping prompt() calls before their respective agent_start events each get their own correct input text, FIFO', async () => {
  const { capture, Session } = makeRig();
  const session = new Session();

  // Both prompt() calls fire before EITHER agent_start arrives — e.g. the
  // host app queues a second message while the first is still being
  // dispatched into the agent loop. With the old single-slot WeakMap, the
  // second prompt() call would silently overwrite the first's pending text.
  await session.prompt('first prompt text');
  await session.prompt('second prompt text');

  session.emit({ type: 'agent_start' });
  session.emit({
    type: 'agent_end',
    messages: [assistantMessage({ content: [{ type: 'text', text: 'first reply' }] })],
    willRetry: false,
  });

  session.emit({ type: 'agent_start' });
  session.emit({
    type: 'agent_end',
    messages: [assistantMessage({ content: [{ type: 'text', text: 'second reply' }] })],
    willRetry: false,
  });

  const rootSpans = capture.spans.filter((s) => attrs(s)['openinference.span.kind'] === 'AGENT');
  assert.equal(rootSpans.length, 2);
  assert.equal(
    attrs(rootSpans[0]!)['input.value'],
    'first prompt text',
    'the FIRST agent_start must claim the FIRST queued prompt() text, not the second',
  );
  assert.equal(attrs(rootSpans[0]!)['output.value'], 'first reply');
  assert.equal(
    attrs(rootSpans[1]!)['input.value'],
    'second prompt text',
    'the SECOND agent_start must claim the SECOND queued prompt() text — with the old single-slot ' +
      'design both roots would incorrectly show the same (second) text',
  );
  assert.equal(attrs(rootSpans[1]!)['output.value'], 'second reply');
});

// Bug 4 -------------------------------------------------------------------

test('a retried run (willRetry: true) reuses the SAME input text on its retry agent_start, which fires with no new prompt() call in between', async () => {
  const { capture, Session } = makeRig();
  const session = new Session();

  await session.prompt('flaky prompt that needs a retry');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: true });
  // The agent loop retries: a fresh agent_start fires with NO new prompt()
  // call in between (the host app never re-sent the text).
  session.emit({ type: 'agent_start' });
  session.emit({
    type: 'agent_end',
    messages: [assistantMessage({ content: [{ type: 'text', text: 'succeeded on retry' }] })],
    willRetry: false,
  });

  const rootSpans = capture.spans.filter((s) => attrs(s)['openinference.span.kind'] === 'AGENT');
  assert.equal(rootSpans.length, 2, 'each attempt gets its own root span');
  assert.equal(
    attrs(rootSpans[0]!)['input.value'],
    'flaky prompt that needs a retry',
    'the first attempt must carry the original prompt text',
  );
  assert.equal(
    attrs(rootSpans[1]!)['input.value'],
    'flaky prompt that needs a retry',
    'the RETRY attempt must still carry the SAME input text — it must survive even though ' +
      'agent_end used to unconditionally delete the pendingInput entry regardless of willRetry',
  );
  assert.equal(attrs(rootSpans[1]!)['output.value'], 'succeeded on retry');

  // A genuine follow-up prompt() call after the retry succeeded must get its
  // own fresh text — proving the retry re-queue doesn't leak forward and
  // make a later, unrelated prompt() replay the old retried text again.
  await session.prompt('a brand new followup prompt');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
  const allRootSpans = capture.spans.filter((s) => attrs(s)['openinference.span.kind'] === 'AGENT');
  assert.equal(allRootSpans.length, 3);
  assert.equal(
    attrs(allRootSpans[2]!)['input.value'],
    'a brand new followup prompt',
    'after a successful retry, a genuine new prompt() call must not accidentally replay the old ' +
      'retried text a third time',
  );
});

// Bug 5 -------------------------------------------------------------------

test('a stray message_start with no rootCtx never parents under whatever span is ambiently active in the host process', async () => {
  const { capture, Session } = makeRig();
  const session = new Session();

  await session.prompt('a stray assistant message with no agent_start ever fired');

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

test('a stray tool_execution_start with no llmCtx/rootCtx never parents under whatever span is ambiently active in the host process', async () => {
  const { capture, Session } = makeRig();
  const session = new Session();

  await session.prompt('a stray tool call with no agent_start ever fired');

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

// Bug 6 -------------------------------------------------------------------

test('agent_start force-closes an orphaned tool span left over from a stray event even when rootSpan was already undefined, instead of leaving it open through the entire next run', async () => {
  const { capture, Session } = makeRig();
  const session = new Session();

  await session.prompt('run 1 finishes cleanly');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });

  // Stray tool_execution_start after agent_end — rootSpan is already
  // undefined here, so the old "if (state.rootSpan)" gate would skip
  // sweeping this orphan at the NEXT agent_start entirely.
  session.emit({
    type: 'tool_execution_start',
    toolCallId: 'orphan',
    toolName: 'bash',
    args: { command: 'echo orphan' },
  });

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
  assert.ok(
    orphanClosedBeforeGapMs >= 0,
    'the orphaned tool span must be force-closed at run 2s agent_start (before the 30ms gap), ' +
      `not left dangling open through the entirety of run 2 (closed ${-orphanClosedBeforeGapMs}ms ` +
      'after the gap started, which only happens if it waited for run 2s agent_end instead)',
  );
});

// Bug 7 -------------------------------------------------------------------

test('turn_end force-closes any tool spans still open at the end of the turn instead of leaving them dangling until agent_end', async () => {
  const { capture, Session } = makeRig();
  const session = new Session();

  await session.prompt('a tool call never gets its tool_execution_end before the turn ends');
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

  instrumentPiCodingAgent(sdk, { apiKey: 'k' });

  const warnings: unknown[][] = [];
  const originalWarn = console.warn;
  console.warn = (...args: unknown[]) => {
    warnings.push(args);
  };
  try {
    instrumentPiCodingAgent(sdk, { apiKey: 'a-second-config-that-gets-ignored' });
  } finally {
    console.warn = originalWarn;
  }

  assert.ok(
    warnings.some((args) => typeof args[0] === 'string' && /already (been )?called/.test(args[0])),
    'the second call must console.warn that instrumentation was already set up and this config ' +
      "is being ignored, matching the function's other two early-return paths",
  );
});
