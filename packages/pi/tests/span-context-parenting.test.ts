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
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { context, trace, ROOT_CONTEXT, TraceFlags } from '@opentelemetry/api';
import type { Context, ContextManager, SpanContext } from '@opentelemetry/api';
import type { ExportResult } from '@opentelemetry/core';
import { ExportResultCode } from '@opentelemetry/core';
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
function makeRig() {
  const capture = new CapturingExporter();

  class FakeAgentSession {
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
  }

  const sdk = { AgentSession: FakeAgentSession };
  // Always private mode (rig-local apiKey/_spanExporter, no shared global
  // provider), so every call registers its own never-auto-removed
  // process.once('beforeExit', ...) hook (see instrumentation.ts and
  // test-helpers.ts's makeRig(), which documents this in full). No test here
  // depends on the beforeExit hook actually firing, so it's safe to strip
  // whatever was just added rather than let it accumulate for the rest of
  // this file's 6 tests (and beyond, for anyone appending more).
  const beforeExitListenersBeforeSetup = new Set(process.listeners('beforeExit'));
  instrumentPiCodingAgent(sdk, { apiKey: 'test-key', _spanExporter: capture });
  for (const listener of process.listeners('beforeExit')) {
    if (!beforeExitListenersBeforeSetup.has(listener)) {
      process.removeListener('beforeExit', listener);
    }
  }

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

test('two back-to-back turns in one agent run each get their own LLM span parented under the shared root, not under each other', async () => {
  const { capture, Session } = makeRig();
  const session = new Session();

  await session.prompt('do two turns');
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

test('a tool span opened during turn 1 keeps its parent bound to turn 1s LLM span (OTel Context is captured immutably at open time) even after turn 2 has already opened its own LLM span before turn 1s tool call closes', async () => {
  const { capture, Session } = makeRig();
  const session = new Session();

  await session.prompt('turn 1s tool call resolves late, after turn 2 already started');
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

test('tool_execution_start firing after turn_end cleared llmCtx but before the next turns message_start falls back to the root span, not a stale ended LLM context', async () => {
  const { capture, Session } = makeRig();
  const session = new Session();

  await session.prompt('a tool fires in the gap between two turns');
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

test('closeLlmSpan span.updateName() changes the name the exporter actually captures — the final response model wins over the initial request model', async () => {
  const { capture, Session } = makeRig();
  const session = new Session();

  await session.prompt('the provider renames the model between request and response');
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
