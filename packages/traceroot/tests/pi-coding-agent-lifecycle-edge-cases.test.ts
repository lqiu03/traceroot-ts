import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { SpanStatusCode, context, propagation, trace } from '@opentelemetry/api';
import { InMemorySpanExporter, SimpleSpanProcessor } from '@opentelemetry/sdk-trace-base';
import type { ReadableSpan } from '@opentelemetry/sdk-trace-base';
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import {
  instrumentPiCodingAgent,
  stampRootOutput,
  type AgentEvent,
  type AgentMessage,
  type AssistantMessage,
  type UserMessage,
} from '../src/pi';
import {
  CapturingExporter,
  assistantMessage,
  attrs,
  makeFakeSessionClass,
} from './pi-test-helpers';

describe('instrumentation edge cases', () => {
  // Local rig, since several tests assert on direct prompt identity / subscribeCallCount.
  function registerCapturingProvider(capture: CapturingExporter): void {
    trace.disable();
    const provider = new NodeTracerProvider();
    provider.addSpanProcessor(new SimpleSpanProcessor(capture));
    provider.register();
  }

  // Local fixture (with subscribeCallCount) rather than the shared one, to keep prototype-identity assertions self-contained.
  function makeFakeSessionClass() {
    return class FakeAgentSession {
      sessionId = 'sess-1';
      private listeners: Array<(event: AgentEvent) => void> = [];
      subscribeCallCount = 0;
      private pending: { resolve: () => void; reject: (err: unknown) => void } | undefined;
      async prompt(_text: string, _options?: unknown): Promise<void> {
        return new Promise<void>((resolve, reject) => {
          this.pending = { resolve, reject };
        });
      }
      subscribe(listener: (event: AgentEvent) => void): () => void {
        this.subscribeCallCount += 1;
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
    };
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

  it('instrumenting the same sdk object twice does not double-wrap prompt', async () => {
    const capture = new CapturingExporter();
    const Session = makeFakeSessionClass();
    const sdk = { AgentSession: Session };

    registerCapturingProvider(capture);
    instrumentPiCodingAgent(sdk, {});
    const wrappedOnce = Session.prototype.prompt;
    instrumentPiCodingAgent(sdk, {});
    const wrappedTwice = Session.prototype.prompt;

    assert.equal(
      wrappedOnce,
      wrappedTwice,
      'a second instrumentPiCodingAgent() call must be a no-op',
    );

    const session = new Session();
    const done = session.prompt('hi');
    session.emit({ type: 'agent_start' });
    session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
    await done;

    assert.equal(
      capture.spans.length,
      1,
      'exactly one root span, not two — proves subscribe() was not registered twice',
    );
  });

  // A real `import * as pi` namespace object is always non-extensible per spec; Object.preventExtensions
  // reproduces that shape — the wrap-once guard must never be stamped directly onto `sdk` itself.
  it('instrumenting a non-extensible sdk object (e.g. a real `import * as pi` ES module namespace) does not throw', async () => {
    const capture = new CapturingExporter();
    const Session = makeFakeSessionClass();
    const sdk = Object.preventExtensions({ AgentSession: Session });

    registerCapturingProvider(capture);
    assert.doesNotThrow(() => {
      instrumentPiCodingAgent(sdk, {});
    });

    const session = new Session();
    const done = session.prompt('hi');
    session.emit({ type: 'agent_start' });
    session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
    await done;

    assert.equal(
      capture.spans.length,
      1,
      'instrumentation still works end-to-end against a non-extensible sdk object',
    );

    const wrappedOnce = Session.prototype.prompt;
    assert.doesNotThrow(() => {
      instrumentPiCodingAgent(sdk, {});
    });
    assert.equal(Session.prototype.prompt, wrappedOnce);
  });

  // Guards silent double-instrumentation across independently-loaded copies sharing one prototype: a
  // module-scoped `Symbol()` guard would fail silently, so this must use the interned Symbol.for(key).
  it('the wrap-once guard key is the globally-interned Symbol.for() value, and a second call warns and is rejected', () => {
    const Session = makeFakeSessionClass();
    const sdk = { AgentSession: Session };

    instrumentPiCodingAgent(sdk, {});

    const guardKeys = Object.getOwnPropertySymbols(Session.prototype).filter(
      (sym) => sym.description === 'traceroot.pi_coding_agent.wrapped',
    );
    assert.equal(guardKeys.length, 1, 'exactly one wrap-once guard key should be stamped');
    assert.equal(
      guardKeys[0],
      Symbol.for('traceroot.pi_coding_agent.wrapped'),
      'the guard key must be the globally-interned Symbol.for() value — not a ' +
        'module-scoped Symbol() — so that two independently-loaded copies of ' +
        "this module sharing one AgentSession.prototype detect each other's " +
        'stamp instead of silently double-wrapping and doubling every span export',
    );

    const originalWarn = console.warn;
    const warnings: unknown[][] = [];
    console.warn = (...args: unknown[]) => {
      warnings.push(args);
    };
    try {
      const wrappedOnce = Session.prototype.prompt;
      instrumentPiCodingAgent(sdk, { captureContent: false });
      assert.equal(
        Session.prototype.prompt,
        wrappedOnce,
        "a second instrumentPiCodingAgent() call with a different config must not re-wrap; it's a no-op",
      );
    } finally {
      console.warn = originalWarn;
    }
    assert.ok(
      warnings.some((args) =>
        args.some((arg) => typeof arg === 'string' && arg.includes('already called for this sdk')),
      ),
      'the second call must log a console.warn explaining its config was dropped',
    );
  });

  it('a module missing AgentSession.prototype.prompt/subscribe degrades to a no-op, never throws', () => {
    const brokenSdk = { AgentSession: { prototype: {} } };
    assert.doesNotThrow(() => {
      const result = instrumentPiCodingAgent(brokenSdk, {});
      assert.equal(result, brokenSdk);
    });

    assert.doesNotThrow(() => {
      instrumentPiCodingAgent(undefined, {});
    });
    assert.doesNotThrow(() => {
      instrumentPiCodingAgent({}, {});
    });
  });

  it('two different session instances on the same instrumented sdk keep fully independent span trees', async () => {
    const capture = new CapturingExporter();
    const Session = makeFakeSessionClass();
    const sdk = { AgentSession: Session };
    registerCapturingProvider(capture);
    instrumentPiCodingAgent(sdk, {});

    const sessionA = new Session();
    const sessionB = new Session();
    (sessionA as { sessionId: string }).sessionId = 'session-a';
    (sessionB as { sessionId: string }).sessionId = 'session-b';

    // Interleaved on purpose: A starts, B starts, A's tool call runs, B's turn ends, A ends.
    const doneA = sessionA.prompt('task A');
    sessionA.emit({ type: 'agent_start' });
    const doneB = sessionB.prompt('task B');
    sessionB.emit({ type: 'agent_start' });
    sessionA.emit({ type: 'message_start', message: assistantMessage() });
    sessionA.emit({ type: 'message_end', message: assistantMessage() });
    sessionA.emit({
      type: 'tool_execution_start',
      toolCallId: 'a-tool',
      toolName: 'bash',
      args: {},
    });
    sessionA.emit({
      type: 'tool_execution_end',
      toolCallId: 'a-tool',
      toolName: 'bash',
      result: {},
      isError: false,
    });
    sessionB.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
    sessionA.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
    await Promise.all([doneA, doneB]);

    assert.equal(capture.spans.length, 4, 'A: root+LLM+tool (3), B: root (1)');

    const bRoot = capture.spans.find((s) => attrs(s)['session.id'] === 'session-b');
    const aRoot = capture.spans.find((s) => attrs(s)['session.id'] === 'session-a');
    assert.ok(bRoot);
    assert.ok(aRoot);
    // Only root spans carry session.id and only tool spans carry tool.call.id, so "everything that
    // isn't B's root" correctly counts A's 3 spans (the LLM span carries neither).
    const aRelated = capture.spans.filter((s) => s !== bRoot);
    assert.equal(aRelated.length, 3, 'A: root + LLM + tool span');
    assert.notEqual(bRoot!.spanContext().traceId, aRoot!.spanContext().traceId);
  });

  it('tool_execution_end for an unknown toolCallId (no matching start) is ignored, not a crash', async () => {
    const capture = new CapturingExporter();
    const Session = makeFakeSessionClass();
    const sdk = { AgentSession: Session };
    registerCapturingProvider(capture);
    instrumentPiCodingAgent(sdk, {});
    const session = new Session();

    const done = session.prompt('hi');
    session.emit({ type: 'agent_start' });
    assert.doesNotThrow(() => {
      session.emit({
        type: 'tool_execution_end',
        toolCallId: 'never-started',
        toolName: 'bash',
        result: {},
        isError: false,
      });
    });
    session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
    await done;

    const toolSpans = capture.spans.filter((s) => attrs(s)['gen_ai.tool.name']);
    assert.equal(toolSpans.length, 0);
  });

  it('agent_end while a tool span is still open force-closes it instead of leaking', async () => {
    const capture = new CapturingExporter();
    const Session = makeFakeSessionClass();
    const sdk = { AgentSession: Session };
    registerCapturingProvider(capture);
    instrumentPiCodingAgent(sdk, {});
    const session = new Session();

    const done = session.prompt('hi');
    session.emit({ type: 'agent_start' });
    session.emit({ type: 'message_start', message: assistantMessage() });
    session.emit({ type: 'message_end', message: assistantMessage() });
    session.emit({
      type: 'tool_execution_start',
      toolCallId: 'dangling',
      toolName: 'bash',
      args: {},
    });
    session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
    await done;

    assert.equal(capture.spans.length, 3, 'root, LLM, and the force-closed dangling tool span');
    const dangling = capture.spans.find((s) => attrs(s)['gen_ai.tool.name'] === 'bash');
    assert.ok(dangling);
    assert.equal(
      attrs(dangling!)['traceroot.pi.force_closed'],
      true,
      'must be marked as abnormally closed, not indistinguishable from a clean tool span',
    );
  });

  it('message_start/message_end for non-assistant roles never opens an LLM span', async () => {
    const capture = new CapturingExporter();
    const Session = makeFakeSessionClass();
    const sdk = { AgentSession: Session };
    registerCapturingProvider(capture);
    instrumentPiCodingAgent(sdk, {});
    const session = new Session();

    const done = session.prompt('hi');
    session.emit({ type: 'agent_start' });
    session.emit({ type: 'message_start', message: { role: 'user', content: 'hi', timestamp: 0 } });
    session.emit({ type: 'message_end', message: { role: 'user', content: 'hi', timestamp: 0 } });
    session.emit({
      type: 'message_start',
      message: {
        role: 'toolResult',
        toolCallId: 'x',
        toolName: 'bash',
        content: [],
        isError: false,
        timestamp: 0,
      },
    });
    session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
    await done;

    assert.equal(
      capture.spans.length,
      1,
      'only the root span — no LLM span for user/toolResult messages',
    );
  });

  it('a handler throw inside span-building is caught and never propagates to session.emit()', async () => {
    const capture = new CapturingExporter();
    const Session = makeFakeSessionClass();
    const sdk = { AgentSession: Session };
    registerCapturingProvider(capture);
    instrumentPiCodingAgent(sdk, {});
    const session = new Session();

    const done = session.prompt('hi');
    session.emit({ type: 'agent_start' });
    // A malformed/missing `usage` field must not crash the listener — attribute setters must tolerate it.
    assert.doesNotThrow(() => {
      session.emit({
        type: 'message_end',
        // @ts-expect-error intentionally malformed to test resilience
        message: { role: 'assistant', content: [], stopReason: 'stop', timestamp: 0 },
      });
    });
    assert.doesNotThrow(() => {
      session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
    });
    await done;
  });

  it('tool args/result containing a circular reference do not crash span creation', async () => {
    const capture = new CapturingExporter();
    const Session = makeFakeSessionClass();
    const sdk = { AgentSession: Session };
    registerCapturingProvider(capture);
    instrumentPiCodingAgent(sdk, {});
    const session = new Session();

    const circular: Record<string, unknown> = {};
    circular.self = circular;

    const done = session.prompt('hi');
    session.emit({ type: 'agent_start' });
    session.emit({ type: 'message_start', message: assistantMessage() });
    session.emit({ type: 'message_end', message: assistantMessage() });
    assert.doesNotThrow(() => {
      session.emit({
        type: 'tool_execution_start',
        toolCallId: 't1',
        toolName: 'bash',
        args: circular,
      });
    });
    assert.doesNotThrow(() => {
      session.emit({
        type: 'tool_execution_end',
        toolCallId: 't1',
        toolName: 'bash',
        result: circular,
        isError: false,
      });
    });
    session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
    await done;

    const toolSpan = capture.spans.find((s) => attrs(s)['gen_ai.tool.call.id'] === 't1');
    assert.ok(toolSpan);
    assert.equal(
      attrs(toolSpan!)['input.value'],
      undefined,
      'circular args must be skipped, not crash or emit garbage',
    );
  });

  // This fake's `async` prompt() turns its throw into a REJECTED PROMISE, exercising the
  // `result.then(onResolve, onReject)` branch — contrast the synchronous-throw test below.
  it('a rejected prompt() (validation failure before agent_start) never creates a dangling root span, and finalizes the root as ERROR', async () => {
    const capture = new CapturingExporter();
    const Session = makeFakeSessionClass();
    Session.prototype.prompt = async function (): Promise<void> {
      throw new Error('no model selected');
    };
    const sdk = { AgentSession: Session };
    registerCapturingProvider(capture);
    instrumentPiCodingAgent(sdk, {});
    const session = new Session();

    await assert.rejects(() => session.prompt('hi'), /no model selected/);
    // agent_start never fires, but the root opened at entry is still finalized as ERROR, not left dangling.
    assert.equal(capture.spans.length, 1);
    const rootSpan = capture.spans[0]!;
    assert.equal(attrs(rootSpan)['openinference.span.kind'], 'AGENT');
    assert.equal(rootSpan.status.code, 2 /* SpanStatusCode.ERROR */);
    assert.equal(
      rootSpan.events.some((e) => e.name === 'exception'),
      true,
    );
  });

  // This fake's prompt() is plain (non-async) and throws before returning a promise, exercising the
  // synchronous catch branch: the wrapper must finalize the root as ERROR and rethrow synchronously.
  it('a prompt() that throws SYNCHRONOUSLY (before ever returning a promise) still finalizes the root as ERROR and rethrows synchronously, not as a rejected promise', () => {
    const capture = new CapturingExporter();
    const Session = makeFakeSessionClass();
    Session.prototype.prompt = function (): never {
      throw new Error('synchronous validation failure');
    };
    const sdk = { AgentSession: Session };
    registerCapturingProvider(capture);
    instrumentPiCodingAgent(sdk, {});
    const session = new Session();

    assert.throws(() => session.prompt('hi'), /synchronous validation failure/);

    assert.equal(capture.spans.length, 1);
    const rootSpan = capture.spans[0]!;
    assert.equal(attrs(rootSpan)['openinference.span.kind'], 'AGENT');
    assert.equal(rootSpan.status.code, 2 /* SpanStatusCode.ERROR */);
    assert.equal(
      rootSpan.events.some((e) => e.name === 'exception'),
      true,
      'the synchronously-thrown error must still be recorded as an exception on the root span',
    );
  });

  it('a prompt() call whose run retries once (willRetry: true) keeps ONE root span open across the retry continuation, closing it exactly once with retry_count stamped, and both attempts’ LLM spans (ERROR then OK) parent under that single root', async () => {
    const capture = new CapturingExporter();
    const Session = makeFakeSessionClass();
    const sdk = { AgentSession: Session };
    registerCapturingProvider(capture);
    instrumentPiCodingAgent(sdk, {});
    const session = new Session();

    // The retry continuation's agent_start/agent_end fire with no new prompt() call, so both attempts
    // (and each one's own child LLM span) must land under the same still-open root.
    const done = session.prompt('hi');
    session.emit({ type: 'agent_start' });
    session.emit({
      type: 'message_start',
      message: assistantMessage({ model: 'attempt-1-model' }),
    });
    session.emit({
      type: 'message_end',
      message: assistantMessage({
        model: 'attempt-1-model',
        stopReason: 'error',
        errorMessage: 'rate limited',
      }),
    });
    session.emit({ type: 'turn_end', message: assistantMessage(), toolResults: [] });
    session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: true });
    // Retry re-enters the loop; no new prompt() call.
    session.emit({ type: 'agent_start' });
    session.emit({
      type: 'message_start',
      message: assistantMessage({ model: 'attempt-2-model' }),
    });
    session.emit({ type: 'message_end', message: assistantMessage({ model: 'attempt-2-model' }) });
    session.emit({ type: 'turn_end', message: assistantMessage(), toolResults: [] });
    session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
    await done;

    const rootSpans = capture.spans.filter((s) => attrs(s)['openinference.span.kind'] === 'AGENT');
    assert.equal(
      rootSpans.length,
      1,
      'the retry continuation shares ONE root span with its first attempt, not two',
    );
    const root = rootSpans[0]!;
    assert.equal(attrs(root)['traceroot.pi.retry_count'], 1);
    assert.equal(attrs(root)['traceroot.pi.will_retry'], undefined, 'the flag is gone');

    const llmSpans = capture.spans.filter((s) => attrs(s)['openinference.span.kind'] === 'LLM');
    assert.equal(llmSpans.length, 2, 'each attempt gets its own child LLM span, not a merged one');
    const errorLlm = llmSpans.find((s) => attrs(s)['gen_ai.request.model'] === 'attempt-1-model');
    const okLlm = llmSpans.find((s) => attrs(s)['gen_ai.request.model'] === 'attempt-2-model');
    assert.ok(errorLlm, 'attempt 1’s LLM span must be present');
    assert.ok(okLlm, 'attempt 2’s LLM span must be present');
    assert.equal(
      errorLlm!.status.code,
      SpanStatusCode.ERROR,
      'attempt 1’s LLM span must carry the ERROR status from its failing stopReason',
    );
    assert.equal(
      okLlm!.status.code,
      SpanStatusCode.UNSET,
      'attempt 2’s LLM span must NOT be ERROR — it is the successful retry',
    );
    assert.equal(
      errorLlm!.parentSpanId,
      root.spanContext().spanId,
      'attempt 1’s (failed) LLM span must parent under the single shared root, not a discarded one',
    );
    assert.equal(
      okLlm!.parentSpanId,
      root.spanContext().spanId,
      'attempt 2’s (succeeded) LLM span must parent under the SAME single shared root as attempt 1',
    );
  });

  it('a genuine second session.prompt() call after the first runs agent_end already fired cleanly produces a fully separate span tree, reusing the same subscribe() listener rather than re-subscribing', async () => {
    const capture = new CapturingExporter();
    const Session = makeFakeSessionClass();
    const sdk = { AgentSession: Session };
    registerCapturingProvider(capture);
    instrumentPiCodingAgent(sdk, {});
    const session = new Session();

    const done1 = session.prompt('first task');
    session.emit({ type: 'agent_start' });
    session.emit({ type: 'message_start', message: assistantMessage({ model: 'run-1-model' }) });
    session.emit({ type: 'message_end', message: assistantMessage({ model: 'run-1-model' }) });
    session.emit({
      type: 'tool_execution_start',
      toolCallId: 'call-1',
      toolName: 'bash',
      args: { command: 'echo run1' },
    });
    session.emit({
      type: 'tool_execution_end',
      toolCallId: 'call-1',
      toolName: 'bash',
      result: {},
      isError: false,
    });
    session.emit({ type: 'turn_end', message: assistantMessage(), toolResults: [] });
    session.emit({
      type: 'agent_end',
      messages: [assistantMessage({ content: [{ type: 'text', text: 'run1 done' }] })],
      willRetry: false,
    });
    await done1;

    assert.equal(capture.spans.length, 3, 'run 1: root + LLM + tool span');

    // Deliberately reuses run 1's toolCallId ("call-1") to prove a stale Map entry can't bleed into run 2.
    const done2 = session.prompt('second task');
    session.emit({ type: 'agent_start' });
    session.emit({ type: 'message_start', message: assistantMessage({ model: 'run-2-model' }) });
    session.emit({ type: 'message_end', message: assistantMessage({ model: 'run-2-model' }) });
    session.emit({
      type: 'tool_execution_start',
      toolCallId: 'call-1',
      toolName: 'bash',
      args: { command: 'echo run2' },
    });
    session.emit({
      type: 'tool_execution_end',
      toolCallId: 'call-1',
      toolName: 'bash',
      result: {},
      isError: false,
    });
    session.emit({ type: 'turn_end', message: assistantMessage(), toolResults: [] });
    session.emit({
      type: 'agent_end',
      messages: [assistantMessage({ content: [{ type: 'text', text: 'run2 done' }] })],
      willRetry: false,
    });
    await done2;

    assert.equal(
      session.subscribeCallCount,
      1,
      'a second real prompt() call on the same session instance must not re-subscribe — the ' +
        'subscribedSessions WeakSet guard must hold across repeated prompt() calls, not just ' +
        'across repeated instrumentPiCodingAgent() calls',
    );

    assert.equal(capture.spans.length, 6, 'run 1 (3) + run 2 (3), none dropped or duplicated');

    const rootSpans = capture.spans.filter((s) => attrs(s)['openinference.span.kind'] === 'AGENT');
    assert.equal(rootSpans.length, 2);
    const run1Root = rootSpans.find((s) => attrs(s)['input.value'] === 'first task');
    const run2Root = rootSpans.find((s) => attrs(s)['input.value'] === 'second task');
    assert.ok(run1Root, "run 1's root span must carry run 1's own prompt text");
    assert.ok(
      run2Root,
      "run 2's root span must carry run 2's own prompt text, not a stale copy of run 1's",
    );
    assert.equal(attrs(run1Root!)['output.value'], 'run1 done');
    assert.equal(attrs(run2Root!)['output.value'], 'run2 done');
    assert.notEqual(
      run1Root!.spanContext().traceId,
      run2Root!.spanContext().traceId,
      'the two runs must live in genuinely separate traces',
    );

    const toolSpans = capture.spans.filter((s) => attrs(s)['gen_ai.tool.call.id'] === 'call-1');
    assert.equal(toolSpans.length, 2, 'each run gets its own span despite reusing the toolCallId');
    const run1Tool = toolSpans.find((s) => s.name.includes('run1'));
    const run2Tool = toolSpans.find((s) => s.name.includes('run2'));
    assert.ok(run1Tool);
    assert.ok(run2Tool);
    assert.notEqual(
      run1Tool!.spanContext().traceId,
      run2Tool!.spanContext().traceId,
      "run 2's reused-id tool span must not be attached to run 1's trace",
    );
    assert.equal(
      run1Tool!.spanContext().traceId,
      run1Root!.spanContext().traceId,
      "run 1's tool span must belong to run 1's own trace",
    );
    assert.equal(
      run2Tool!.spanContext().traceId,
      run2Root!.spanContext().traceId,
      "run 2's tool span must belong to run 2's own trace, not leak forward from run 1's now-stale state",
    );
  });
});

describe('install rollback', () => {
  // Two failure points along the install path must both leave AgentSession.prototype exactly as found:
  // a failure mid-patching, and a failure from the final wrap-once stamp after every patch succeeded.

  const WRAPPED = Symbol.for('traceroot.pi_coding_agent.wrapped');

  it('a setup failure after the wrap-once decision does not leave the guard stamped or the prototype half-patched', () => {
    const provider = new NodeTracerProvider();
    provider.register();
    try {
      // `steer` throws the first time setup probes it, standing in for any exception mid-install.
      class ThrowingSteerSession {
        sessionId = 's';
        prompt(): void {}
        subscribe(): () => void {
          return () => {};
        }
        get steer(): unknown {
          throw new Error('injected setup failure while probing steer');
        }
      }
      const proto = ThrowingSteerSession.prototype as unknown as Record<PropertyKey, unknown>;
      const originalPrompt = proto.prompt;
      const sdk = { AgentSession: ThrowingSteerSession };

      // A mid-setup failure must surface as a clear, rethrown install error, not be swallowed.
      assert.throws(
        () => instrumentPiCodingAgent(sdk, {}),
        /failed to install/i,
        'a mid-setup failure must surface as a clear instrumentation-install error',
      );

      // The guard must NOT be stamped after a failed install: leaving it true would reject every retry.
      assert.equal(
        proto[WRAPPED],
        undefined,
        'the wrap-once guard must not be stamped when install failed partway through',
      );

      // prompt must be the ORIGINAL method, not a half-installed wrapper, or a retry double-instruments.
      assert.equal(
        proto.prompt,
        originalPrompt,
        'a failed install must roll the prompt patch back, leaving the prototype unpatched',
      );
    } finally {
      trace.disable();
    }
  });

  it('a throw from the final WRAPPED stamp rolls every method patch back to its original', () => {
    const provider = new NodeTracerProvider();
    provider.register();
    try {
      // Every method is present, so every patch succeeds and install reaches the final stamping step.
      class FullSession {
        sessionId = 's';
        prompt(): void {}
        subscribe(): () => void {
          return () => {};
        }
        steer(): void {}
        followUp(): void {}
        dispose(): void {}
      }
      const proto = FullSession.prototype as unknown as Record<PropertyKey, unknown>;

      // Pre-define WRAPPED as non-configurable so setup proceeds, but the final stamp's defineProperty
      // can't redefine it and throws — standing in for a frozen/sealed prototype at that final step.
      Object.defineProperty(proto, WRAPPED, {
        value: false,
        configurable: false,
        writable: false,
        enumerable: false,
      });

      const originalPrompt = proto.prompt;
      const originalSteer = proto.steer;
      const originalFollowUp = proto.followUp;
      const originalDispose = proto.dispose;
      const sdk = { AgentSession: FullSession };

      assert.throws(
        () => instrumentPiCodingAgent(sdk, {}),
        /failed to install/i,
        'a throw from the final WRAPPED stamp must surface as an instrumentation-install error',
      );

      // Every patch must roll back even though only the trailing stamp failed, or the guard stays unset.
      assert.equal(proto.prompt, originalPrompt, 'prompt must be rolled back to the original');
      assert.equal(proto.steer, originalSteer, 'steer must be rolled back to the original');
      assert.equal(
        proto.followUp,
        originalFollowUp,
        'followUp must be rolled back to the original',
      );
      assert.equal(proto.dispose, originalDispose, 'dispose must be rolled back to the original');
    } finally {
      trace.disable();
    }
  });
});

describe('close-root-span backward scan', () => {
  // stampRootOutput's search for the last assistant message must keep walking past trailing
  // non-assistant messages; it only stamps output.value, so the test ends the span explicitly.
  function makeTracer() {
    const capture = new CapturingExporter();
    const provider = new NodeTracerProvider();
    provider.addSpanProcessor(new SimpleSpanProcessor(capture));
    const tracer = provider.getTracer('close-root-span-backward-scan-test');
    return { tracer, capture };
  }

  function userMessage(text: string): UserMessage {
    return { role: 'user', content: text, timestamp: 0 };
  }

  function assistantMessage(text: string): AssistantMessage {
    return {
      role: 'assistant',
      content: [{ type: 'text', text }],
      api: 'anthropic-messages',
      provider: 'anthropic',
      model: 'claude-sonnet-5',
      usage: {
        input: 10,
        output: 5,
        cacheRead: 0,
        cacheWrite: 0,
        totalTokens: 15,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
      },
      stopReason: 'stop',
      timestamp: 0,
    };
  }

  function attrs(span: ReadableSpan): Record<string, unknown> {
    return span.attributes as Record<string, unknown>;
  }

  it('stampRootOutput still finds the last assistant message when later messages are not assistant messages', () => {
    const { tracer, capture } = makeTracer();
    const span = tracer.startSpan('AgentSession.prompt');

    // The assistant reply is buried, not the final entry.
    const finalMessages: AgentMessage[] = [
      userMessage('question'),
      assistantMessage('the real answer'),
      {
        role: 'toolResult',
        toolCallId: 't1',
        toolName: 'bash',
        content: [],
        isError: false,
        timestamp: 0,
      },
    ];

    stampRootOutput(span, finalMessages, true);
    span.end();

    const exported = capture.spans[0];
    assert.ok(exported, 'expected the root span to have been exported');
    assert.equal(attrs(exported)['output.value'], 'the real answer');
  });
});

describe('tracer reresolution', () => {
  // TraceRoot.shutdown() swaps the OTel API's ProxyTracerProvider for a new instance rather than
  // mutating it, so a tracer captured once at wrap time would go dark; createReresolvingTracer instead
  // re-resolves through the global `trace` facade on every span-open.

  function registerProvider(): { provider: NodeTracerProvider; exporter: InMemorySpanExporter } {
    const exporter = new InMemorySpanExporter();
    const provider = new NodeTracerProvider();
    provider.addSpanProcessor(new SimpleSpanProcessor(exporter));
    provider.register();
    return { provider, exporter };
  }

  // The three process-wide resets TraceRoot.shutdown() performs.
  async function runShutdownSequence(provider: NodeTracerProvider): Promise<void> {
    await provider.shutdown();
    trace.disable();
    context.disable();
    propagation.disable();
  }

  it('an already-instrumented AgentSession keeps exporting across a shutdown()/initialize() cycle', async () => {
    const a = registerProvider();
    let b: { provider: NodeTracerProvider; exporter: InMemorySpanExporter } | undefined;
    try {
      const Session = makeFakeSessionClass();
      const sdk = { AgentSession: Session };
      instrumentPiCodingAgent(sdk, {}); // wraps the prototype once, tracer re-resolved per span

      const s1 = new Session();
      const done1 = s1.prompt('first run');
      s1.emit({ type: 'agent_start' });
      s1.emit({
        type: 'agent_end',
        messages: [assistantMessage({ content: [{ type: 'text', text: 'done A' }] })],
        willRetry: false,
      });
      await done1;
      assert.ok(
        a.exporter.getFinishedSpans().some((s) => attrs(s)['openinference.span.kind'] === 'AGENT'),
        'sanity: the first run must export through provider A',
      );

      // The prototype stays wrapped, so recovery depends entirely on the tracer re-resolving.
      await runShutdownSequence(a.provider);
      b = registerProvider();

      const s2 = new Session();
      const done2 = s2.prompt('second run');
      s2.emit({ type: 'agent_start' });
      s2.emit({
        type: 'agent_end',
        messages: [assistantMessage({ content: [{ type: 'text', text: 'done B' }] })],
        willRetry: false,
      });
      await done2;

      const rootB = b.exporter
        .getFinishedSpans()
        .find((s) => attrs(s)['openinference.span.kind'] === 'AGENT');
      assert.ok(
        rootB,
        'the second run must export through provider B after the cycle — no silent black hole',
      );
      assert.equal(attrs(rootB!)['input.value'], 'second run');
    } finally {
      trace.disable();
    }
  });
});

describe('real SDK shape smoke', () => {
  // src/pi.ts hand-cites private internals of the real SDK, none part of a stable public contract;
  // these verify those shapes against the actual devDependency so a version bump fails loudly here.

  // ESM-only, so loaded via dynamic import() rather than a static import this CommonJS file can't use.
  type RealSdk = {
    AgentSession?: { prototype: Record<string, unknown> };
  };
  let cachedSdk: Promise<RealSdk> | undefined;
  function loadRealSdk(): Promise<RealSdk> {
    if (!cachedSdk) {
      cachedSdk = import('@earendil-works/pi-coding-agent') as unknown as Promise<RealSdk>;
    }
    return cachedSdk;
  }

  it('the real AgentSession still exposes every prototype method instrumentPiCodingAgent patches or reads', async () => {
    const sdk = await loadRealSdk();
    const AgentSession = sdk.AgentSession;
    assert.equal(typeof AgentSession, 'function', 'AgentSession must still be an exported class');
    const proto = AgentSession!.prototype;

    // Required by the install guard, which disables itself if either is missing.
    assert.equal(
      typeof proto.prompt,
      'function',
      'AgentSession.prototype.prompt must exist -- the whole patch layer keys off it',
    );
    assert.equal(
      typeof proto.subscribe,
      'function',
      'AgentSession.prototype.subscribe must exist -- the entire span tree is built from one subscribe() listener',
    );

    // Optional-guarded in the patch layer, but expected present on the real SDK.
    assert.equal(
      typeof proto.steer,
      'function',
      'AgentSession.prototype.steer must exist -- patched as a standalone first-interaction entry point',
    );
    assert.equal(
      typeof proto.followUp,
      'function',
      'AgentSession.prototype.followUp must exist -- patched as a standalone first-interaction entry point',
    );
    assert.equal(
      typeof proto.dispose,
      'function',
      'AgentSession.prototype.dispose must exist -- patched to force-close in-flight spans on teardown',
    );

    // isStreaming is a getter read live on every prompt() call; a renamed/dropped getter would silently
    // reintroduce the mid-stream-steer trace-beheading bug this check exists to prevent.
    assert.equal(
      typeof Object.getOwnPropertyDescriptor(proto, 'isStreaming')?.get,
      'function',
      'AgentSession.prototype.isStreaming must still be a getter -- proto.prompt reads it to detect a mid-stream queue-only steer/followUp call',
    );
  });

  it('the real AgentSession.subscribe() still pushes onto a private _eventListeners array its unsubscribe closure splices back out', async () => {
    const sdk = await loadRealSdk();
    // The "no cleanup needed" design rests on subscribe() pushing onto a private field literally named
    // _eventListeners, which dispose() reassigns to []; a rename leaves this seeded array untouched.
    const proto = sdk.AgentSession!.prototype as unknown as { subscribe(l: unknown): () => void };
    const instance = Object.create(proto) as {
      _eventListeners: unknown[];
      subscribe(l: unknown): () => void;
    };
    instance._eventListeners = [];
    const listener = (): void => {};

    const unsubscribe = instance.subscribe(listener);
    assert.equal(typeof unsubscribe, 'function', 'subscribe() must return an unsubscribe closure');
    assert.ok(
      instance._eventListeners.includes(listener),
      "subscribe() must push the listener onto a private array field named _eventListeners -- pi.ts's no-cleanup design depends on this exact field name",
    );

    unsubscribe();
    assert.ok(
      !instance._eventListeners.includes(listener),
      'the unsubscribe closure returned by subscribe() must splice the listener back out of _eventListeners',
    );
  });
});
