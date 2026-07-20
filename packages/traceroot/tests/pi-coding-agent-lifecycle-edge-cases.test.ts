import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { SpanStatusCode, context, propagation, trace } from '@opentelemetry/api';
import type { ExportResult } from '@opentelemetry/core';
import { ExportResultCode } from '@opentelemetry/core';
import { InMemorySpanExporter, SimpleSpanProcessor } from '@opentelemetry/sdk-trace-base';
import type { ReadableSpan, SpanExporter } from '@opentelemetry/sdk-trace-base';
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import { instrumentPiCodingAgent } from '../src/pi/instrumentation';
import { stampRootOutput } from '../src/pi/spans';
import type { AgentEvent, AgentMessage, AssistantMessage, UserMessage } from '../src/pi/types';
import { assistantMessage, attrs, makeFakeSessionClass } from './pi-test-helpers';

describe('instrumentation edge cases', () => {
  // Copied locally — no shared state across test files, matching
  // packages/mastra/tests/exporter-adversarial.test.ts's explicit convention.
  class CapturingExporter implements SpanExporter {
    readonly spans: ReadableSpan[] = [];
    export(spans: ReadableSpan[], resultCallback: (result: ExportResult) => void): void {
      this.spans.push(...spans);
      resultCallback({ code: ExportResultCode.SUCCESS });
    }
    async shutdown(): Promise<void> {}
  }

  // Registers a real, freshly-registered global TracerProvider wired to
  // `capture`, replacing the deleted private-exporter (`_spanExporter`)
  // injection path — see pi-test-helpers.ts's makeRig() for the full
  // isolation rationale behind calling trace.disable() first. This file keeps
  // its own local rig (per-test CapturingExporter/Session construction, not
  // pi-test-helpers.ts's makeRig()) because several tests below assert on
  // direct AgentSession.prototype.prompt identity/subscribeCallCount, which
  // need the raw Session class, not the rig's wrapped return shape.
  function registerCapturingProvider(capture: CapturingExporter): void {
    trace.disable();
    const provider = new NodeTracerProvider();
    provider.addSpanProcessor(new SimpleSpanProcessor(capture));
    provider.register();
  }

  // prompt()'s returned promise settles on the FINAL agent_end (willRetry !==
  // true), not merely by being called — see pi-test-helpers.ts's module header
  // for the full rationale (mirrored here since this file keeps its own local
  // fixture rather than importing makeFakeSessionClass, to keep
  // subscribeCallCount and direct prototype-identity assertions self-contained).
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

  it('instrumenting a non-extensible sdk object (e.g. a real `import * as pi` ES module namespace) does not throw', async () => {
    // `import * as pi from "@earendil-works/pi-coding-agent"` — the exact
    // usage documented in this package's own README — hands back an ES module
    // namespace exotic object. Per the ECMAScript spec, module namespace
    // objects are always non-extensible, even though the values reachable
    // through them (like AgentSession and its prototype) are ordinary,
    // extensible objects. `Object.preventExtensions` reproduces that shape
    // precisely enough to catch a regression: instrumentPiCodingAgent() must
    // never try to stamp its wrap-once guard directly onto `sdk` itself.
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

    // A second call against the same frozen namespace must still be an
    // idempotent no-op, proving the wrap-once guard now lives somewhere that
    // actually persists (AgentSession.prototype), not on the frozen `sdk`.
    const wrappedOnce = Session.prototype.prompt;
    assert.doesNotThrow(() => {
      instrumentPiCodingAgent(sdk, {});
    });
    assert.equal(Session.prototype.prompt, wrappedOnce);
  });

  it('the wrap-once guard key is the globally-interned Symbol.for() value, and a second call warns and is rejected', () => {
    // Regression test for silent double-instrumentation across two
    // independently-loaded copies of this package sharing one
    // AgentSession.prototype (e.g. a monorepo hoisting/dedup failure that
    // leaves two differently-versioned installs of @traceroot-ai/pi both
    // patching the same underlying @earendil-works/pi-coding-agent instance).
    //
    // A module-scoped `Symbol()` guard would fail this scenario silently: each
    // loaded copy gets its own distinct, non-interned symbol, so one copy's
    // wrap-once stamp is invisible to the other copy's guard check — both
    // copies patch prompt/steer/followUp/dispose, and every real session call
    // then runs through two independent listener layers, doubling every span
    // export forever with zero warning. Symbol.for(key), by spec, is looked up
    // in the process-wide global symbol registry: ANY code anywhere in the
    // process that calls Symbol.for() with this exact string gets back the
    // IDENTICAL symbol value, so a second copy's guard check correctly sees the
    // first copy's stamp and rejects (warn + no-op) instead of silently
    // double-wrapping.
    //
    // Asserted directly against the registry rather than by loading two
    // physical copies of the module: Symbol.for(key) is spec-guaranteed to
    // return the exact same value on every call anywhere in the process, so
    // "the stamped guard key equals Symbol.for(that same key)" is exactly the
    // property that makes cross-copy detection work.
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

    // Confirm the warn-and-reject path actually fires on a second call: a
    // different config (a different tenant's captureContent setting, here)
    // must be rejected, not silently applied on top of the first.
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
    // Only the root span carries session.id and only tool spans carry
    // gen_ai.tool.call.id — the LLM span carries neither, so "everything
    // that isn't B's root" is the correct way to count A's 3 spans.
    const aRelated = capture.spans.filter((s) => s !== bRoot);
    assert.equal(aRelated.length, 3, 'A: root + LLM + tool span');
    // B's root span must not be a parent/child of anything in A's tree.
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
    // No tool_execution_end — simulates an aborted run mid-tool-call.
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
    // A message_end with `role: 'assistant'` but a malformed/missing `usage`
    // field must not crash the listener — attribute setters must tolerate it.
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

  // This fake's prompt() is declared `async`, so `throw` inside it is captured
  // into a REJECTED PROMISE — the call to session.prompt('hi') itself never
  // throws; only the promise it returns rejects. This exercises proto.prompt's
  // `result.then(onResolve, onReject)` branch. See the DEDICATED
  // synchronous-throw test directly below for the materially different case
  // (the call to session.prompt(...) itself throwing, before ever returning a
  // promise), which exercises proto.prompt's OTHER catch branch — the
  // `try { result = originalPrompt.call(...) } catch (err) { ... }` around the
  // call itself.
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
    // agent_start never fires for a run that failed validation before starting,
    // but the root span proto.prompt opened at entry is still finalized (as
    // ERROR, per the DECIDED rejection/throw boundary policy) rather than left
    // dangling open forever.
    assert.equal(capture.spans.length, 1);
    const rootSpan = capture.spans[0]!;
    assert.equal(attrs(rootSpan)['openinference.span.kind'], 'AGENT');
    assert.equal(rootSpan.status.code, 2 /* SpanStatusCode.ERROR */);
    assert.equal(
      rootSpan.events.some((e) => e.name === 'exception'),
      true,
    );
  });

  // F5: both of this suite's pre-existing "sync" fakes (the one above, and
  // pi-test-helpers.ts's shouldReject predicate) are ASYNC-function throws —
  // i.e. rejected promises, not genuine synchronous throws — so
  // instrumentation.ts's proto.prompt SYNCHRONOUS catch branch
  // (`try { result = originalPrompt.call(this, text, options); } catch (err)
  // { ...; throw err; }`) had zero real coverage anywhere in this suite. This
  // fake's prompt() is a plain (non-async) function that throws BEFORE ever
  // constructing or returning a promise, exercising that exact branch: the
  // wrapper must catch it, finalize the root as ERROR with the exception
  // recorded, and then RETHROW SYNCHRONOUSLY — the call to session.prompt(...)
  // itself must throw, not return a rejected promise.
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

    // Under the pre-fix (agent_end-anchored) model this scenario produced TWO
    // separate root spans, each independently closed, with an unconditional
    // per-attempt `traceroot.pi.will_retry` flag (true, then false) — pi's own
    // internal retry control flow leaked into trace *structure*. Flipped here
    // for the new one-trace-per-prompt() contract: the retry continuation's
    // agent_start/agent_end fire with NO new prompt() call in between (see
    // instrumentation.ts's module header on _runAgentPrompt's continuation
    // loop), so both attempts must land under the SAME still-open root, and
    // that root closes exactly once — when prompt()'s own promise settles —
    // carrying retry_count: 1 instead of a per-attempt will_retry flag.
    //
    // F4: extended beyond the bare root+retry_count assertion to also drive a
    // real child LLM span through each attempt (attempt 1 fails at the
    // provider with an error stopReason and triggers the retry; attempt 2
    // succeeds), proving the retry's own child spans — not just the root —
    // land under the single shared trace.
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
    // A retry re-enters the loop and fires a fresh agent_start/agent_end pair
    // — no new prompt() call, same window.
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

    // Run 1: a full, clean turn with a tool call, then agent_end.
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

    // Run 2: a genuine second call to prompt() on the SAME session instance
    // (e.g. the user sends a second chat message), well after run 1's
    // prompt() promise already settled and tore its state down. Deliberately
    // reuses run 1's toolCallId ("call-1") to prove a stale Map entry from run
    // 1 cannot bleed into run 2 — the prompt() settle already cleared
    // state.toolSpans (via agent_end's own sweep).
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
    // ReadableSpan has no top-level traceId field — it lives under
    // spanContext().traceId, matching how root-span trace ids are read above.
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
  /**
   * Lens: instrumentPiCodingAgent() install rollback.
   *
   * instrumentPiCodingAgent() stamps AgentSession.prototype with its wrap-once
   * guard. Two distinct failure points along that install path must both leave
   * the prototype exactly as found — never half-patched, never stamped without
   * being fully patched:
   *
   *  - A failure DURING the method patching (a throwing `steer` probe below).
   *    If the wrap-once stamp were applied BEFORE setup finishes and setup then
   *    throws, the prototype is left permanently marked "wrapped" while the SDK
   *    is never actually patched — so every later instrumentPiCodingAgent()
   *    call is silently rejected with "config ignored" instead of the real
   *    setup failure ever surfacing.
   *
   *  - A failure from the FINAL wrap-once stamp itself, after every method
   *    patch has already succeeded (Object.defineProperty(proto, WRAPPED, ...)
   *    throwing because the prototype was frozen/sealed in between, or a
   *    conflicting non-configurable WRAPPED-keyed property already exists).
   *    Originally that stamp sat OUTSIDE the try/catch, so its throw escaped
   *    rollback: the prototype was left fully patched but UNSTAMPED. The next
   *    instrumentPiCodingAgent() call then saw no guard, re-patched the
   *    already-patched prompt/steer/followUp/dispose, and every event emitted
   *    DUPLICATE spans forever.
   *
   * Both tests prove the stamp lands only after setup fully succeeds and lives
   * inside the try: the failure surfaces as a clear error, and a partial
   * install is rolled back rather than left half-applied (which a later retry
   * would then double-wrap).
   */

  // Same registry key the instrumentation stamps on AgentSession.prototype.
  const WRAPPED = Symbol.for('traceroot.pi_coding_agent.wrapped');

  it('a setup failure after the wrap-once decision does not leave the guard stamped or the prototype half-patched', () => {
    // Register a real global provider so the install takes the shared-mode path
    // (no private provider, no beforeExit hook) — keeps this test focused purely
    // on the guard-ordering/rollback behavior, not the flush-hook plumbing.
    const provider = new NodeTracerProvider();
    provider.register();
    try {
      // An AgentSession whose `steer` getter throws the first time the
      // instrumentation probes `typeof proto.steer === 'function'` during setup,
      // standing in for ANY exception thrown partway through installing the
      // patches. prompt() is patched before that probe, so a naive install
      // leaves prompt half-wrapped when the probe throws.
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

      // A mid-setup failure must surface as a clear, rethrown install error —
      // not be swallowed, and not be masked as a later "config ignored".
      assert.throws(
        () => instrumentPiCodingAgent(sdk, {}),
        /failed to install/i,
        'a mid-setup failure must surface as a clear instrumentation-install error',
      );

      // The wrap-once guard must NOT be stamped after a failed install: leaving
      // it true would silently reject every subsequent install attempt.
      assert.equal(
        proto[WRAPPED],
        undefined,
        'the wrap-once guard must not be stamped when install failed partway through',
      );

      // The prototype must be left exactly as found: prompt must be the ORIGINAL
      // method, not a half-installed wrapper. Otherwise a later retry re-wraps an
      // already-wrapped prompt, double-instrumenting every call.
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
    // Shared-mode path (a real global provider registered) so the install does
    // not build a private provider or a beforeExit hook — keeps this focused on
    // the stamp-failure rollback, not flush plumbing.
    const provider = new NodeTracerProvider();
    provider.register();
    try {
      // A fully-patchable AgentSession: prompt/subscribe/steer/followUp/dispose
      // all present, so EVERY method patch below succeeds and the install reaches
      // its final stamping step with the prototype fully patched.
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

      // Booby-trap the final stamp: pre-define WRAPPED as a NON-configurable,
      // non-writable data property whose value is falsy. The falsy value lets the
      // early `if (proto[WRAPPED])` guard pass (so setup proceeds and all method
      // patches apply), but the closing Object.defineProperty(proto, WRAPPED,
      // { value: true, ... }) cannot redefine a non-configurable property to a new
      // value — it throws a TypeError, standing in for a frozen/sealed prototype
      // or a genuine WRAPPED-key collision at that exact final step.
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

      // (a) The stamp failure must surface as a clear, rethrown install error.
      assert.throws(
        () => instrumentPiCodingAgent(sdk, {}),
        /failed to install/i,
        'a throw from the final WRAPPED stamp must surface as an instrumentation-install error',
      );

      // (b) Every method patch must be rolled back to its ORIGINAL function, even
      // though all of them succeeded and only the trailing stamp failed. Without
      // rollback covering the stamp, these would remain wrapped while the guard
      // is unset — the exact state that double-instruments on the next call.
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
  /**
   * Lens: stampRootOutput's search for the last assistant message
   * (src/spans.ts) — it must scan backward directly rather than copying the
   * whole finalMessages array and reversing it before find() ever runs.
   * Mirrors spans-truncation.test.ts's last test in spirit ('a single huge
   * string tool argument is capped during serialization, not only after the
   * fact'): the *final output* of a copy+reverse+find and a backward for-loop
   * is byte-identical, so asserting only on the exported attribute can't tell
   * a fixed implementation apart from the wasteful one. Instead this spies on
   * Array.prototype.reverse to observe *how* the search happened — a
   * full-length reverse() call is exactly the bug: it means the entire history
   * was copied and reversed up front, even in the common case where the very
   * last message already is the assistant reply.
   *
   * stampRootOutput() only stamps output.value; it does not end the span (that
   * is finalizeRootSpan's job, called once when the enclosing prompt() call's
   * own promise settles — see instrumentation.ts's module header for the split
   * rationale), so each test below ends the span explicitly to make it visible
   * to the capturing exporter.
   */

  // Copied locally — no shared state across test files, matching every other
  // *.test.ts file's explicit convention in this package.
  class CapturingExporter implements SpanExporter {
    readonly spans: ReadableSpan[] = [];
    export(spans: ReadableSpan[], resultCallback: (result: ExportResult) => void): void {
      this.spans.push(...spans);
      resultCallback({ code: ExportResultCode.SUCCESS });
    }
    async shutdown(): Promise<void> {}
  }

  // closeRootSpan only needs a real Span (setAttribute/end), not the full
  // instrumentPiCodingAgent event pipeline — a direct NodeTracerProvider +
  // SimpleSpanProcessor rig (same shape as src/provider.ts's createTracing,
  // minus the OTLP exporter) is the most direct way to unit-test this
  // function in isolation.
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

  it('stampRootOutput finds the last assistant message without copying+reversing the full history', () => {
    const { tracer, capture } = makeTracer();
    const span = tracer.startSpan('AgentSession.prompt');

    // Long-running session shape from the finding: many turns accumulated,
    // and the very last message already is the assistant reply — the case a
    // backward scan resolves in one comparison.
    const finalMessages: AgentMessage[] = [];
    for (let i = 0; i < 5000; i++) {
      finalMessages.push(userMessage(`turn ${i}`));
    }
    finalMessages.push(assistantMessage('final answer'));

    const originalReverse = Array.prototype.reverse;
    let reverseCalledOnFullHistory = false;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    Array.prototype.reverse = function reverseSpy(this: any[]) {
      if (this.length === finalMessages.length) {
        reverseCalledOnFullHistory = true;
      }
      return originalReverse.call(this);
    };

    try {
      stampRootOutput(span, finalMessages, true);
    } finally {
      Array.prototype.reverse = originalReverse;
    }
    span.end();

    assert.equal(
      reverseCalledOnFullHistory,
      false,
      'stampRootOutput must not copy and reverse the entire finalMessages array to find the last assistant message',
    );

    const exported = capture.spans[0];
    assert.ok(exported, 'expected the root span to have been exported');
    assert.equal(attrs(exported)['output.value'], 'final answer');
  });

  it('stampRootOutput still finds the last assistant message when later messages are not assistant messages', () => {
    const { tracer, capture } = makeTracer();
    const span = tracer.startSpan('AgentSession.prompt');

    // The assistant reply is buried, not the final entry — a correct backward
    // scan must keep walking past the trailing non-assistant messages instead
    // of stopping at the wrong one.
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
  /**
   * Lens: tracer re-resolution across a shutdown()/initialize() provider swap.
   *
   * Ported from packages/pi/tests/provider-shared-mode-behavior.test.ts, which
   * is otherwise deleted for this in-tree migration (it mostly probed the old
   * standalone package's private-vs-shared-provider mode detection — apiKey,
   * _spanExporter, hasRealGlobalProvider() — none of which exist in this
   * integration; see packages/traceroot/src/pi/config.ts's own header: there
   * is no export-pipeline configuration here at all, since core always
   * guarantees a real, globally-registered OTel provider before
   * instrumentPiCodingAgent() ever runs). This one scenario survives because it
   * proves something still true and still load-bearing: the collapsed
   * tracer-acquisition approach (see instrumentation.ts's
   * createReresolvingTracer) must keep exporting through a provider SWAP, not
   * just through the single provider that was live at wrap time.
   *
   * TraceRoot.shutdown() (and anything else that calls trace.disable()) swaps
   * the OTel API's internal ProxyTracerProvider for a brand-new instance rather
   * than mutating the old one, so a tracer captured once at wrap time and
   * closed over forever would stay bound to the old, now-detached provider —
   * every subsequent span would go dark permanently after a
   * shutdown()/initialize() cycle, with no recovery path (the Symbol.for()
   * wrap-once guard blocks re-instrumenting to pick up a fresh tracer).
   * createReresolvingTracer instead re-resolves through the global `trace`
   * facade on every span-open, so this reproduces the exact repro: register
   * provider A, instrument, run once (lands in A), run the shutdown reset
   * sequence, register provider B, run again on the SAME already-instrumented
   * session (must land in B).
   */

  function registerProvider(): { provider: NodeTracerProvider; exporter: InMemorySpanExporter } {
    const exporter = new InMemorySpanExporter();
    const provider = new NodeTracerProvider();
    provider.addSpanProcessor(new SimpleSpanProcessor(exporter));
    provider.register();
    return { provider, exporter };
  }

  // Exactly the three process-wide resets TraceRoot.shutdown() performs, after
  // tearing down its own provider.
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

      // First run lands in A.
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

      // Simulate TraceRoot.shutdown() -> TraceRoot.initialize(): the prototype is
      // still wrapped (Symbol.for() guard blocks re-instrumentation), so recovery
      // depends entirely on the captured tracer re-resolving, not re-wrapping.
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
  /**
   * Lens: the private @earendil-works/pi-coding-agent internals that
   * src/instrumentation.ts and src/pi/types.ts document by hand.
   *
   * Those files cite specific fields and methods of the real, installed SDK --
   * the private `_eventListeners` listener array, the `isStreaming` getter,
   * and the standalone steer()/followUp()/dispose() entry points -- all read
   * straight out of the installed source, none of them part of a stable
   * public contract. This test verifies those hand-mirrored shapes against
   * the actual installed `@earendil-works/pi-coding-agent@0.80.x`
   * devDependency, so a version bump that renames or drops any of them fails
   * loudly here instead of silently shipping broken instrumentation.
   *
   * It asserts shape only -- it never constructs a live AgentSession (the
   * real constructor needs a full agent/session/settings runtime) or
   * performs any agent work.
   */

  // @earendil-works/pi-coding-agent is ESM-only (its package.json exports only an
  // `import` condition), so it is loaded via a dynamic import() -- which always
  // goes through Node's ESM resolver and matches that condition -- rather than a
  // static import that this CommonJS-compiled test file would turn into a
  // require() the package deliberately does not support. Cached so the (heavy)
  // module graph is only evaluated once across the tests below. Accessing exports
  // off a loose record means a REMOVED export surfaces as a clear assertion
  // failure below, not an opaque module-link error.
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

    // prompt + subscribe are REQUIRED by instrumentation.ts's install guard --
    // it disables itself entirely if either is missing.
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

    // steer/followUp/dispose are optional-guarded in the patch layer, but
    // instrumentation.ts's comments assert they ARE present on the real SDK; if
    // that stops being true the comments (and the behavior they justify) are
    // stale.
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

    // isStreaming is a getter, not a plain data property -- instrumentation.ts's
    // proto.prompt reads it live on every prompt() call (isQueueOnlySteer) to
    // detect a mid-stream queue-only steer/followUp. A future SDK bump that
    // renames or drops this getter would not throw -- isQueueOnlySteer would
    // just silently and permanently evaluate to false, reintroducing the
    // mid-stream-steer trace-beheading bug that check exists to prevent -- so
    // assert its shape here rather than let that regression ship silently.
    assert.equal(
      typeof Object.getOwnPropertyDescriptor(proto, 'isStreaming')?.get,
      'function',
      'AgentSession.prototype.isStreaming must still be a getter -- proto.prompt reads it to detect a mid-stream queue-only steer/followUp call',
    );
  });

  it('the real AgentSession.subscribe() still pushes onto a private _eventListeners array its unsubscribe closure splices back out', async () => {
    const sdk = await loadRealSdk();
    // instrumentation.ts's module header and its "no cleanup needed" design rest
    // entirely on this: subscribe(listener) pushes onto a private array field
    // literally named _eventListeners, and dispose() reassigns that same field to
    // [] to stop delivery -- with no per-listener unsubscribe() call from our
    // side. A bare prototype instance (the real constructor needs a full runtime,
    // so it is deliberately bypassed) with the field pre-seeded lets us drive the
    // REAL subscribe()/unsubscribe() and confirm they still operate on a field of
    // exactly that name; a rename leaves this seeded array untouched and fails.
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
      "subscribe() must push the listener onto a private array field named _eventListeners -- instrumentation.ts's no-cleanup design depends on this exact field name",
    );

    unsubscribe();
    assert.ok(
      !instance._eventListeners.includes(listener),
      'the unsubscribe closure returned by subscribe() must splice the listener back out of _eventListeners',
    );
  });
});
