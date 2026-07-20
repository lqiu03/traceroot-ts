import assert from 'node:assert/strict';
import { afterEach, describe, it } from 'node:test';
import { trace } from '@opentelemetry/api';
import type { ReadableSpan } from '@opentelemetry/sdk-trace-base';
import { InMemorySpanExporter, Span, SimpleSpanProcessor } from '@opentelemetry/sdk-trace-base';
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import { instrumentPiCodingAgent } from '../src/pi/instrumentation';
import type { AgentEvent, AssistantMessage } from '../src/pi/types';
import { TraceRoot, _resetForTesting } from '../src/traceroot';
import {
  assistantMessage,
  attrs,
  CapturingExporter,
  makeFakeSessionClass,
  makeRig,
} from './pi-test-helpers';
import { assistantMessage as baseAssistantMessage } from './pi-test-helpers';

describe('instrumentation', () => {
  // instrumentation.test.ts asserts on specific usage/cost numbers (see the LLM
  // span assertions below), so it overrides test-helpers.ts's minimal
  // placeholder usage with realistic values here — the one place a reader
  // needs to look to find them, rather than a second copy of assistantMessage()
  // with different numbers baked in silently (see test-helpers.ts's docstring).
  function assistantMessage(overrides: Partial<AssistantMessage> = {}): AssistantMessage {
    return baseAssistantMessage({
      usage: {
        input: 100,
        output: 20,
        cacheRead: 0,
        cacheWrite: 0,
        totalTokens: 120,
        cost: { input: 0.001, output: 0.0006, cacheRead: 0, cacheWrite: 0, total: 0.0016 },
      },
      ...overrides,
    });
  }

  it('a full turn with one tool call produces a correctly nested AGENT -> LLM -> TOOL span tree', async () => {
    const { capture, Session } = makeRig();
    const session = new Session();

    // The root AGENT span is now anchored on prompt()'s own promise window
    // (see pi-test-helpers.ts's module header) — NOT awaited yet, so the
    // events below drive the run while the root is still open.
    const done = session.prompt('list files in /tmp');
    session.emit({ type: 'agent_start' });
    session.emit({ type: 'turn_start' });
    session.emit({ type: 'message_start', message: assistantMessage() });
    session.emit({
      type: 'message_end',
      message: assistantMessage({
        content: [
          { type: 'text', text: "I'll list the files now." },
          { type: 'toolCall', id: 't1', name: 'bash' },
        ],
      }),
    });
    session.emit({
      type: 'tool_execution_start',
      toolCallId: 't1',
      toolName: 'bash',
      args: { command: 'ls /tmp' },
    });
    session.emit({
      type: 'tool_execution_end',
      toolCallId: 't1',
      toolName: 'bash',
      result: { content: [{ type: 'text', text: 'a.txt' }] },
      isError: false,
    });
    session.emit({
      type: 'turn_end',
      message: assistantMessage(),
      toolResults: [],
    });
    session.emit({
      type: 'agent_end',
      messages: [assistantMessage({ content: [{ type: 'text', text: 'listed the files' }] })],
      willRetry: false,
    });
    await done;

    assert.equal(capture.spans.length, 3, 'expected exactly root, LLM, and tool spans');

    const [llmSpan, toolSpan, rootSpan] = capture.spans;

    assert.equal(rootSpan.name, 'AgentSession.prompt');
    assert.equal(attrs(rootSpan)['openinference.span.kind'], 'AGENT');
    assert.equal(attrs(rootSpan)['session.id'], 'sess-1');
    // pi no longer self-stamps traceroot.sdk.name; core's TraceRootSpanProcessor
    // owns it uniformly (matching the Claude Agent SDK integration). This rig
    // wires no such processor, so the attribute is absent here.
    assert.equal(attrs(rootSpan)['traceroot.sdk.name'], undefined);
    assert.equal(attrs(rootSpan)['input.value'], 'list files in /tmp');
    assert.equal(attrs(rootSpan)['output.value'], 'listed the files');
    // traceroot.pi.will_retry is gone (removed along with agent_end owning the
    // close); traceroot.pi.retry_count replaces it, now stamped once when the
    // enclosing prompt() call settles. No retry happened in this run, so it is 0.
    assert.equal(attrs(rootSpan)['traceroot.pi.retry_count'], 0);

    assert.equal(attrs(llmSpan)['openinference.span.kind'], 'LLM');
    assert.equal(attrs(llmSpan)['gen_ai.system'], 'anthropic');
    assert.equal(attrs(llmSpan)['gen_ai.request.model'], 'claude-sonnet-5');
    assert.equal(attrs(llmSpan)['gen_ai.usage.input_tokens'], 100);
    assert.equal(attrs(llmSpan)['gen_ai.usage.output_tokens'], 20);
    assert.equal(
      attrs(llmSpan)['output.value'],
      "I'll list the files now.",
      'captureContent:true (the default) must populate output.value on the LLM span too, not just the root span',
    );
    assert.equal(
      llmSpan.parentSpanId,
      rootSpan.spanContext().spanId,
      'LLM span must be a child of the root span',
    );

    assert.equal(toolSpan.name, 'bash: ls /tmp');
    assert.equal(attrs(toolSpan)['openinference.span.kind'], 'TOOL');
    assert.equal(attrs(toolSpan)['gen_ai.tool.name'], 'bash');
    assert.equal(attrs(toolSpan)['gen_ai.tool.call.id'], 't1');
    assert.equal(
      toolSpan.parentSpanId,
      llmSpan.spanContext().spanId,
      'tool span must be a child of the LLM span that requested it, not a sibling under root',
    );
  });

  it('two concurrent tool calls in one turn each get their own correctly-keyed span', async () => {
    const { capture, Session } = makeRig();
    const session = new Session();

    const done = session.prompt('do two things');
    session.emit({ type: 'agent_start' });
    session.emit({ type: 'message_start', message: assistantMessage() });
    session.emit({ type: 'message_end', message: assistantMessage() });
    session.emit({
      type: 'tool_execution_start',
      toolCallId: 'a',
      toolName: 'read',
      args: { path: '/x.txt' },
    });
    session.emit({
      type: 'tool_execution_start',
      toolCallId: 'b',
      toolName: 'read',
      args: { path: '/y.txt' },
    });
    session.emit({
      type: 'tool_execution_end',
      toolCallId: 'b',
      toolName: 'read',
      result: {},
      isError: false,
    });
    session.emit({
      type: 'tool_execution_end',
      toolCallId: 'a',
      toolName: 'read',
      result: {},
      isError: false,
    });
    session.emit({ type: 'turn_end', message: assistantMessage(), toolResults: [] });
    session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
    await done;

    const toolSpans = capture.spans.filter(
      (s) => (s.attributes as Record<string, unknown>)['gen_ai.tool.call.id'],
    );
    assert.equal(toolSpans.length, 2);
    const ids = toolSpans
      .map((s) => (s.attributes as Record<string, unknown>)['gen_ai.tool.call.id'])
      .sort();
    assert.deepEqual(ids, ['a', 'b']);
    assert.equal(toolSpans[0]!.name, 'read: y.txt', 'b ended first, so it should export first');
    assert.equal(toolSpans[1]!.name, 'read: x.txt');
  });

  it('a failed LLM turn (stopReason error) marks the LLM span as ERROR', async () => {
    const { capture, Session } = makeRig();
    const session = new Session();

    const done = session.prompt('trigger a provider error');
    session.emit({ type: 'agent_start' });
    session.emit({ type: 'message_start', message: assistantMessage() });
    session.emit({
      type: 'message_end',
      message: assistantMessage({ stopReason: 'error', errorMessage: 'rate limited' }),
    });
    session.emit({ type: 'turn_end', message: assistantMessage(), toolResults: [] });
    session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
    await done;

    const llmSpan = capture.spans.find(
      (s) => (s.attributes as Record<string, unknown>)['openinference.span.kind'] === 'LLM',
    );
    assert.ok(llmSpan);
    assert.equal(llmSpan!.status.code, 2 /* SpanStatusCode.ERROR */);
    assert.equal(llmSpan!.status.message, 'rate limited');
  });

  it('a failed tool call (isError) marks the TOOL span as ERROR', async () => {
    const { capture, Session } = makeRig();
    const session = new Session();

    const done = session.prompt('run a failing command');
    session.emit({ type: 'agent_start' });
    session.emit({ type: 'message_start', message: assistantMessage() });
    session.emit({ type: 'message_end', message: assistantMessage() });
    session.emit({
      type: 'tool_execution_start',
      toolCallId: 't1',
      toolName: 'bash',
      args: { command: 'false' },
    });
    session.emit({
      type: 'tool_execution_end',
      toolCallId: 't1',
      toolName: 'bash',
      result: { content: [{ type: 'text', text: 'exit 1' }] },
      isError: true,
    });
    session.emit({ type: 'turn_end', message: assistantMessage(), toolResults: [] });
    session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
    await done;

    const toolSpan = capture.spans.find(
      (s) => (s.attributes as Record<string, unknown>)['gen_ai.tool.call.id'] === 't1',
    );
    assert.ok(toolSpan);
    assert.equal(toolSpan!.status.code, 2 /* SpanStatusCode.ERROR */);
  });

  it('captureContent: false suppresses input.value/output.value but keeps other attributes', async () => {
    const { capture, Session } = makeRig({ captureContent: false });
    const session = new Session();

    const done = session.prompt('sensitive prompt text');
    session.emit({ type: 'agent_start' });
    session.emit({ type: 'message_start', message: assistantMessage() });
    session.emit({
      type: 'message_end',
      message: assistantMessage({ content: [{ type: 'text', text: 'sensitive llm reply' }] }),
    });
    session.emit({ type: 'turn_end', message: assistantMessage(), toolResults: [] });
    session.emit({
      type: 'agent_end',
      messages: [assistantMessage({ content: [{ type: 'text', text: 'sensitive reply' }] })],
      willRetry: false,
    });
    await done;

    const [llmSpan, rootSpan] = capture.spans;
    assert.equal(attrs(rootSpan!)['input.value'], undefined);
    assert.equal(attrs(rootSpan!)['output.value'], undefined);
    assert.equal(attrs(rootSpan!)['session.id'], 'sess-1');

    assert.ok(llmSpan, 'expected an LLM span to have been captured');
    assert.equal(attrs(llmSpan!)['openinference.span.kind'], 'LLM');
    assert.equal(
      attrs(llmSpan!)['output.value'],
      undefined,
      'captureContent:false must suppress output.value on the LLM span too, not just the root span',
    );
  });

  it('captureToolIo: false suppresses tool input.value/output.value but keeps the tool name', async () => {
    const { capture, Session } = makeRig({ captureToolIo: false });
    const session = new Session();

    const done = session.prompt('run a tool');
    session.emit({ type: 'agent_start' });
    session.emit({ type: 'message_start', message: assistantMessage() });
    session.emit({ type: 'message_end', message: assistantMessage() });
    // Use a path-based tool call, not a bash command: describeToolCallSpan's own
    // contract (see span-name.test.ts) is that a bash command up to 60 chars
    // appears verbatim in the span NAME by design — that tradeoff is unrelated
    // to captureToolIo, which only gates the separate input.value/output.value
    // attributes tested below. A path arg, by contrast, is unconditionally
    // reduced to its basename, which is what this test asserts.
    session.emit({
      type: 'tool_execution_start',
      toolCallId: 't1',
      toolName: 'read',
      args: { path: '/Users/alice/secret-project/notes.txt' },
    });
    session.emit({
      type: 'tool_execution_end',
      toolCallId: 't1',
      toolName: 'read',
      result: { secret: 'leaked?' },
      isError: false,
    });
    session.emit({ type: 'turn_end', message: assistantMessage(), toolResults: [] });
    session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
    await done;

    const toolSpan = capture.spans.find(
      (s) => (s.attributes as Record<string, unknown>)['gen_ai.tool.call.id'] === 't1',
    );
    assert.ok(toolSpan);
    assert.equal(attrs(toolSpan!)['input.value'], undefined);
    assert.equal(attrs(toolSpan!)['output.value'], undefined);
    assert.equal(attrs(toolSpan!)['gen_ai.tool.name'], 'read');
    assert.equal(toolSpan!.name, 'read: notes.txt');
    assert.ok(
      !toolSpan!.name.includes('secret-project'),
      'the full path must never appear in the span name',
    );
  });

  it('captureContent:false suppresses input.value on the ROOT span for an empty-string prompt exactly as for an absent one, while captureContent:true legitimately records the empty string', async () => {
    async function rootInputValue(
      captureContent: boolean,
      promptText: string | undefined,
    ): Promise<{ hasKey: boolean; value: unknown }> {
      const { capture, Session } = makeRig({ captureContent });
      const session = new Session();
      // Calling prompt() with a non-string bypasses proto.prompt's own
      // `typeof text === 'string'` guard, simulating a caller whose prompt
      // text is genuinely absent (as opposed to the empty-string case below)
      // while still registering the subscribe() listener that
      // instrumentPiCodingAgent wires up inside the wrapped prompt() call itself.
      const done = session.prompt(promptText as unknown as string);
      session.emit({ type: 'agent_start' });
      session.emit({ type: 'agent_end', messages: [], willRetry: false });
      await done;
      const root = capture.spans[0]!;
      return {
        hasKey: Object.prototype.hasOwnProperty.call(attrs(root), 'input.value'),
        value: attrs(root)['input.value'],
      };
    }

    const falseEmpty = await rootInputValue(false, '');
    assert.equal(falseEmpty.hasKey, false, 'captureContent:false must omit input.value for ""');
    assert.equal(falseEmpty.value, undefined);

    const falseUndefined = await rootInputValue(false, undefined);
    assert.equal(
      falseUndefined.hasKey,
      false,
      'captureContent:false must omit input.value for undefined',
    );
    assert.equal(falseUndefined.value, undefined);

    // Contrast: with captureContent on, "" is a real (if uninformative) value
    // and must be distinguishable from "no prompt text at all".
    const trueEmpty = await rootInputValue(true, '');
    assert.equal(trueEmpty.hasKey, true, 'captureContent:true legitimately sets input.value to ""');
    assert.equal(trueEmpty.value, '');

    const trueUndefined = await rootInputValue(true, undefined);
    assert.equal(
      trueUndefined.hasKey,
      false,
      'no prompt text at all must leave input.value unset even when captureContent is on',
    );
    assert.equal(trueUndefined.value, undefined);
  });

  // Boundary policy 2 (the prompt()-anchored root model): an early-return
  // prompt() call — a handled "/command", a queue-only steer/followUp — resolves
  // without ever reaching pi's internal run loop, so no agent_start/agent_end
  // follows. The wrapped prompt() must still open a root on entry and finalize it
  // OK on settle, yielding exactly one childless root. Migrated here from the
  // deleted pi-test-helpers.test.ts (its F2 case): this is the suite's only
  // span-asserting coverage of the no-events early-return path, so it belongs
  // with the instrumentation behavior it exercises, not among the fixture tests.
  it('an early-return prompt() call (resolved with no agent_start/agent_end) exports exactly one childless, OK-status AGENT root span', async () => {
    const { capture, Session } = makeRig();
    const session = new Session();

    const done = session.prompt('a handled slash command');
    session.resolvePrompt();
    await assert.doesNotReject(() => done);

    assert.equal(
      capture.spans.length,
      1,
      'exactly one span: the root, with ZERO child spans (no agent_start ever fired to open any)',
    );
    const rootSpan = capture.spans[0]!;
    assert.equal(rootSpan.name, 'AgentSession.prompt');
    assert.equal(attrs(rootSpan)['openinference.span.kind'], 'AGENT');
    assert.equal(
      rootSpan.status.code,
      1 /* SpanStatusCode.OK */,
      'finalize() explicitly stamps OK on a resolved call (a successfully-resolved ' +
        'promise, even with zero children, is not merely "unset")',
    );
    assert.equal(
      rootSpan.parentSpanId,
      undefined,
      'the root itself has no parent (it is the trace root)',
    );
  });
});

describe('wiring', () => {
  /**
   * Wiring-only coverage for instrumentModules.piCodingAgent, now that
   * @traceroot-ai/pi has been folded in-tree (packages/traceroot/src/pi/):
   * initialize() calls instrumentPiCodingAgent() (./src/pi/instrumentation.ts)
   * directly, with no dynamic require(), no optional-peer-package diagnosis,
   * and no apiKey/baseUrl threading (this integration builds no export
   * pipeline of its own — see pi/config.ts's module header).
   *
   * This file replaces the old pi-coding-agent.test.ts (deleted) and
   * pi-broken-transitive-dep.test.ts (deleted), whose subject — the
   * dynamic-`require('@traceroot-ai/pi')` machinery, the "not installed" vs.
   * "installed but broken" MODULE_NOT_FOUND classification, and
   * missing-export detection — no longer exists. It is deliberately narrow:
   * proving the WIRING path (unwrap, dispatch, warn-don't-throw) is correct.
   * Deep capture-toggle span behavior belongs to a later phase's moved pi
   * behavioral suite, and cross-package no-mocks span assertions already live
   * in pi-coding-agent-integration.test.ts.
   */

  // A minimal, self-contained fake `import * as pi from
  // '@earendil-works/pi-coding-agent'` namespace: a class exposing
  // prompt/subscribe/dispose on its prototype, which is exactly the structural
  // surface instrumentPiCodingAgent() patches (see src/pi/instrumentation.ts).
  // A fresh class per call, since instrumentPiCodingAgent()'s wrap-once guard
  // is stamped on AgentSession.prototype itself.
  function makePiModule() {
    class FakeAgentSession {
      sessionId = 'wiring-test-sess';
      async prompt(_text: string, _options?: unknown): Promise<void> {}
      subscribe(_listener: (event: { type: string }) => void): () => void {
        return () => {};
      }
      dispose(): void {}
    }
    return { AgentSession: FakeAgentSession };
  }

  const BASE = {
    apiKey: 'trk_pi_wiring',
    baseUrl: 'http://127.0.0.1:9',
    disableBatch: true as const,
    gitRepo: 'traceroot-ai/traceroot-ts',
    gitRef: 'pi-wiring-ref',
  };

  afterEach(() => {
    _resetForTesting();
  });

  describe('instrumentModules.piCodingAgent wiring', () => {
    it('patches AgentSession.prototype.prompt for the bare module ref form', () => {
      const pi = makePiModule();
      const originalPrompt = pi.AgentSession.prototype.prompt;

      TraceRoot.initialize({ ...BASE, instrumentModules: { piCodingAgent: pi } });

      assert.equal(TraceRoot.isInitialized(), true);
      assert.notStrictEqual(
        pi.AgentSession.prototype.prompt,
        originalPrompt,
        'the bare module ref form must patch AgentSession.prototype.prompt',
      );
    });

    it('accepts the { module, config } wrapper form and still patches the prototype', () => {
      const pi = makePiModule();
      const originalPrompt = pi.AgentSession.prototype.prompt;

      TraceRoot.initialize({
        ...BASE,
        instrumentModules: {
          piCodingAgent: { module: pi, config: { captureContent: false } },
        },
      });

      assert.equal(TraceRoot.isInitialized(), true);
      assert.notStrictEqual(
        pi.AgentSession.prototype.prompt,
        originalPrompt,
        'the { module, config } wrapper must be unwrapped and still patch the prototype',
      );
    });

    it('leaves the prototype untouched when piCodingAgent is null', () => {
      const pi = makePiModule();
      const originalPrompt = pi.AgentSession.prototype.prompt;

      // Passing pi only as a bystander module so we have something to assert
      // against; piCodingAgent itself is explicitly null.
      TraceRoot.initialize({ ...BASE, instrumentModules: { piCodingAgent: null } });

      assert.equal(TraceRoot.isInitialized(), true);
      assert.strictEqual(
        pi.AgentSession.prototype.prompt,
        originalPrompt,
        'a null piCodingAgent must never touch any AgentSession prototype',
      );
    });

    it('leaves the prototype untouched when piCodingAgent is absent', () => {
      const pi = makePiModule();
      const originalPrompt = pi.AgentSession.prototype.prompt;

      TraceRoot.initialize({ ...BASE, instrumentModules: {} });

      assert.equal(TraceRoot.isInitialized(), true);
      assert.strictEqual(
        pi.AgentSession.prototype.prompt,
        originalPrompt,
        'an absent piCodingAgent key must never touch any AgentSession prototype',
      );
    });

    it('warns but leaves isInitialized() true for a misshapen module (pi warns, does not throw)', () => {
      // No AgentSession.prototype.prompt/subscribe -- instrumentPiCodingAgent()
      // detects this and warns rather than throwing, deliberately unlike
      // wireClaudeAgentSDKInstrumentation() (which throws on a missing query()).
      const misshapen = { AgentSession: class {} };

      const warnings: string[] = [];
      const originalWarn = console.warn;
      console.warn = (...args: unknown[]) => {
        warnings.push(args.join(' '));
      };
      try {
        TraceRoot.initialize({ ...BASE, instrumentModules: { piCodingAgent: misshapen } });
      } finally {
        console.warn = originalWarn;
      }

      assert.equal(
        TraceRoot.isInitialized(),
        true,
        'a misshapen piCodingAgent must warn-and-noop, not abort initialize()',
      );
      assert.ok(
        warnings.some((w) => w.includes('instrumentation disabled')),
        'expected a warning that instrumentation was disabled for the misshapen module',
      );
    });

    it('wires cleanly alongside another instrumentModules entry in the same call', () => {
      const pi = makePiModule();
      const originalPrompt = pi.AgentSession.prototype.prompt;
      const claude = {
        query(_params: unknown): AsyncIterable<unknown> {
          return { async *[Symbol.asyncIterator]() {} };
        },
      };
      const originalQuery = claude.query;

      TraceRoot.initialize({
        ...BASE,
        instrumentModules: { claudeAgentSDK: claude, piCodingAgent: pi },
      });

      assert.equal(TraceRoot.isInitialized(), true);
      assert.notStrictEqual(
        pi.AgentSession.prototype.prompt,
        originalPrompt,
        'piCodingAgent must still be wired alongside claudeAgentSDK',
      );
      assert.notStrictEqual(
        claude.query,
        originalQuery,
        'claudeAgentSDK must still be wired alongside piCodingAgent',
      );
    });
  });
});

describe('integration', () => {
  /**
   * Real cross-package integration: drives the ACTUAL in-tree pi instrumentation
   * (packages/traceroot/src/pi/) through a REAL TraceRoot.initialize({
   * instrumentModules }) call — no Module._load interception, no mocked pi
   * export. This is the end-to-end proof that the shared-pipeline wiring works:
   * initialize() calls instrumentPiCodingAgent() directly, pi auto-discovers
   * TraceRoot's freshly-registered global provider and runs in shared mode
   * (this in-tree integration never builds an export pipeline of its own), and
   * the spans pi produces land in TraceRoot's own pipeline WITH
   * TraceRootSpanProcessor's enrichment (environment / git repo / git ref /
   * span path) applied.
   *
   * Every other pi<->traceroot test mocks one side or the other; this one mocks
   * neither.
   */

  // A realistically-shaped fake AgentSession module, mirroring the shape pi's
  // own tests drive (a private listener array; prompt/subscribe/emit/dispose) —
  // the exact structural surface `import * as pi from
  // '@earendil-works/pi-coding-agent'` exposes and instrumentPiCodingAgent()
  // patches. This is passed straight into instrumentModules.piCodingAgent.
  interface FakeAgentEvent {
    type: string;
    [key: string]: unknown;
  }
  function makePiModule() {
    class FakeAgentSession {
      sessionId = 'integration-sess';
      private listeners: Array<(event: FakeAgentEvent) => void> = [];
      // prompt()'s returned promise settles only once its final agent_end
      // fires (willRetry !== true) — mirrors the real SDK, where prompt()
      // awaits its whole internal retry/compaction/follow-up loop; see
      // pi-test-helpers.ts's module header for the full rationale.
      private pending: { resolve: () => void; reject: (err: unknown) => void } | undefined;
      async prompt(_text: string): Promise<void> {
        return new Promise<void>((resolve, reject) => {
          this.pending = { resolve, reject };
        });
      }
      subscribe(listener: (event: FakeAgentEvent) => void): () => void {
        this.listeners.push(listener);
        return () => {
          this.listeners = this.listeners.filter((l) => l !== listener);
        };
      }
      emit(event: FakeAgentEvent): void {
        for (const listener of this.listeners) listener(event);
        if (event.type === 'agent_end' && !event.willRetry && this.pending) {
          const { resolve } = this.pending;
          this.pending = undefined;
          resolve();
        }
      }
      dispose(): void {
        this.listeners = [];
      }
    }
    return { AgentSession: FakeAgentSession };
  }

  // Minimal valid AssistantMessage (pi's src/types.ts shape) — enough for pi's
  // span builders to read model/provider/usage/content without throwing.
  function assistantMessage(text: string): FakeAgentEvent['message'] {
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

  function attachInMemoryExporterToGlobalProvider(): InMemorySpanExporter {
    // TraceRoot.initialize() register()s a NodeTracerProvider as the OTel global
    // delegate. Reach that same provider and add an in-memory exporter so we can
    // observe exactly what its pipeline exports — the spans have already passed
    // through TraceRootSpanProcessor's onStart enrichment by export time.
    const proxy = trace.getTracerProvider() as { getDelegate?: () => unknown };
    const delegate = (typeof proxy.getDelegate === 'function' ? proxy.getDelegate() : proxy) as {
      addSpanProcessor(processor: SimpleSpanProcessor): void;
    };
    const exporter = new InMemorySpanExporter();
    delegate.addSpanProcessor(new SimpleSpanProcessor(exporter));
    return exporter;
  }

  const attrOf = (span: ReadableSpan): Record<string, unknown> =>
    span.attributes as Record<string, unknown>;
  const spanKind = (span: ReadableSpan): unknown => attrOf(span)['openinference.span.kind'];

  afterEach(() => {
    _resetForTesting();
  });

  it('the real in-tree pi instrumentation wired through TraceRoot.initialize exports enriched spans into TraceRoot own pipeline', async () => {
    // Irrelevant to this in-tree integration (it builds no export pipeline of
    // its own and threads no apiKey), but unset for parity with how a real
    // host would configure TraceRoot without env-var credentials.
    delete process.env.TRACEROOT_API_KEY;

    const pi = makePiModule();
    const Session = pi.AgentSession;

    TraceRoot.initialize({
      apiKey: 'trk_integration_key',
      // Local, unroutable endpoint: TraceRoot's own OTLP exporter must never
      // POST integration junk to the real backend. Irrelevant to pi, which runs
      // in shared mode and builds no exporter of its own.
      baseUrl: 'http://127.0.0.1:9',
      disableBatch: true,
      environment: 'integration-test',
      gitRepo: 'traceroot-ai/traceroot-ts',
      gitRef: 'integration-ref',
      instrumentModules: { piCodingAgent: pi },
    });
    assert.equal(TraceRoot.isInitialized(), true);

    const captured = attachInMemoryExporterToGlobalProvider();

    // Full prompt() -> agent_start -> (LLM turn) -> agent_end cycle, through the
    // REAL pi instrumentation initialize() just installed on Session.prototype.
    const session = new Session();
    const done = session.prompt('summarize the repository');
    session.emit({ type: 'agent_start' });
    session.emit({ type: 'turn_start' });
    session.emit({ type: 'message_start', message: assistantMessage('working on it') });
    session.emit({ type: 'message_end', message: assistantMessage('working on it') });
    session.emit({ type: 'turn_end', message: assistantMessage('working on it'), toolResults: [] });
    session.emit({
      type: 'agent_end',
      messages: [assistantMessage('the repository has three packages')],
      willRetry: false,
    });
    await done;

    const spans = captured.getFinishedSpans();
    const rootSpan = spans.find((s) => spanKind(s) === 'AGENT');
    const llmSpan = spans.find((s) => spanKind(s) === 'LLM');

    // The core proof: the real in-tree pi instrumentation actually produced
    // spans through TraceRoot's shared provider.
    assert.ok(
      rootSpan,
      'the real in-tree pi instrumentation must export an AGENT root span through TraceRoot shared provider',
    );

    // pi span shape survived the real round trip.
    assert.equal(rootSpan!.name, 'AgentSession.prompt');
    assert.equal(attrOf(rootSpan!)['session.id'], 'integration-sess');
    assert.equal(attrOf(rootSpan!)['input.value'], 'summarize the repository');
    assert.equal(attrOf(rootSpan!)['output.value'], 'the repository has three packages');
    // pi no longer self-stamps SDK identity on its root span; core's
    // TraceRootSpanProcessor.onStart owns traceroot.sdk.name uniformly across
    // every span (same as the Claude Agent SDK integration), so the shared
    // pipeline's 'traceroot-ts' stamp is the expected value here.
    assert.equal(attrOf(rootSpan!)['traceroot.sdk.name'], 'traceroot-ts');

    // TraceRootSpanProcessor enrichment was applied to pi's span on the way out
    // — this is what proves the spans travelled through TraceRoot's OWN pipeline,
    // not a private pi pipeline.
    assert.equal(attrOf(rootSpan!)['deployment.environment'], 'integration-test');
    assert.equal(attrOf(rootSpan!)['traceroot.git.repo'], 'traceroot-ai/traceroot-ts');
    assert.equal(attrOf(rootSpan!)['traceroot.git.ref'], 'integration-ref');
    assert.deepEqual(
      attrOf(rootSpan!)['traceroot.span.path'],
      ['AgentSession.prompt'],
      'the root span must carry TraceRootSpanProcessor span-path enrichment',
    );

    // The LLM child span routed through the same provider and nests under the
    // root, and its span path chains off the root's — proving whole-tree
    // enrichment, not just the root.
    assert.ok(llmSpan, 'the LLM child span must also route through TraceRoot provider');
    assert.equal(
      llmSpan!.parentSpanId,
      rootSpan!.spanContext().spanId,
      'the LLM span must nest under the AGENT root span',
    );
    assert.deepEqual(
      attrOf(llmSpan!)['traceroot.span.path'],
      ['AgentSession.prompt', 'claude-sonnet-5'],
      'the LLM span path must chain off the root span path via TraceRootSpanProcessor',
    );
  });
});

describe('session dispose', () => {
  /**
   * Verifies AgentSession.dispose()'s real, confirmed contract (see
   * instrumentation.ts's module header and types.ts's AgentSessionInstance
   * doc comment for the full verified call chain, read directly out of the
   * real, installed @earendil-works/pi-coding-agent@0.80.6 dist/core/
   * agent-session.js): dispose() clears every listener registered via
   * subscribe() — including instrumentPiCodingAgent()'s own — by reassigning
   * the session's internal listener array to a fresh empty one, with no
   * per-listener unsubscribe() call required from either side.
   *
   * FakeAgentSession here intentionally mirrors that exact mechanism (push on
   * subscribe, splice via the returned closure, reassign to [] on dispose) so
   * this test exercises the same shape the real SDK does, not a simplified
   * stand-in that would pass for the wrong reason.
   */

  // Registers a real, freshly-registered global TracerProvider wired to
  // `capture`, replacing the deleted private-exporter (`_spanExporter`)
  // injection path — see pi-test-helpers.ts's makeRig() for the full
  // isolation rationale behind calling trace.disable() first. This file keeps
  // its own local instrumentPiCodingAgent() calls (not pi-test-helpers.ts's
  // makeRig()) so it can drive session.dispose() directly on the raw session.
  function registerCapturingProvider(capture: CapturingExporter): void {
    trace.disable();
    const provider = new NodeTracerProvider();
    provider.addSpanProcessor(new SimpleSpanProcessor(capture));
    provider.register();
  }

  it('session.dispose() does not throw and requires no extra cleanup call from instrumentPiCodingAgent()', async () => {
    const capture = new CapturingExporter();
    const Session = makeFakeSessionClass();
    const sdk = { AgentSession: Session };
    registerCapturingProvider(capture);
    instrumentPiCodingAgent(sdk, {});
    const session = new Session();

    // Drive one full, cleanly-closed turn so instrumentPiCodingAgent()'s
    // subscribe() listener is actually registered and has produced a span,
    // matching real usage instead of disposing an untouched session.
    const done = session.prompt('do something');
    session.emit({ type: 'agent_start' });
    session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
    await done;
    assert.equal(capture.spans.length, 1, 'the normal run must still produce its span');

    assert.doesNotThrow(() => {
      session.dispose();
    }, 'dispose() must be safe to call even though instrumentPiCodingAgent() never captured or called the subscribe() unsubscribe function itself');
    assert.equal(session.disposed, true);

    // instrumentPiCodingAgent() never stored or invoked the unsubscribe
    // function subscribe() returned — it relies entirely on dispose() clearing
    // the SDK's own listener array. Firing more events post-dispose (as a
    // buggy or unusual host might) must produce no further spans, proving
    // instrumentation.ts needs no dispose-time hook of its own: the host
    // session's own dispose() is sufficient to stop delivery on its own.
    session.emit({ type: 'agent_start' });
    session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
    assert.equal(
      capture.spans.length,
      1,
      'no new spans may appear after dispose() — the listener must no longer be reachable',
    );
  });

  it('dispose() on a session that never had prompt() called (no traceroot-pi subscription registered yet) is still safe', async () => {
    const capture = new CapturingExporter();
    const Session = makeFakeSessionClass();
    const sdk = { AgentSession: Session };
    registerCapturingProvider(capture);
    instrumentPiCodingAgent(sdk, {});
    const session = new Session();

    assert.doesNotThrow(() => {
      session.dispose();
    });
    assert.equal(session.disposed, true);
    assert.equal(capture.spans.length, 0);
  });

  it('dispose() mid-run (before agent_end) force-closes and exports any still-open AGENT/LLM/TOOL spans instead of leaking them', async () => {
    const capture = new CapturingExporter();
    const Session = makeFakeSessionClass();
    const sdk = { AgentSession: Session };
    registerCapturingProvider(capture);
    instrumentPiCodingAgent(sdk, {});
    const session = new Session();

    // Open a run and leave it mid-flight: prompt() opens the AGENT (root)
    // span, message_start opens an LLM span, tool_execution_start opens a
    // TOOL span — and crucially agent_end never fires, exactly the "host
    // disposes while a run is in progress" scenario. Not awaited: prompt()'s
    // promise never settles on its own here (see pi-test-helpers.ts's module
    // header) since this run is deliberately abandoned mid-flight.
    session.prompt('long-running task');
    session.emit({ type: 'agent_start' });
    session.emit({
      type: 'message_start',
      message: assistantMessage({ model: 'mid-run-model' }),
    });
    session.emit({
      type: 'tool_execution_start',
      toolCallId: 'call-1',
      toolName: 'read_file',
      args: { path: '/tmp/x' },
    });

    // Nothing has exported yet — all three spans are still open.
    assert.equal(
      capture.spans.length,
      0,
      'no span should export before dispose() while still open',
    );

    assert.doesNotThrow(() => {
      session.dispose();
    }, 'dispose() must never throw even though it now force-closes in-flight spans');
    assert.equal(
      session.disposed,
      true,
      'the real dispose() must still run and mark the session disposed',
    );

    assert.equal(
      capture.spans.length,
      3,
      'the open AGENT root span, LLM span, and TOOL span must all be force-closed and exported by dispose()',
    );

    const rootSpan = capture.spans.find((s) => s.name === 'AgentSession.prompt');
    const llmSpan = capture.spans.find((s) => attrs(s)['gen_ai.request.model'] === 'mid-run-model');
    const toolSpan = capture.spans.find((s) => attrs(s)['gen_ai.tool.call.id'] === 'call-1');

    assert.ok(rootSpan, 'the AGENT root span must be exported');
    assert.ok(llmSpan, 'the LLM span must be exported');
    assert.ok(toolSpan, 'the TOOL span must be exported');

    assert.equal(
      attrs(rootSpan!)['traceroot.pi.force_closed'],
      true,
      'the root span must be marked force_closed, distinguishing it from a normal agent_end close',
    );
    assert.equal(
      attrs(llmSpan!)['traceroot.pi.force_closed'],
      true,
      'the LLM span must be marked force_closed',
    );
    assert.equal(
      attrs(toolSpan!)['traceroot.pi.force_closed'],
      true,
      'the TOOL span must be marked force_closed',
    );

    // Firing more events post-dispose must produce no further spans — the
    // real dispose() already cleared _eventListeners, and our own patch must
    // not have reintroduced a way to reach the (now torn-down) state.
    session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
    assert.equal(
      capture.spans.length,
      3,
      'no further spans may appear after dispose() already force-closed everything',
    );
  });

  it('a session reused after dispose() re-subscribes and resumes tracing on its next prompt()', async () => {
    // Nothing in the real SDK prevents calling prompt()/steer()/followUp()
    // again on a session instance after dispose() — dispose() only clears the
    // SDK's own _eventListeners array (see instrumentation.ts's module
    // header), it does not make the session instance itself unusable. Before
    // the fix, instrumentPiCodingAgent()'s own `subscribedSessions` WeakSet
    // permanently remembered this session as "already subscribed" and never
    // re-attached its span listener on the next prompt() call, so every run
    // after the first dispose() silently produced zero spans.
    const capture = new CapturingExporter();
    const Session = makeFakeSessionClass();
    const sdk = { AgentSession: Session };
    registerCapturingProvider(capture);
    instrumentPiCodingAgent(sdk, {});
    const session = new Session();

    // First run: clean prompt/agent_start/agent_end cycle.
    const done1 = session.prompt('first run');
    session.emit({ type: 'agent_start' });
    session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
    await done1;
    assert.equal(capture.spans.length, 1, 'the first run must export its span');

    session.dispose();
    assert.equal(session.disposed, true);

    // Second run on the SAME session instance, after dispose(). A real host
    // reusing a session (or a test harness that calls dispose() defensively
    // between runs) must still get tracing on this next run.
    const done2 = session.prompt('second run, after dispose()');
    session.emit({ type: 'agent_start' });
    session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
    await done2;

    assert.equal(
      capture.spans.length,
      2,
      'the second run (after dispose() and reuse) must ALSO export its span, not be silently ' +
        'dropped because instrumentPiCodingAgent() thinks this session is still subscribed',
    );
  });

  it('dispose() force-closes the OTHER open spans even when one span throws while being force-closed', async () => {
    // spans.ts's closeDanglingSpan() guards its own setAttr(FORCE_CLOSED) call
    // in a try/catch (warn-and-continue) precisely so a throwing
    // span.setAttribute() can never prevent endSpanSafe()/span.end() from
    // running for THAT span — let alone abort the rest of dispose()'s sweep.
    // If one open span's setAttribute() throws — a misbehaving Span
    // implementation, or a bug triggered by that span's own attribute values —
    // it still gets force-closed and exported (just without the force_closed
    // marker, since the attribute write itself failed), and the other
    // still-open spans must likewise still be force-closed and exported.
    const capture = new CapturingExporter();
    const Session = makeFakeSessionClass();
    const sdk = { AgentSession: Session };
    registerCapturingProvider(capture);
    instrumentPiCodingAgent(sdk, {});
    const session = new Session();

    session.prompt('multi-tool run');
    session.emit({ type: 'agent_start' });
    session.emit({
      type: 'message_start',
      message: assistantMessage({ model: 'sweep-test-model' }),
    });
    // Three open tool spans left dangling (call-1, call-2, call-3), plus the
    // still-open LLM span and root span — 5 spans total, none of which have
    // had agent_end/message_end/tool_execution_end fire for them.
    session.emit({
      type: 'tool_execution_start',
      toolCallId: 'call-1',
      toolName: 'read_file',
      args: { path: '/tmp/a' },
    });
    session.emit({
      type: 'tool_execution_start',
      toolCallId: 'call-2',
      toolName: 'read_file',
      args: { path: '/tmp/b' },
    });
    session.emit({
      type: 'tool_execution_start',
      toolCallId: 'call-3',
      toolName: 'read_file',
      args: { path: '/tmp/c' },
    });
    assert.equal(
      capture.spans.length,
      0,
      'no span should export before dispose() while still open',
    );

    // Poison exactly the call-2 tool span: throw the first time dispose()'s
    // sweep tries to mark it force_closed, simulating a single misbehaving
    // span mid-sweep. Identified by its own gen_ai.tool.call.id attribute
    // (already set at tool_execution_start time), not by call order, so this
    // stays correct regardless of Map iteration order.
    type SetAttributeFn = typeof Span.prototype.setAttribute;
    const originalSetAttribute: SetAttributeFn = Span.prototype.setAttribute;
    Span.prototype.setAttribute = function (
      this: Span,
      key: string,
      value?: Parameters<SetAttributeFn>[1],
    ): Span {
      if (
        key === 'traceroot.pi.force_closed' &&
        this.attributes['gen_ai.tool.call.id'] === 'call-2'
      ) {
        throw new Error('injected span failure for call-2');
      }
      return originalSetAttribute.call(this, key, value);
    } as SetAttributeFn;

    const warnings: string[] = [];
    const originalWarn = console.warn;
    console.warn = (...args: unknown[]): void => {
      warnings.push(args.map((a) => String(a)).join(' '));
    };

    try {
      assert.doesNotThrow(() => {
        session.dispose();
      }, 'dispose() must not throw even though force-closing one span (call-2) failed internally');
    } finally {
      Span.prototype.setAttribute = originalSetAttribute;
      console.warn = originalWarn;
    }
    assert.equal(session.disposed, true);

    // call-2's setAttribute(FORCE_CLOSED) throw is caught INSIDE
    // closeDanglingSpan() (spans.ts) — endSpanSafe() still runs unconditionally
    // afterward, so call-2 is force-closed and exported too, just without the
    // force_closed marker (the attribute write itself failed). All 5 spans
    // (root, LLM, call-1, call-2, call-3) must still be force-closed and
    // exported despite the injected failure.
    assert.equal(
      capture.spans.length,
      5,
      'all 5 spans (root, LLM, call-1, call-2, call-3) must still export — a setAttribute failure ' +
        'on the force_closed marker must never prevent the span itself from being ended',
    );
    const exportedToolCallIds = capture.spans
      .map((s) => (s.attributes as Record<string, unknown>)['gen_ai.tool.call.id'])
      .filter((id): id is string => typeof id === 'string')
      .sort();
    assert.deepEqual(
      exportedToolCallIds,
      ['call-1', 'call-2', 'call-3'],
      'call-1, call-2, and call-3 must all be exported; call-2 (poisoned) still ends normally, ' +
        'it just never got its force_closed attribute set',
    );
    const rootSpan = capture.spans.find((s) => s.name === 'AgentSession.prompt');
    const llmSpan = capture.spans.find(
      (s) =>
        (s.attributes as Record<string, unknown>)['gen_ai.request.model'] === 'sweep-test-model',
    );
    const call2Span = capture.spans.find(
      (s) => (s.attributes as Record<string, unknown>)['gen_ai.tool.call.id'] === 'call-2',
    );
    assert.ok(rootSpan, 'the root span must still be exported despite call-2 throwing mid-sweep');
    assert.ok(llmSpan, 'the LLM span must still be exported despite call-2 throwing mid-sweep');
    assert.ok(call2Span, 'call-2 must still be exported despite its own setAttribute failure');
    assert.equal(
      (call2Span!.attributes as Record<string, unknown>)['traceroot.pi.force_closed'],
      undefined,
      'call-2 never got its force_closed attribute set, since setAttribute threw on that call',
    );
    assert.ok(
      warnings.some((w) => w.includes('failed to mark a dangling span force_closed')),
      'the injected setAttribute failure on call-2 must still be surfaced via console.warn',
    );
  });

  // Lens: a host listener that disposes the session synchronously while
  // handling agent_end races pi's own agent_end handler.
  //
  // If a host registers its own session.subscribe() listener BEFORE pi does
  // (i.e. before its first prompt()) and that listener calls session.dispose()
  // synchronously on agent_end, dispose()'s sweep force-closes this run's root
  // span before pi's own agent_end handler runs for the SAME event. Because
  // dispose() reassigns the listener array rather than mutating it mid-dispatch,
  // pi's handler still fires afterward — but finds state.rootSpan already gone.
  // Pi must DETECT that its root span was force-closed out from under it and
  // surface it, rather than silently skipping the real close and quietly
  // producing an incomplete trace.
  it('agent_end after a reentrant dispose() force-closed the root span is surfaced, not silently skipped', async () => {
    const capture = new CapturingExporter();
    const Session = makeFakeSessionClass();
    const sdk = { AgentSession: Session };
    registerCapturingProvider(capture);
    instrumentPiCodingAgent(sdk, {});
    const session = new Session();

    // Host listener registered BEFORE pi's own subscribe() (which happens on the
    // first prompt() below), so it sits ahead of pi's handler in the dispatch
    // order. It disposes the session the instant it sees agent_end — the exact
    // reentrant race. dispose() reassigns the listener array, so pi's own
    // agent_end handler still runs in this same emit loop, just after dispose()
    // already tore the run's root span down.
    session.subscribe((event: AgentEvent) => {
      if (event.type === 'agent_end') session.dispose();
    });

    const warnings: string[] = [];
    const originalWarn = console.warn;
    console.warn = (...args: unknown[]): void => {
      warnings.push(args.map((a) => String(a)).join(' '));
    };
    try {
      const done = session.prompt('do the work'); // pi subscribes here, after the host
      session.emit({ type: 'agent_start' });
      session.emit({
        type: 'agent_end',
        messages: [assistantMessage({ content: [{ type: 'text', text: 'the final answer' }] })],
        willRetry: false,
      });
      // The reentrant dispose() already force-closed the root before pi's own
      // agent_end handler ran (see below) — its identity guard means
      // proto.prompt's own finalize() becomes a no-op once this settles, so
      // awaiting it is still safe (it never re-ends the already-closed span).
      await done;
    } finally {
      console.warn = originalWarn;
    }

    // dispose()'s sweep force-closed the root span exactly once, so it still
    // exports — but as a FORCE_CLOSED span lacking the normal agent_end output.
    assert.equal(
      capture.spans.length,
      1,
      'the root span is force-closed exactly once by dispose()',
    );
    const rootSpan = capture.spans[0];
    assert.equal(attrs(rootSpan)['openinference.span.kind'], 'AGENT');
    assert.equal(
      attrs(rootSpan)['traceroot.pi.force_closed'],
      true,
      'dispose() force-closed the root span, so it is marked force_closed',
    );
    assert.equal(
      attrs(rootSpan)['output.value'],
      undefined,
      "the run's completion output never made it onto the force-closed root span",
    );

    // Because dispose() beat pi's own close, the completion output could not be
    // recorded. Pi must SURFACE that the root span was force-closed by a
    // reentrant dispose() rather than silently doing nothing.
    assert.ok(
      warnings.some((w) => /reentrant dispose/i.test(w) && /agent_end/i.test(w)),
      'pi must warn that agent_end arrived after a reentrant dispose() force-closed the root span',
    );
  });
});

describe('steer / followUp', () => {
  /**
   * instrumentPiCodingAgent() must patch AgentSession.prototype.steer and
   * .followUp, not just .prompt — steer()/followUp() are standalone public
   * SDK entry points (see @earendil-works/pi-coding-agent's
   * dist/core/agent-session.d.ts:359/:367) a host can call directly without
   * ever calling prompt() on the session first. Before this fix, a host whose
   * first interaction with a session was steer()/followUp() got zero tracing:
   * subscribe() was never called (no listener attached), so every subsequent
   * AgentEvent — agent_start through agent_end — silently produced no spans
   * at all, for the entire lifetime of that session.
   *
   * Deliberately NOT asserting that steer()/followUp() text becomes a root
   * span's input.value: verified against the real, installed
   * @earendil-works/pi-agent-core@0.80.6 (dist/agent.js:169-176), Agent.steer()
   * / Agent.followUp() only ever enqueue into an internal queue — neither one
   * ever itself triggers a fresh run (only Agent.prompt()/continue() do, via
   * runPromptMessages()). Queuing their text into the prompt-only pendingInput
   * FIFO (consumed exclusively by agent_start) would misattribute it to
   * whatever LATER, unrelated run's agent_start happens to fire next — a
   * worse bug than the one being fixed here. Attaching the listener is the
   * correct, minimal fix for the actual "zero tracing" defect.
   */

  // Registers a real, freshly-registered global TracerProvider wired to
  // `capture`, replacing the deleted private-exporter (`_spanExporter`)
  // injection path — see pi-test-helpers.ts's makeRig() for the full
  // isolation rationale behind calling trace.disable() first.
  function registerCapturingProvider(capture: CapturingExporter): void {
    trace.disable();
    const provider = new NodeTracerProvider();
    provider.addSpanProcessor(new SimpleSpanProcessor(capture));
    provider.register();
  }

  // Extends the shared FakeAgentSession with the standalone steer()/followUp()
  // entry points the base fixture deliberately omits (see test-helpers.ts, and
  // provider-shared-mode-behavior.test.ts's own makeSteerableSessionClass, which
  // adds steer() the same way for the same reason), so a host whose first
  // interaction is steer()/followUp() can be exercised here. Fresh per call,
  // like makeFakeSessionClass itself, so prototype patches never stack across
  // tests.
  function makeSteerAndFollowUpSessionClass() {
    const Base = makeFakeSessionClass();
    return class SteerAndFollowUpAgentSession extends Base {
      async steer(_text: string, _images?: unknown[]): Promise<void> {}
      async followUp(_text: string, _images?: unknown[]): Promise<void> {}
    };
  }

  it('calling steer() as the FIRST interaction (no prior prompt() call) still attaches tracing — its run is ROOTLESS (bypasses prompt()), but its child spans still export', async () => {
    // FLIPPED for the new model: only the wrapped prompt() call opens a root
    // span (see instrumentation.ts's module header on the rootless-bypass
    // boundary policy) — steer()/followUp() never did, and agent_start no
    // longer synthesizes one either. Before this change, agent_start
    // unconditionally opened a root regardless of what triggered it, so this
    // test could assert on an exported AGENT span. Under the new model, a run
    // whose ONLY interaction was steer() (no enclosing prompt() call) produces
    // NO root span at all — but the underlying bug this test guards against
    // ("steer() as first interaction gets zero tracing at all") is still real,
    // so it is rephrased to prove the LISTENER is attached: a tool span fired
    // during this bypass run still exports (as a parentless mini-trace),
    // which could only happen if session.subscribe() had actually been called.
    const capture = new CapturingExporter();
    const Session = makeSteerAndFollowUpSessionClass();
    const sdk = { AgentSession: Session };
    registerCapturingProvider(capture);
    instrumentPiCodingAgent(sdk, {});
    const session = new Session();

    // No session.prompt(...) call anywhere — steer() is the only entry point
    // this host ever uses on this session.
    await session.steer('do X instead');
    session.emit({ type: 'agent_start' });
    session.emit({
      type: 'tool_execution_start',
      toolCallId: 'bypass-tool',
      toolName: 'bash',
      args: {},
    });
    session.emit({
      type: 'tool_execution_end',
      toolCallId: 'bypass-tool',
      toolName: 'bash',
      result: {},
      isError: false,
    });
    session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });

    assert.equal(
      capture.spans.find((s) => attrs(s)['openinference.span.kind'] === 'AGENT'),
      undefined,
      'a run that bypasses prompt() entirely must never synthesize a root AGENT span',
    );
    const toolSpan = capture.spans.find((s) => attrs(s)['gen_ai.tool.call.id'] === 'bypass-tool');
    assert.ok(
      toolSpan,
      'steer() must still attach the span listener itself — otherwise this bypass run’s tool span ' +
        'would never have been captured at all, proving prompt() was not required first',
    );
    assert.equal(
      toolSpan!.parentSpanId,
      undefined,
      'with no root open, the bypass run’s tool span parents under ROOT_CONTEXT (a fresh, ' +
        'standalone parentless mini-trace)',
    );
  });

  it('calling followUp() as the FIRST interaction (no prior prompt() call) still attaches tracing — its run is ROOTLESS (bypasses prompt()), but its child spans still export', async () => {
    const capture = new CapturingExporter();
    const Session = makeSteerAndFollowUpSessionClass();
    const sdk = { AgentSession: Session };
    registerCapturingProvider(capture);
    instrumentPiCodingAgent(sdk, {});
    const session = new Session();

    await session.followUp('also check Y');
    session.emit({ type: 'agent_start' });
    session.emit({
      type: 'tool_execution_start',
      toolCallId: 'bypass-tool',
      toolName: 'bash',
      args: {},
    });
    session.emit({
      type: 'tool_execution_end',
      toolCallId: 'bypass-tool',
      toolName: 'bash',
      result: {},
      isError: false,
    });
    session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });

    assert.equal(
      capture.spans.find((s) => attrs(s)['openinference.span.kind'] === 'AGENT'),
      undefined,
      'a run that bypasses prompt() entirely must never synthesize a root AGENT span',
    );
    const toolSpan = capture.spans.find((s) => attrs(s)['gen_ai.tool.call.id'] === 'bypass-tool');
    assert.ok(
      toolSpan,
      'followUp() must still attach the span listener itself, not rely on prompt() having been ' +
        'called first',
    );
  });

  it('a session already subscribed via prompt() does not get double-subscribed when steer()/followUp() are called later', async () => {
    const capture = new CapturingExporter();
    const Session = makeSteerAndFollowUpSessionClass();
    const sdk = { AgentSession: Session };
    registerCapturingProvider(capture);
    instrumentPiCodingAgent(sdk, {});
    const session = new Session();

    // prompt() is not awaited immediately (its promise settles only once its
    // final agent_end fires — see pi-test-helpers.ts's module header); steer()
    // and followUp() are independent, immediately-resolving calls that must
    // reuse the SAME listener prompt() already attached, not subscribe again.
    const done = session.prompt('start the run');
    await session.steer('steer mid-run');
    await session.followUp('follow up after');
    session.emit({ type: 'agent_start' });
    session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
    await done;

    assert.equal(
      capture.spans.length,
      1,
      'exactly one root span — proves steer()/followUp() reused the same listener prompt() already attached, instead of subscribing a second time',
    );
  });
});
