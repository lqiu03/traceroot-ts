import assert from 'node:assert/strict';
import { afterEach, describe, it } from 'node:test';
import { trace } from '@opentelemetry/api';
import type { ReadableSpan } from '@opentelemetry/sdk-trace-base';
import { InMemorySpanExporter, Span, SimpleSpanProcessor } from '@opentelemetry/sdk-trace-base';
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import { instrumentPiCodingAgent, type AgentEvent, type AssistantMessage } from '../src/pi';
import { TraceRoot, _resetForTesting } from '../src/traceroot';
import {
  assistantMessage,
  assistantMessage as baseAssistantMessage,
  attrs,
  CapturingExporter,
  makeFakeSessionClass,
  makeRig,
} from './pi-test-helpers';

describe('pi instrumentation', () => {
  // Overrides the shared helper's placeholder usage since this file asserts on specific numbers.
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

    // NOT awaited yet; see pi-test-helpers CONTRACT.
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
    // traceroot.sdk.name is owned by TraceRootSpanProcessor, which this rig doesn't wire.
    assert.equal(attrs(rootSpan)['traceroot.sdk.name'], undefined);
    assert.equal(attrs(rootSpan)['input.value'], 'list files in /tmp');
    assert.equal(attrs(rootSpan)['output.value'], 'listed the files');
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
    // A path arg, not a bash command, so the name reduces to its basename.
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
      // Non-string bypasses the `typeof text === 'string'` guard (prompt text genuinely absent).
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

describe('pi wiring', () => {
  // Wiring-only coverage for instrumentModules.piCodingAgent; deep span behavior is covered elsewhere.

  // Fresh per call, since the wrap-once guard is stamped on the prototype itself.
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
      // No prompt/subscribe on the prototype; warns rather than throwing (unlike claude-agent-sdk's wiring).
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

describe('pi integration', () => {
  // Drives the real pi instrumentation through TraceRoot.initialize() with no mocked pi export.
  interface FakeAgentEvent {
    type: string;
    [key: string]: unknown;
  }
  function makePiModule() {
    class FakeAgentSession {
      sessionId = 'integration-sess';
      private listeners: Array<(event: FakeAgentEvent) => void> = [];
      // Settles only on final agent_end (willRetry !== true); see pi-test-helpers.ts's CONTRACT.
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

  // Minimal valid AssistantMessage, enough for pi's span builders to read.
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

  // Reaches the provider TraceRoot.initialize() registered as the OTel global delegate.
  function attachInMemoryExporterToGlobalProvider(): InMemorySpanExporter {
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
    delete process.env.TRACEROOT_API_KEY;

    const pi = makePiModule();
    const Session = pi.AgentSession;

    TraceRoot.initialize({
      apiKey: 'trk_integration_key',
      // Local, unroutable: must never POST to the real backend.
      baseUrl: 'http://127.0.0.1:9',
      disableBatch: true,
      environment: 'integration-test',
      gitRepo: 'traceroot-ai/traceroot-ts',
      gitRef: 'integration-ref',
      instrumentModules: { piCodingAgent: pi },
    });
    assert.equal(TraceRoot.isInitialized(), true);

    const captured = attachInMemoryExporterToGlobalProvider();

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

    assert.ok(
      rootSpan,
      'the real in-tree pi instrumentation must export an AGENT root span through TraceRoot shared provider',
    );

    assert.equal(rootSpan!.name, 'AgentSession.prompt');
    assert.equal(attrOf(rootSpan!)['session.id'], 'integration-sess');
    assert.equal(attrOf(rootSpan!)['input.value'], 'summarize the repository');
    assert.equal(attrOf(rootSpan!)['output.value'], 'the repository has three packages');
    // traceroot.sdk.name is owned by TraceRootSpanProcessor.onStart, uniformly across every span.
    assert.equal(attrOf(rootSpan!)['traceroot.sdk.name'], 'traceroot-ts');

    assert.equal(attrOf(rootSpan!)['deployment.environment'], 'integration-test');
    assert.equal(attrOf(rootSpan!)['traceroot.git.repo'], 'traceroot-ai/traceroot-ts');
    assert.equal(attrOf(rootSpan!)['traceroot.git.ref'], 'integration-ref');
    assert.deepEqual(
      attrOf(rootSpan!)['traceroot.span.path'],
      ['AgentSession.prompt'],
      'the root span must carry TraceRootSpanProcessor span-path enrichment',
    );

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

describe('pi session dispose', () => {
  // dispose() clears every listener via subscribe(), reassigning the internal array to a fresh empty one.

  // Local instrumentPiCodingAgent() calls (not makeRig()) so tests can drive dispose() on the raw session.
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

    const done = session.prompt('do something');
    session.emit({ type: 'agent_start' });
    session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
    await done;
    assert.equal(capture.spans.length, 1, 'the normal run must still produce its span');

    assert.doesNotThrow(() => {
      session.dispose();
    }, 'dispose() must be safe to call even though instrumentPiCodingAgent() never captured or called the subscribe() unsubscribe function itself');
    assert.equal(session.disposed, true);

    // instrumentPiCodingAgent() relies on dispose() clearing the SDK's own listener array, not its own hook.
    session.emit({ type: 'agent_start' });
    session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
    assert.equal(
      capture.spans.length,
      1,
      'no new spans may appear after dispose() -- the listener must no longer be reachable',
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

    // "Host disposes mid-run": open spans, never fire agent_end. Not awaited: abandoned mid-flight.
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

    session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
    assert.equal(
      capture.spans.length,
      3,
      'no further spans may appear after dispose() already force-closed everything',
    );
  });

  // Guards a real bug: subscribedSessions used to permanently remember a session as subscribed
  // and never re-attach its listener after dispose(), so every run after the first dispose() produced zero spans.
  it('a session reused after dispose() re-subscribes and resumes tracing on its next prompt()', async () => {
    const capture = new CapturingExporter();
    const Session = makeFakeSessionClass();
    const sdk = { AgentSession: Session };
    registerCapturingProvider(capture);
    instrumentPiCodingAgent(sdk, {});
    const session = new Session();

    const done1 = session.prompt('first run');
    session.emit({ type: 'agent_start' });
    session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
    await done1;
    assert.equal(capture.spans.length, 1, 'the first run must export its span');

    session.dispose();
    assert.equal(session.disposed, true);

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

  // closeDanglingSpan() guards its setAttribute(FORCE_CLOSED) call in a try/catch (warn-and-continue)
  // so a throwing setAttribute() never prevents that span's own end() or the rest of dispose()'s sweep.
  it('dispose() force-closes the OTHER open spans even when one span throws while being force-closed', async () => {
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
    // 5 spans total left dangling: root, LLM, and call-1/2/3.
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

    // Poison call-2 (by tool.call.id, not call order): throw on its first force_closed mark.
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

    assert.equal(
      capture.spans.length,
      5,
      'all 5 spans (root, LLM, call-1, call-2, call-3) must still export -- a setAttribute failure ' +
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

  // Guards a reentrancy race: a host listener registered before pi's own that calls dispose()
  // synchronously on agent_end force-closes the root before pi's own agent_end handler runs.
  it('agent_end after a reentrant dispose() force-closed the root still exports it exactly once', async () => {
    const capture = new CapturingExporter();
    const Session = makeFakeSessionClass();
    const sdk = { AgentSession: Session };
    registerCapturingProvider(capture);
    instrumentPiCodingAgent(sdk, {});
    const session = new Session();

    // Registered before pi's own subscribe() (below), so it sits ahead of pi's handler.
    session.subscribe((event: AgentEvent) => {
      if (event.type === 'agent_end') session.dispose();
    });

    const done = session.prompt('do the work'); // pi subscribes here, after the host
    session.emit({ type: 'agent_start' });
    session.emit({
      type: 'agent_end',
      messages: [assistantMessage({ content: [{ type: 'text', text: 'the final answer' }] })],
      willRetry: false,
    });
    await done;

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
  });
});

describe('pi steer / followUp', () => {
  // Must patch .steer/.followUp too, not just .prompt: real bug was a host whose first interaction
  // was steer()/followUp() got zero tracing. Their text is deliberately never asserted onto the root's
  // input.value: they only enqueue, never trigger a run, so attributing it would misattribute to a later run.

  function registerCapturingProvider(capture: CapturingExporter): void {
    trace.disable();
    const provider = new NodeTracerProvider();
    provider.addSpanProcessor(new SimpleSpanProcessor(capture));
    provider.register();
  }

  // Extends the shared FakeAgentSession with steer()/followUp(). Fresh per call so patches don't stack.
  function makeSteerAndFollowUpSessionClass() {
    const Base = makeFakeSessionClass();
    return class SteerAndFollowUpAgentSession extends Base {
      async steer(_text: string, _images?: unknown[]): Promise<void> {}
      async followUp(_text: string, _images?: unknown[]): Promise<void> {}
    };
  }

  // Only prompt() opens a root span; a steer()-only run produces none, so this checks its tool span
  // still exports (as a parentless mini-trace) to prove the listener is attached, not zero-tracing.
  it('calling steer() as the FIRST interaction (no prior prompt() call) still attaches tracing -- its run is ROOTLESS (bypasses prompt()), but its child spans still export', async () => {
    const capture = new CapturingExporter();
    const Session = makeSteerAndFollowUpSessionClass();
    const sdk = { AgentSession: Session };
    registerCapturingProvider(capture);
    instrumentPiCodingAgent(sdk, {});
    const session = new Session();

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
      "steer() must still attach the span listener itself -- otherwise this bypass run's tool span " +
        'would never have been captured at all, proving prompt() was not required first',
    );
    assert.equal(
      toolSpan!.parentSpanId,
      undefined,
      "with no root open, the bypass run's tool span parents under ROOT_CONTEXT (a fresh, " +
        'standalone parentless mini-trace)',
    );
  });

  it('calling followUp() as the FIRST interaction (no prior prompt() call) still attaches tracing -- its run is ROOTLESS (bypasses prompt()), but its child spans still export', async () => {
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

    const done = session.prompt('start the run');
    await session.steer('steer mid-run');
    await session.followUp('follow up after');
    session.emit({ type: 'agent_start' });
    session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
    await done;

    assert.equal(
      capture.spans.length,
      1,
      'exactly one root span -- proves steer()/followUp() reused the same listener prompt() already attached, instead of subscribing a second time',
    );
  });
});
