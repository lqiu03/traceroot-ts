import { afterEach, beforeEach, describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { context, propagation, trace } from '@opentelemetry/api';
import { InMemorySpanExporter, SimpleSpanProcessor } from '@opentelemetry/sdk-trace-base';
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import { wireClaudeAgentSDKInstrumentation } from '../src/claude-agent-sdk';

type HookInput = Record<string, unknown>;
type HookCallback = (input: HookInput, toolUseId?: string) => Promise<Record<string, unknown>>;
type QueryParams = {
  prompt?: string;
  options?: {
    model?: string;
    hooks?: Record<string, Array<{ hooks: HookCallback[] }>>;
  };
};

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function runHooks(params: QueryParams, event: string, input: HookInput, toolUseId: string) {
  const matchers = params.options?.hooks?.[event] ?? [];
  for (const matcher of matchers) {
    for (const hook of matcher.hooks) {
      await hook({ ...input, hook_event_name: event, tool_use_id: toolUseId }, toolUseId);
    }
  }
}

// A span processor that counts start/end events per span name so we can detect
// leaked spans (started but never ended -> never exported).
class CountingSpanProcessor {
  readonly started = new Map<string, number>();
  readonly ended = new Map<string, number>();
  onStart(span: { name: string }): void {
    this.started.set(span.name, (this.started.get(span.name) ?? 0) + 1);
  }
  onEnd(span: { name: string }): void {
    this.ended.set(span.name, (this.ended.get(span.name) ?? 0) + 1);
  }
  shutdown(): Promise<void> {
    return Promise.resolve();
  }
  forceFlush(): Promise<void> {
    return Promise.resolve();
  }
  leaked(name: string): number {
    return (this.started.get(name) ?? 0) - (this.ended.get(name) ?? 0);
  }
}

const LLM_SPAN = 'anthropic.messages.create';
const QUERY_SPAN = 'ClaudeAgent.query';

describe('Claude Agent SDK instrumentation (opus review pass)', () => {
  let exporter: InMemorySpanExporter;
  let provider: NodeTracerProvider;
  let counter: CountingSpanProcessor;

  beforeEach(() => {
    exporter = new InMemorySpanExporter();
    counter = new CountingSpanProcessor();
    provider = new NodeTracerProvider();
    provider.addSpanProcessor(new SimpleSpanProcessor(exporter));
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    provider.addSpanProcessor(counter as any);
    provider.register();
  });

  afterEach(async () => {
    await provider.shutdown();
    exporter.reset();
    trace.disable();
    context.disable();
    propagation.disable();
  });

  it('closes the query span (no leak) when a for-await loop breaks after partial consumption', async () => {
    const sdk = {
      async *query(params: QueryParams) {
        yield {
          type: 'assistant',
          message: {
            id: 'msg_1',
            role: 'assistant',
            model: 'claude-opus-4-7',
            usage: { input_tokens: 10, output_tokens: 1 },
            content: [{ type: 'tool_use', id: 'toolu_1', name: 'WebSearch', input: {} }],
          },
        };
        await runHooks(params, 'PreToolUse', { tool_name: 'WebSearch', tool_input: {} }, 'toolu_1');
        // The consumer breaks before these ever arrive.
        await sleep(50);
        yield { type: 'result', result: 'never reached', usage: { output_tokens: 1 } };
      },
    };

    wireClaudeAgentSDKInstrumentation(sdk);

    let seen = 0;
    for await (const _message of sdk.query({
      prompt: 'p',
      options: { model: 'claude-opus-4-7' },
    })) {
      seen += 1;
      break; // abandon after first message -> for-await calls iterator.return()
    }

    assert.equal(seen, 1);
    const query = exporter.getFinishedSpans().find((s) => s.name === QUERY_SPAN);
    assert.ok(query, 'query span should be ended on break, not leaked');
    assert.equal(counter.leaked(QUERY_SPAN), 0, 'query span leaked after break');
    // The in-flight WebSearch tool span (PreToolUse fired, no PostToolUse) must be swept.
    assert.equal(counter.leaked('WebSearch'), 0, 'in-flight tool span leaked after break');
  });

  it('leaves the query span open when a manual iterator is abandoned without return() (characterization)', async () => {
    const sdk = {
      async *query(_params: QueryParams) {
        yield {
          type: 'assistant',
          message: {
            id: 'msg_1',
            role: 'assistant',
            model: 'claude-opus-4-7',
            usage: { input_tokens: 10, output_tokens: 1 },
            content: [{ type: 'text', text: 'hi' }],
          },
        };
        yield { type: 'result', result: 'done', usage: { output_tokens: 1 } };
      },
    };
    wireClaudeAgentSDKInstrumentation(sdk);

    const asyncIterable = sdk.query({ prompt: 'p', options: { model: 'claude-opus-4-7' } });
    const iterator = asyncIterable[Symbol.asyncIterator]();
    await iterator.next();
    // Drop the iterator without draining or calling return(). This is inherent
    // to the async-iterator protocol: nothing can close the span. Documented here
    // so a regression that DOES leak in the drained/returned paths is caught by the
    // other tests, while this genuinely-unavoidable case is expected.
    assert.equal(
      counter.leaked(QUERY_SPAN),
      1,
      'abandoned-without-return leaves span open (expected)',
    );
  });

  it('does not leak the query/LLM spans when the underlying generator throws mid-stream', async () => {
    const sdk = {
      async *query(_params: QueryParams) {
        yield {
          type: 'assistant',
          message: {
            id: 'msg_1',
            role: 'assistant',
            model: 'claude-opus-4-7',
            usage: { input_tokens: 10, output_tokens: 5 },
            content: [{ type: 'text', text: 'partial' }],
          },
        };
        throw new Error('stream exploded');
      },
    };
    wireClaudeAgentSDKInstrumentation(sdk);

    await assert.rejects(async () => {
      for await (const _m of sdk.query({ prompt: 'p', options: { model: 'claude-opus-4-7' } })) {
        // consume
      }
    }, /stream exploded/);

    const query = exporter.getFinishedSpans().find((s) => s.name === QUERY_SPAN)!;
    assert.ok(query, 'query span missing');
    assert.equal(query.status.code, 2, 'query span should be ERROR on mid-stream throw');
    assert.equal(counter.leaked(QUERY_SPAN), 0, 'query span leaked on mid-stream throw');
    assert.equal(counter.leaked(LLM_SPAN), 0, 'LLM span leaked on mid-stream throw');
  });

  it('does not forward interrupt()/setPermissionMode() (documented limitation)', async () => {
    const sdk = {
      async *query(_params: QueryParams) {
        yield { type: 'result', result: 'done', usage: { output_tokens: 1 } };
      },
    };
    // Attach the streaming-input methods the real SDK Query object would expose.
    (sdk.query as unknown as Record<string, unknown>).interrupt = () => {};
    wireClaudeAgentSDKInstrumentation(sdk);

    const handle = sdk.query({ prompt: 'p', options: { model: 'claude-opus-4-7' } }) as unknown as {
      interrupt?: unknown;
      setPermissionMode?: unknown;
    };
    assert.equal(typeof handle.interrupt, 'undefined', 'interrupt is not forwarded (by design)');
    assert.equal(
      typeof handle.setPermissionMode,
      'undefined',
      'setPermissionMode is not forwarded',
    );
  });

  it('keeps two concurrently-iterated queries as independent, correctly-parented trees', async () => {
    const sdk = {
      async *query(_params: QueryParams) {
        yield {
          type: 'assistant',
          message: {
            id: 'msg',
            role: 'assistant',
            model: 'claude-opus-4-7',
            usage: { input_tokens: 7, output_tokens: 3 },
            content: [{ type: 'text', text: 'hi' }],
          },
        };
        await sleep(5);
        yield { type: 'result', result: 'done', usage: { output_tokens: 3 } };
      },
    };
    wireClaudeAgentSDKInstrumentation(sdk);

    const tracer = trace.getTracer('test');
    const rootA = tracer.startSpan('rootA');
    const rootB = tracer.startSpan('rootB');

    const itA = await context.with(trace.setSpan(context.active(), rootA), () =>
      sdk.query({ prompt: 'A', options: { model: 'claude-opus-4-7' } })[Symbol.asyncIterator](),
    );
    const itB = await context.with(trace.setSpan(context.active(), rootB), () =>
      sdk.query({ prompt: 'B', options: { model: 'claude-opus-4-7' } })[Symbol.asyncIterator](),
    );

    // Interleave the two streams.
    await itA.next();
    await itB.next();
    await itA.next();
    await itB.next();
    await itA.next();
    await itB.next();
    rootA.end();
    rootB.end();

    const spans = exporter.getFinishedSpans();
    const querySpans = spans.filter((s) => s.name === QUERY_SPAN);
    assert.equal(querySpans.length, 2, 'each concurrent query should get its own span');
    const qA = querySpans.find((s) => s.parentSpanId === rootA.spanContext().spanId)!;
    const qB = querySpans.find((s) => s.parentSpanId === rootB.spanContext().spanId)!;
    assert.ok(qA, 'query A not parented under rootA');
    assert.ok(qB, 'query B not parented under rootB');
    assert.notEqual(qA.spanContext().spanId, qB.spanContext().spanId);

    const llmSpans = spans.filter((s) => s.name === LLM_SPAN);
    assert.equal(llmSpans.length, 2);
    for (const llm of llmSpans) {
      assert.ok(
        llm.parentSpanId === qA.spanContext().spanId ||
          llm.parentSpanId === qB.spanContext().spanId,
        'LLM span must be parented under one of the two query spans',
      );
      assert.equal(llm.attributes['llm.token_count.completion'], 3);
    }
  });

  it('wraps query() only once when wireClaudeAgentSDKInstrumentation runs twice on the same module', async () => {
    let callCount = 0;
    const sdk = {
      async *query(_params: QueryParams) {
        callCount += 1;
        yield {
          type: 'assistant',
          message: {
            id: 'msg',
            role: 'assistant',
            model: 'claude-opus-4-7',
            usage: { input_tokens: 1, output_tokens: 1 },
            content: [{ type: 'text', text: 'hi' }],
          },
        };
        yield { type: 'result', result: 'done', usage: { output_tokens: 1 } };
      },
    };
    wireClaudeAgentSDKInstrumentation(sdk);
    wireClaudeAgentSDKInstrumentation(sdk); // must be a no-op (WRAPPED guard)

    for await (const _m of sdk.query({ prompt: 'p', options: { model: 'claude-opus-4-7' } })) {
      // consume
    }

    assert.equal(callCount, 1, 'double-wiring must not double-invoke the underlying query()');
    const querySpans = exporter.getFinishedSpans().filter((s) => s.name === QUERY_SPAN);
    assert.equal(
      querySpans.length,
      1,
      'double-wiring must not produce nested/duplicate query spans',
    );
  });

  it('handles a query() called with an empty params object (no prompt, no options)', async () => {
    const sdk = {
      async *query(_params: QueryParams) {
        yield {
          type: 'assistant',
          message: {
            id: 'msg',
            role: 'assistant',
            model: 'claude-opus-4-7',
            usage: { input_tokens: 4, output_tokens: 2 },
            content: [{ type: 'text', text: 'hi' }],
          },
        };
        yield { type: 'result', result: 'done', usage: { output_tokens: 2 } };
      },
    };
    wireClaudeAgentSDKInstrumentation(sdk);

    // Should not throw despite prompt/options being absent.
    for await (const _m of sdk.query({} as QueryParams)) {
      // consume
    }

    const query = exporter.getFinishedSpans().find((s) => s.name === QUERY_SPAN)!;
    assert.ok(query, 'query span should still be produced for empty params');
    assert.equal(query.status.code, 1);
    const llm = exporter.getFinishedSpans().find((s) => s.name === LLM_SPAN)!;
    assert.ok(llm, 'LLM span should still be produced');
    // Model comes from the message, not options.
    assert.equal(llm.attributes['llm.model_name'], 'claude-opus-4-7');
  });

  it('does not leak the LLM span when the final assistant chunk of a group is missing its .message', async () => {
    // A group is opened by an assistant chunk that carries usage (span created
    // eagerly). A later chunk in the SAME group (same model/parent, no id) is
    // malformed: it has no `.message`. emitLLMSpan bails on `!lastMessage.message`
    // WITHOUT ending the already-open span, and endInFlight only .clear()s the
    // active-span map. If this leaks, the span is started but never exported.
    const sdk = {
      async *query(_params: QueryParams) {
        yield {
          type: 'assistant',
          message: {
            // no id, so grouping falls back to parent+model equality
            role: 'assistant',
            model: 'claude-opus-4-7',
            usage: { input_tokens: 10, output_tokens: 5 },
            content: [{ type: 'text', text: 'partial' }],
          },
        };
        // Malformed assistant chunk: same (absent) id, same parent, but no `.message`.
        yield { type: 'assistant' };
        yield { type: 'result', result: 'done', usage: { output_tokens: 5 } };
      },
    };
    wireClaudeAgentSDKInstrumentation(sdk);

    for await (const _m of sdk.query({ prompt: 'p', options: { model: 'claude-opus-4-7' } })) {
      // consume
    }

    assert.equal(
      counter.leaked(LLM_SPAN),
      0,
      'LLM span was started but never ended (leak) when the last chunk of the group lacks .message',
    );
  });

  it('produces an LLM span with model but no token attributes when usage is entirely absent', async () => {
    const sdk = {
      async *query(_params: QueryParams) {
        yield {
          type: 'assistant',
          message: {
            id: 'msg',
            role: 'assistant',
            model: 'claude-opus-4-7',
            // no usage at all
            content: [{ type: 'text', text: 'hi' }],
          },
        };
        yield { type: 'result', result: 'done' };
      },
    };
    wireClaudeAgentSDKInstrumentation(sdk);

    for await (const _m of sdk.query({ prompt: 'p', options: { model: 'claude-opus-4-7' } })) {
      // consume
    }

    const llm = exporter.getFinishedSpans().find((s) => s.name === LLM_SPAN)!;
    assert.ok(llm, 'LLM span should still be produced when usage is absent');
    assert.equal(llm.attributes['llm.model_name'], 'claude-opus-4-7');
    assert.equal(llm.attributes['llm.token_count.prompt'], undefined);
    assert.equal(llm.attributes['llm.token_count.completion'], undefined);
    assert.equal(counter.leaked(LLM_SPAN), 0);
  });

  it('drops negative token counts instead of recording them as span attributes', async () => {
    const sdk = {
      async *query(_params: QueryParams) {
        yield {
          type: 'assistant',
          message: {
            id: 'msg',
            role: 'assistant',
            model: 'claude-opus-4-7',
            usage: { input_tokens: -5, output_tokens: -3 },
            content: [{ type: 'text', text: 'hi' }],
          },
        };
        yield { type: 'result', result: 'done', usage: { output_tokens: -3 } };
      },
    };
    wireClaudeAgentSDKInstrumentation(sdk);

    for await (const _m of sdk.query({ prompt: 'p', options: { model: 'claude-opus-4-7' } })) {
      // consume
    }

    const llm = exporter.getFinishedSpans().find((s) => s.name === LLM_SPAN)!;
    assert.ok(llm);
    assert.equal(llm.attributes['llm.token_count.prompt'], undefined);
    assert.equal(llm.attributes['llm.token_count.completion'], undefined);
    assert.equal(llm.attributes['llm.token_count.total'], undefined);
  });
});
