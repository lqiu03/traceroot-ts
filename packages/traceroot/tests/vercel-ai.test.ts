// Vercel AI SDK integration tests.
//
// These exercise the production stack:
//   TraceRootSpanProcessor → OpenInferenceSimpleSpanProcessor → InMemorySpanExporter
//
// Real Vercel AI SDK calls go through @arizeai/openinference-vercel, which mutates
// span attributes onEnd to OpenInference semantic conventions. The assertions are
// on the post-mapping attribute names (input.value, output.value, llm.model_name,
// llm.token_count.*, openinference.span.kind, tool_call.*) so the suite catches
// regressions in either layer.

import { afterEach, beforeEach, describe, it } from 'node:test';
import assert from 'node:assert/strict';

import { InMemorySpanExporter, ReadableSpan } from '@opentelemetry/sdk-trace-base';
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import { context, propagation, SpanStatusCode, trace } from '@opentelemetry/api';
import { OpenInferenceSimpleSpanProcessor } from '@arizeai/openinference-vercel';

import { generateText, streamText, tool } from 'ai';
import { MockLanguageModelV3, simulateReadableStream } from 'ai/test';
import { z } from 'zod';

import { TraceRootSpanProcessor } from '../src/processor';
import { observe, _resetObserveState } from '../src/observe';
import { usingAttributes } from '../src/usingAttributes';

// ─────────────────────────────────────────────────────────────────────────────
// Test infrastructure
// ─────────────────────────────────────────────────────────────────────────────

interface TestRig {
  exporter: InMemorySpanExporter;
  provider: NodeTracerProvider;
}

function makeRig(
  opts: {
    environment?: string;
    gitRepo?: string;
    gitRef?: string;
  } = {},
): TestRig {
  const exporter = new InMemorySpanExporter();
  const oi = new OpenInferenceSimpleSpanProcessor({ exporter });
  const provider = new NodeTracerProvider();
  provider.addSpanProcessor(new TraceRootSpanProcessor(oi, opts));
  provider.register();
  return { exporter, provider };
}

async function teardownRig(rig: TestRig): Promise<void> {
  await rig.provider.shutdown();
  rig.exporter.reset();
  trace.disable();
  context.disable();
  propagation.disable();
  // observe.ts caches a tracer at module level; clear it so the next test's
  // freshly-registered TracerProvider is picked up.
  _resetObserveState();
}

function findSpan(spans: ReadableSpan[], name: string): ReadableSpan {
  const found = spans.find((s) => s.name === name);
  assert.ok(found, `expected a span named "${name}", got [${spans.map((s) => s.name).join(', ')}]`);
  return found;
}

function findSpans(spans: ReadableSpan[], name: string): ReadableSpan[] {
  return spans.filter((s) => s.name === name);
}

// ─────────────────────────────────────────────────────────────────────────────
// Mock builders for MockLanguageModelV3
// ─────────────────────────────────────────────────────────────────────────────

// LanguageModelV3Usage shape — nested objects, not flat numbers.
const STD_USAGE = {
  inputTokens: { total: 11, noCache: 11, cacheRead: undefined, cacheWrite: undefined },
  outputTokens: { total: 7, text: 7, reasoning: undefined },
};

function staticTextModel(text: string) {
  return new MockLanguageModelV3({
    modelId: 'mock-static',
    provider: 'mock-provider',
    doGenerate: async () => ({
      content: [{ type: 'text', text }],
      finishReason: 'stop',
      usage: STD_USAGE,
      warnings: [],
    }),
  });
}

function streamingTextModel(deltas: string[]) {
  return new MockLanguageModelV3({
    modelId: 'mock-stream',
    provider: 'mock-provider',
    doStream: async () => ({
      stream: simulateReadableStream({
        chunks: [
          { type: 'stream-start' as const, warnings: [] },
          { type: 'text-start' as const, id: 'tx1' },
          ...deltas.map((delta) => ({ type: 'text-delta' as const, id: 'tx1', delta })),
          { type: 'text-end' as const, id: 'tx1' },
          {
            type: 'finish' as const,
            finishReason: 'stop' as const,
            usage: STD_USAGE,
          },
        ],
      }),
    }),
  });
}

function throwingModel(message: string) {
  return new MockLanguageModelV3({
    modelId: 'mock-throw',
    provider: 'mock-provider',
    doGenerate: async () => {
      throw new Error(message);
    },
  });
}

function streamThatErrorsAfter(deltas: string[], errorMessage: string) {
  return new MockLanguageModelV3({
    modelId: 'mock-stream-err',
    provider: 'mock-provider',
    doStream: async () => ({
      stream: simulateReadableStream({
        chunks: [
          { type: 'stream-start' as const, warnings: [] },
          { type: 'text-start' as const, id: 'tx1' },
          ...deltas.map((delta) => ({ type: 'text-delta' as const, id: 'tx1', delta })),
          { type: 'error' as const, error: new Error(errorMessage) },
        ],
      }),
    }),
  });
}

// Two-call mock: first call returns a tool-call, second call returns a final text.
function toolCallingModel(toolName: string, toolInput: object, finalText: string) {
  let call = 0;
  return new MockLanguageModelV3({
    modelId: 'mock-tools',
    provider: 'mock-provider',
    doGenerate: async () => {
      call++;
      if (call === 1) {
        return {
          content: [
            {
              type: 'tool-call' as const,
              toolCallId: 't1',
              toolName,
              input: JSON.stringify(toolInput),
            },
          ],
          finishReason: 'tool-calls',
          usage: STD_USAGE,
          warnings: [],
        };
      }
      return {
        content: [{ type: 'text', text: finalText }],
        finishReason: 'stop',
        usage: STD_USAGE,
        warnings: [],
      };
    },
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Suite
// ─────────────────────────────────────────────────────────────────────────────

describe('Vercel AI SDK integration via OpenInference', () => {
  let rig: TestRig;

  beforeEach(() => {
    rig = makeRig();
  });

  afterEach(async () => {
    await teardownRig(rig);
  });

  // ── 1. Non-streaming generateText ─────────────────────────────────────────
  describe('non-streaming generateText', () => {
    it('produces a parent ai.generateText span and a doGenerate child with mapped attributes', async () => {
      const result = await generateText({
        model: staticTextModel('hello world'),
        prompt: 'say hi',
        experimental_telemetry: { isEnabled: true },
      });
      assert.equal(result.text, 'hello world');

      await rig.provider.forceFlush();
      const spans = rig.exporter.getFinishedSpans();

      const wrapper = findSpan(spans, 'ai.generateText');
      const inner = findSpan(spans, 'ai.generateText.doGenerate');
      assert.equal(
        spans.length,
        2,
        `expected exactly 2 spans, got ${spans.length}: [${spans.map((s) => s.name).join(', ')}]`,
      );

      // Per OI's VercelSDKFunctionNameToSpanKindMap:
      //   ai.generateText             → AGENT (the orchestrating wrapper)
      //   ai.generateText.doGenerate  → LLM   (the actual provider call)
      assert.equal(wrapper.attributes['openinference.span.kind'], 'AGENT');
      assert.equal(inner.attributes['openinference.span.kind'], 'LLM');

      // Input/output mapped from ai.prompt / ai.response.text
      assert.ok(wrapper.attributes['input.value'], 'wrapper input.value should be set');
      assert.ok(
        String(wrapper.attributes['input.value']).includes('say hi'),
        `wrapper input.value should contain prompt; got: ${wrapper.attributes['input.value']}`,
      );
      assert.equal(wrapper.attributes['output.value'], 'hello world');

      // Model metadata: model_name lives on both spans; the OI mapping only
      // emits llm.token_count.* on LLM-kind spans, so it lands on the inner
      // doGenerate. The AGENT wrapper carries the raw ai.usage.* aggregates.
      assert.equal(inner.attributes['llm.model_name'], 'mock-static');
      assert.equal(inner.attributes['llm.token_count.prompt'], 11);
      assert.equal(inner.attributes['llm.token_count.completion'], 7);
      assert.equal(wrapper.attributes['ai.usage.inputTokens'], 11);
      assert.equal(wrapper.attributes['ai.usage.outputTokens'], 7);

      // Parent / child relationship preserved through the OI processor
      assert.equal(
        inner.parentSpanId,
        wrapper.spanContext().spanId,
        'doGenerate must be a child of generateText',
      );

      // Status: not ERROR on success (AI SDK may set OK explicitly).
      assert.notEqual(wrapper.status.code, SpanStatusCode.ERROR);

      // TraceRootSpanProcessor still injects SDK metadata after the swap
      assert.equal(wrapper.attributes['traceroot.sdk.name'], 'traceroot-ts');
      assert.ok(wrapper.attributes['traceroot.sdk.version']);
    });
  });

  // ── 2. Streaming streamText ───────────────────────────────────────────────
  describe('streaming streamText', () => {
    it('keeps the wrapper span open until the stream is consumed and lands usage on close', async () => {
      const result = streamText({
        model: streamingTextModel(['hel', 'lo']),
        prompt: 'go',
        experimental_telemetry: { isEnabled: true },
      });

      let collected = '';
      for await (const chunk of result.textStream) collected += chunk;
      assert.equal(collected, 'hello');

      await rig.provider.forceFlush();
      const spans = rig.exporter.getFinishedSpans();

      const wrapper = findSpan(spans, 'ai.streamText');
      const inner = findSpan(spans, 'ai.streamText.doStream');

      assert.equal(findSpans(spans, 'ai.streamText').length, 1);
      assert.equal(findSpans(spans, 'ai.streamText.doStream').length, 1);

      // Token usage on the final LLM span (per OI: llm.token_count.* only on LLM-kind spans).
      // The AGENT-kind wrapper still carries the raw ai.usage.* aggregates.
      assert.equal(inner.attributes['llm.token_count.prompt'], 11);
      assert.equal(inner.attributes['llm.token_count.completion'], 7);
      assert.equal(wrapper.attributes['ai.usage.inputTokens'], 11);
      assert.equal(wrapper.attributes['ai.usage.outputTokens'], 7);

      // Output text reconstructed on both wrapper and inner
      assert.equal(wrapper.attributes['output.value'], 'hello');
      assert.equal(inner.attributes['output.value'], 'hello');

      assert.notEqual(wrapper.status.code, SpanStatusCode.ERROR);
      assert.ok(wrapper.endTime[0] >= wrapper.startTime[0], 'wrapper.endTime must be set');
    });
  });

  // ── 3. Tool calling ───────────────────────────────────────────────────────
  describe('tool calling', () => {
    it('emits an ai.toolCall span with TOOL kind and tool input/output', async () => {
      const result = await generateText({
        model: toolCallingModel('add', { a: 2, b: 3 }, 'sum is 5'),
        prompt: 'compute',
        tools: {
          add: tool({
            description: 'add two numbers',
            inputSchema: z.object({ a: z.number(), b: z.number() }),
            execute: async ({ a, b }) => a + b,
          }),
        },
        stopWhen: ({ steps }) => steps.length >= 2,
        experimental_telemetry: { isEnabled: true },
      });
      assert.equal(result.text, 'sum is 5');

      await rig.provider.forceFlush();
      const spans = rig.exporter.getFinishedSpans();

      const toolSpan = findSpan(spans, 'ai.toolCall');
      assert.equal(toolSpan.attributes['openinference.span.kind'], 'TOOL');
      assert.equal(toolSpan.attributes['tool.name'], 'add');

      const argsAttr = toolSpan.attributes['tool.parameters'] ?? toolSpan.attributes['input.value'];
      assert.ok(
        argsAttr,
        `tool args should be on the span; saw keys: ${Object.keys(toolSpan.attributes).join(', ')}`,
      );
      assert.ok(
        String(argsAttr).includes('"a":2') && String(argsAttr).includes('"b":3'),
        `tool args should contain a:2 b:3; got: ${argsAttr}`,
      );

      const resultAttr = toolSpan.attributes['output.value'];
      assert.ok(resultAttr, 'tool result should be on the span');
      assert.ok(
        String(resultAttr).includes('5'),
        `tool result should contain 5; got: ${resultAttr}`,
      );

      // Parent of the tool span is the generateText wrapper
      const wrapper = findSpan(spans, 'ai.generateText');
      assert.equal(toolSpan.parentSpanId, wrapper.spanContext().spanId);
    });
  });

  // ── 4. Error paths ────────────────────────────────────────────────────────
  describe('error paths', () => {
    it('marks spans ERROR when the provider throws and leaves no leaks', async () => {
      await assert.rejects(
        () =>
          generateText({
            model: throwingModel('boom'),
            prompt: 'go',
            experimental_telemetry: { isEnabled: true },
          }),
        /boom/,
      );

      await rig.provider.forceFlush();
      const spans = rig.exporter.getFinishedSpans();

      const wrapper = findSpan(spans, 'ai.generateText');
      const inner = findSpan(spans, 'ai.generateText.doGenerate');
      assert.equal(spans.length, 2, 'no leaked or extra spans on error path');
      assert.equal(wrapper.status.code, SpanStatusCode.ERROR);
      assert.equal(inner.status.code, SpanStatusCode.ERROR);
      assert.ok(
        wrapper.events.some((e) => e.name === 'exception'),
        'wrapper should record exception event',
      );
    });

    it('marks the streaming wrapper ERROR when the stream emits a fatal error', async () => {
      const stream = streamText({
        model: streamThatErrorsAfter(['par'], 'stream-failed'),
        prompt: 'go',
        experimental_telemetry: { isEnabled: true },
        onError: () => {
          // suppress unhandled — we want to assert on spans, not throw out of consumer
        },
      });
      let collected = '';
      try {
        for await (const chunk of stream.textStream) collected += chunk;
      } catch {
        // expected
      }

      await rig.provider.forceFlush();
      const spans = rig.exporter.getFinishedSpans();

      const wrapper = findSpan(spans, 'ai.streamText');
      assert.equal(wrapper.status.code, SpanStatusCode.ERROR, 'wrapper status should be ERROR');
      assert.equal(findSpans(spans, 'ai.streamText').length, 1, 'no leaked wrapper spans');
      assert.equal(
        findSpans(spans, 'ai.streamText.doStream').length,
        1,
        'no leaked doStream spans',
      );
    });
  });

  // ── 5. Nested call: Vercel inside an observe()-wrapped function ───────────
  describe('nested call', () => {
    it('preserves parent/child when generateText runs inside an observe() span', async () => {
      const _result = await observe({ name: 'outer-fn', type: 'span' }, async () =>
        generateText({
          model: staticTextModel('inside'),
          prompt: 'p',
          experimental_telemetry: { isEnabled: true },
        }),
      );

      await rig.provider.forceFlush();
      const spans = rig.exporter.getFinishedSpans();

      const outer = findSpan(spans, 'outer-fn');
      const wrapper = findSpan(spans, 'ai.generateText');

      assert.equal(
        wrapper.parentSpanId,
        outer.spanContext().spanId,
        'ai.generateText must be a direct child of the observe() span',
      );
      assert.equal(wrapper.spanContext().traceId, outer.spanContext().traceId);
    });
  });

  // ── 6. Pass-through: non-Vercel spans must come through untouched ─────────
  describe('non-Vercel pass-through', () => {
    it('does not mutate manual spans that were never produced by the AI SDK', async () => {
      const tracer = trace.getTracer('manual-test');
      await new Promise<void>((resolve) => {
        tracer.startActiveSpan('manual.work', (span) => {
          span.setAttribute('foo', 'bar');
          span.setAttribute('input.value', 'preserved-input');
          span.end();
          resolve();
        });
      });

      await rig.provider.forceFlush();
      const spans = rig.exporter.getFinishedSpans();

      const manual = findSpan(spans, 'manual.work');
      assert.equal(spans.length, 1);
      assert.equal(manual.attributes['foo'], 'bar', 'arbitrary attrs preserved');
      assert.equal(
        manual.attributes['input.value'],
        'preserved-input',
        'OI processor must not overwrite existing input.value on a non-Vercel span',
      );
      assert.equal(manual.attributes['llm.model_name'], undefined);
      assert.equal(manual.attributes['llm.token_count.prompt'], undefined);
      assert.equal(manual.attributes['openinference.span.kind'], undefined);

      assert.equal(manual.attributes['traceroot.sdk.name'], 'traceroot-ts');
    });
  });

  // ── 7. TraceRootSpanProcessor injection survives the processor swap ───────
  describe('TraceRootSpanProcessor injection', () => {
    it('applies traceroot.git.* / deployment.environment to AI SDK spans', async () => {
      await teardownRig(rig);
      rig = makeRig({ environment: 'staging', gitRepo: 'org/repo', gitRef: 'abc123' });

      await generateText({
        model: staticTextModel('ok'),
        prompt: 'p',
        experimental_telemetry: { isEnabled: true },
      });

      await rig.provider.forceFlush();
      const wrapper = findSpan(rig.exporter.getFinishedSpans(), 'ai.generateText');

      assert.equal(wrapper.attributes['deployment.environment'], 'staging');
      assert.equal(wrapper.attributes['traceroot.git.repo'], 'org/repo');
      assert.equal(wrapper.attributes['traceroot.git.ref'], 'abc123');
      assert.equal(wrapper.attributes['traceroot.sdk.name'], 'traceroot-ts');
    });
  });

  // ── 8. Session/user context propagation (the #2651 fix) ───────────────────
  describe('session/user context propagation', () => {
    it('lifts session.id and user.id from usingAttributes() onto Vercel AI SDK spans', async () => {
      await usingAttributes(
        {
          sessionId: 'sess-42',
          userId: 'u-7',
          tags: ['prod', 'beta'],
          metadata: { feature: 'chat' },
        },
        async () => {
          await generateText({
            model: staticTextModel('hi'),
            prompt: 'p',
            experimental_telemetry: { isEnabled: true },
          });
        },
      );

      await rig.provider.forceFlush();
      const wrapper = findSpan(rig.exporter.getFinishedSpans(), 'ai.generateText');

      assert.equal(
        wrapper.attributes['session.id'],
        'sess-42',
        'session.id must propagate from usingAttributes() to ai.generateText (the #2651 case)',
      );
      assert.equal(wrapper.attributes['user.id'], 'u-7');
      const tagsAttr = wrapper.attributes['tag.tags'];
      assert.ok(tagsAttr, 'tag.tags must be set');
      assert.ok(
        String(tagsAttr).includes('prod') && String(tagsAttr).includes('beta'),
        `tag.tags should contain prod and beta; got: ${tagsAttr}`,
      );
    });
  });
});
