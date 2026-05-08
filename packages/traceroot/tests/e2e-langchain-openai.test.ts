// E2E (in-memory): proves the entire OI Instrumentor family round-trips through
// the OI Vercel SpanProcessor without distortion.
//
// Why LangChain + OpenAI specifically: LangChain / OpenAI / Anthropic / Bedrock /
// claude-agent-sdk all emit through @arizeai/openinference-instrumentation-* with
// the same OI semantic-conventions contract. If LangChain's spans round-trip
// untouched, the others do too by construction.
//
// Skips unless OPENAI_API_KEY is set. Single small ChatOpenAI.invoke call (~$0.0001).

import { afterEach, beforeEach, describe, it } from 'node:test';
import assert from 'node:assert/strict';

import { InMemorySpanExporter, ReadableSpan } from '@opentelemetry/sdk-trace-base';
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import { context, propagation, trace } from '@opentelemetry/api';
import { OpenInferenceSimpleSpanProcessor } from '@arizeai/openinference-vercel';
import { LangChainInstrumentation } from '@arizeai/openinference-instrumentation-langchain';

import { TraceRootSpanProcessor } from '../src/processor';
import { _resetObserveState } from '../src/observe';

interface TestRig {
  exporter: InMemorySpanExporter;
  provider: NodeTracerProvider;
  lcInstr: LangChainInstrumentation;
}

function makeRig(): TestRig {
  const exporter = new InMemorySpanExporter();
  const oi = new OpenInferenceSimpleSpanProcessor({ exporter });
  const provider = new NodeTracerProvider();
  provider.addSpanProcessor(new TraceRootSpanProcessor(oi, { environment: 'e2e-langchain' }));
  provider.register();

  // Manually patch LangChain after the TracerProvider is registered so the
  // Instrumentor picks up the right tracer.
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const lcCallbackManager = require('@langchain/core/callbacks/manager');
  const lcInstr = new LangChainInstrumentation();
  lcInstr.manuallyInstrument(lcCallbackManager);

  return { exporter, provider, lcInstr };
}

async function teardownRig(rig: TestRig): Promise<void> {
  rig.lcInstr.disable();
  await rig.provider.shutdown();
  rig.exporter.reset();
  trace.disable();
  context.disable();
  propagation.disable();
  _resetObserveState();
}

const SKIP = !process.env.OPENAI_API_KEY;

describe('E2E: LangChain + OpenAI (proves OI Instrumentor family transparency)', () => {
  let rig: TestRig;

  beforeEach(() => {
    rig = makeRig();
  });

  afterEach(async () => {
    await teardownRig(rig);
  });

  it(
    'LangChain Instrumentor emits OI-shaped spans that pass through OI Vercel processor unchanged',
    { skip: SKIP },
    async () => {
      // Lazy import after the TracerProvider is registered.
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      const { ChatOpenAI } = require('@langchain/openai');
      const model = new ChatOpenAI({
        model: 'gpt-4o-mini',
        temperature: 0,
        maxRetries: 0,
      });

      const response = await model.invoke([{ role: 'user', content: 'reply with the word ok' }]);
      assert.ok(response, 'LangChain invoke must return a response');

      await rig.provider.forceFlush();
      const spans: ReadableSpan[] = rig.exporter.getFinishedSpans();
      assert.ok(spans.length > 0, 'expected at least one span from LangChain Instrumentor');

      // Find the LLM span — LangChain Instrumentor sets openinference.span.kind === 'LLM'.
      const llmSpan = spans.find((s) => s.attributes['openinference.span.kind'] === 'LLM');
      assert.ok(
        llmSpan,
        `expected an LLM-kind span; got names=[${spans.map((s) => s.name).join(', ')}], kinds=[${spans.map((s) => s.attributes['openinference.span.kind']).join(', ')}]`,
      );

      // ── OI attributes round-trip unchanged through OI Vercel processor ──
      assert.equal(llmSpan.attributes['openinference.span.kind'], 'LLM');

      const modelName = llmSpan.attributes['llm.model_name'];
      assert.equal(typeof modelName, 'string', 'llm.model_name must be a string');
      assert.ok(
        String(modelName).includes('gpt-4o-mini'),
        `llm.model_name should contain gpt-4o-mini; got ${String(modelName)}`,
      );

      const promptTokens = llmSpan.attributes['llm.token_count.prompt'];
      const completionTokens = llmSpan.attributes['llm.token_count.completion'];
      assert.equal(
        typeof promptTokens,
        'number',
        `llm.token_count.prompt must be a number; got ${typeof promptTokens}`,
      );
      assert.equal(typeof completionTokens, 'number');
      assert.ok((promptTokens as number) > 0);
      assert.ok((completionTokens as number) > 0);

      const inputValue = llmSpan.attributes['input.value'];
      const outputValue = llmSpan.attributes['output.value'];
      assert.equal(typeof inputValue, 'string');
      assert.equal(typeof outputValue, 'string');
      assert.ok(String(inputValue).length > 0, 'input.value must be non-empty');
      assert.ok(String(outputValue).length > 0, 'output.value must be non-empty');

      // ── No Vercel ai.* keys forced onto a non-Vercel span ──
      assert.equal(
        llmSpan.attributes['ai.model.id'],
        undefined,
        'OI Vercel processor must not inject ai.model.id onto LangChain spans',
      );
      assert.equal(llmSpan.attributes['ai.prompt'], undefined);
      assert.equal(llmSpan.attributes['ai.response.text'], undefined);
      assert.equal(llmSpan.attributes['ai.usage.promptTokens'], undefined);

      // ── Span name not renamed (OI Instrumentor span name preserved) ──
      // LangChain Instrumentor uses names like "ChatOpenAI", not "ai.generateText".
      assert.ok(
        !llmSpan.name.startsWith('ai.'),
        `LangChain span name must not be coerced to ai.* form; got ${llmSpan.name}`,
      );

      // ── TraceRoot SDK markers applied to the LangChain span ──
      assert.equal(llmSpan.attributes['traceroot.sdk.name'], 'traceroot-ts');
      assert.equal(llmSpan.attributes['deployment.environment'], 'e2e-langchain');
    },
  );
});
