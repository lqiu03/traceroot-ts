// E2E (in-memory): vanilla traceroot.observe() with no AI semantics.
//
// Verifies that an observe()-wrapped span passes through the OI Vercel
// processor cleanly: CHAIN kind preserved, no Vercel ai.* keys forced on,
// no LLM keys forced on, custom user attributes preserved exactly, span
// name not renamed, and SDK markers applied.
//
// No external dependencies. Runs unconditionally.

import { afterEach, beforeEach, describe, it } from 'node:test';
import assert from 'node:assert/strict';

import { InMemorySpanExporter, ReadableSpan } from '@opentelemetry/sdk-trace-base';
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import { context, propagation, SpanStatusCode, trace } from '@opentelemetry/api';
import { OpenInferenceSimpleSpanProcessor } from '@arizeai/openinference-vercel';

import { TraceRootSpanProcessor } from '../src/processor';
import { observe, _resetObserveState } from '../src/observe';

interface TestRig {
  exporter: InMemorySpanExporter;
  provider: NodeTracerProvider;
}

function makeRig(): TestRig {
  const exporter = new InMemorySpanExporter();
  const oi = new OpenInferenceSimpleSpanProcessor({ exporter });
  const provider = new NodeTracerProvider();
  provider.addSpanProcessor(new TraceRootSpanProcessor(oi, { environment: 'e2e-vanilla' }));
  provider.register();
  return { exporter, provider };
}

async function teardownRig(rig: TestRig): Promise<void> {
  await rig.provider.shutdown();
  rig.exporter.reset();
  trace.disable();
  context.disable();
  propagation.disable();
  _resetObserveState();
}

function findSpan(spans: ReadableSpan[], name: string): ReadableSpan {
  const found = spans.find((s) => s.name === name);
  assert.ok(found, `expected a span named "${name}", got [${spans.map((s) => s.name).join(', ')}]`);
  return found;
}

describe('E2E: vanilla observe() with no AI semantics', () => {
  let rig: TestRig;

  beforeEach(() => {
    rig = makeRig();
  });

  afterEach(async () => {
    await teardownRig(rig);
  });

  it('observe() span: CHAIN kind round-trips unchanged through OI Vercel processor', async () => {
    // observe() tags spans with openinference.span.kind='CHAIN' so they show up
    // correctly in OI-aware UIs. Verify the OI Vercel processor preserves the
    // CHAIN kind unchanged and does not inject any AI/LLM attributes onto a
    // non-Vercel span.
    const result = await observe({ name: 'billing.calculate' }, async () => {
      const activeSpan = trace.getActiveSpan();
      assert.ok(activeSpan, 'observe() must create an active span');
      activeSpan.setAttribute('app.feature', 'invoice');
      activeSpan.setAttribute('customer.tier', 'enterprise');
      activeSpan.setAttribute('items.count', 3);
      return { total: 42 };
    });
    assert.deepEqual(result, { total: 42 });

    await rig.provider.forceFlush();
    const spans = rig.exporter.getFinishedSpans();
    assert.equal(spans.length, 1);

    const span = findSpan(spans, 'billing.calculate');

    // ── observe() set CHAIN; OI Vercel processor preserved it unchanged ──
    assert.equal(
      span.attributes['openinference.span.kind'],
      'CHAIN',
      'observe() sets CHAIN kind; OI Vercel processor must preserve, not overwrite',
    );

    // ── No Vercel ai.* keys forced onto a non-Vercel observe() span ──
    assert.equal(span.attributes['ai.model.id'], undefined);
    assert.equal(span.attributes['ai.prompt'], undefined);
    assert.equal(span.attributes['ai.response.text'], undefined);
    assert.equal(span.attributes['ai.usage.promptTokens'], undefined);

    // ── No LLM-specific keys forced (observe() is CHAIN, not LLM) ──
    assert.equal(span.attributes['llm.model_name'], undefined);
    assert.equal(span.attributes['llm.token_count.prompt'], undefined);
    assert.equal(span.attributes['llm.token_count.completion'], undefined);

    // ── Custom user attrs preserved exactly ──
    assert.equal(span.attributes['app.feature'], 'invoice');
    assert.equal(span.attributes['customer.tier'], 'enterprise');
    assert.equal(span.attributes['items.count'], 3);

    // ── Span name not renamed ──
    assert.equal(span.name, 'billing.calculate');

    // ── Status not forced to ERROR by the OI Vercel processor ──
    assert.notEqual(span.status.code, SpanStatusCode.ERROR);

    // ── TraceRoot SDK markers applied ──
    assert.equal(span.attributes['traceroot.sdk.name'], 'traceroot-ts');
    assert.equal(span.attributes['deployment.environment'], 'e2e-vanilla');
  });
});
