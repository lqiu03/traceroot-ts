// E2E (in-memory): vanilla traceroot.observe() with no AI semantics.
//
// Directly addresses Xinwei's residual flag about the openinference.span.kind=undefined
// write inside @arizeai/openinference-vercel's addOpenInferenceAttributesToSpan.
// Proves: even though the OI Vercel processor's onEnd helper assigns
// `span.attributes['openinference.span.kind'] = undefined` for non-Vercel spans,
// the assignment of undefined does NOT manifest as an exported attribute. Manual
// user spans come out clean — no AI/LLM keys, no openinference.span.kind, custom
// attrs preserved exactly.
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

  it('observe() span: openinference.span.kind=CHAIN round-trips unchanged through OI Vercel processor', async () => {
    // observe() deliberately tags spans with openinference.span.kind='CHAIN' so
    // they show up correctly in OI-aware UIs. The OI Vercel processor must
    // preserve the CHAIN kind unchanged (existingOISpanKind branch in
    // getOISpanKindFromAttributes returns the existing value verbatim).
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

  it('raw tracer span (no AI semantics, no OI kind set): openinference.span.kind key absent on export', async () => {
    // THE residual case Xinwei flagged. A user span created via raw OTel API
    // (no observe() wrapper, no Instrumentor, no OI attributes) goes through
    // the OI Vercel processor's addOpenInferenceAttributesToSpan, which does:
    //   span.attributes['openinference.span.kind'] = undefined;
    // Verify the assignment of undefined does NOT manifest as an exported
    // attribute key (OTel JS attribute system filters undefined values).
    const tracer = trace.getTracer('raw-manual-test');
    await new Promise<void>((resolve) => {
      tracer.startActiveSpan('raw.work', (span) => {
        span.setAttribute('foo', 'bar');
        span.setAttribute('items.count', 5);
        span.end();
        resolve();
      });
    });

    await rig.provider.forceFlush();
    const span = findSpan(rig.exporter.getFinishedSpans(), 'raw.work');

    // ── THE residual edge case ──
    // The OI Vercel processor's addOpenInferenceAttributesToSpan does:
    //   span.attributes['openinference.span.kind'] = undefined;
    // (direct property assignment, bypassing OTel's setAttribute validation).
    // Result: the key IS in the in-memory attribute object with value undefined,
    // BUT serialization (JSON, OTLP wire format) drops undefined values, so it
    // does not manifest as an observable attribute on export.
    //
    // Verify both halves:
    //   1. In-memory value is undefined (not 'LLM' / 'CHAIN' / any string)
    //   2. JSON-serialized form does not contain the key
    const kind = span.attributes['openinference.span.kind'];
    assert.equal(
      kind,
      undefined,
      `openinference.span.kind value must be undefined for raw non-AI spans; got ${JSON.stringify(kind)}`,
    );
    const serialized = JSON.stringify(span.attributes);
    assert.ok(
      !serialized.includes('openinference.span.kind'),
      `openinference.span.kind must not appear in JSON-serialized attributes; got ${serialized}`,
    );

    // ── No AI/LLM keys injected ──
    assert.equal(span.attributes['llm.model_name'], undefined);
    assert.equal(span.attributes['llm.token_count.prompt'], undefined);
    assert.equal(span.attributes['input.value'], undefined);
    assert.equal(span.attributes['output.value'], undefined);
    assert.equal(span.attributes['ai.model.id'], undefined);

    // ── Custom user attrs preserved ──
    assert.equal(span.attributes['foo'], 'bar');
    assert.equal(span.attributes['items.count'], 5);

    // ── Span name not renamed ──
    assert.equal(span.name, 'raw.work');

    // ── Status remains UNSET (no AI gate triggered) ──
    assert.equal(
      span.status.code,
      SpanStatusCode.UNSET,
      'raw manual span status must remain UNSET; OK or ERROR would mean a gate broke',
    );

    // ── TraceRoot SDK markers still applied ──
    assert.equal(span.attributes['traceroot.sdk.name'], 'traceroot-ts');
  });
});
