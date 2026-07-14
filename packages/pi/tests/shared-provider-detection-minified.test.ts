/**
 * Lens: shared-provider detection must survive production-bundler
 * class-name mangling.
 *
 * hasRealGlobalProvider() decides shared-vs-private mode. A detection that
 * keys off `delegate.constructor.name === 'NoopTracerProvider'` is fragile:
 * webpack prod mode / esbuild / Terser routinely rename classes, which can
 * silently flip the decision either way (a real provider misread as the
 * no-op, or vice versa). These tests pin the decision to observable span
 * BEHAVIOR — recording state and SpanContext validity — which no minifier
 * can rename away.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { trace } from '@opentelemetry/api';
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import {
  AlwaysOffSampler,
  InMemorySpanExporter,
  SimpleSpanProcessor,
} from '@opentelemetry/sdk-trace-base';
import { createTracing } from '../src/provider';
import { resolveConfig } from '../src/config';

test('a real registered provider is detected as shared even when its class name is minified to "NoopTracerProvider"', () => {
  const exporter = new InMemorySpanExporter();
  // A real provider whose class NAME has been mangled to exactly the string
  // the old name-string check special-cased — spans still record through it;
  // only the reported class name changed, as a minifier would do. Scoped to a
  // one-off subclass so the real NodeTracerProvider.name is left untouched for
  // every other test in the process.
  class MinifiedProvider extends NodeTracerProvider {}
  Object.defineProperty(MinifiedProvider, 'name', { value: 'NoopTracerProvider' });
  const provider = new MinifiedProvider();
  provider.addSpanProcessor(new SimpleSpanProcessor(exporter));
  provider.register();
  try {
    const handle = createTracing(resolveConfig({ apiKey: 'test-key' }));
    // Old code: constructor.name === 'NoopTracerProvider' -> misclassified as
    // the no-op -> builds a private provider (ownsProvider true). The
    // behavioral probe must instead see a recording span and pick shared mode.
    assert.equal(
      handle.ownsProvider,
      false,
      'a real provider must be detected as shared even when its class name is mangled to NoopTracerProvider',
    );

    // And it must genuinely route spans to that real provider...
    handle.tracer.startSpan('routed-through-shared').end();
    assert.deepEqual(
      exporter.getFinishedSpans().map((s) => s.name),
      ['routed-through-shared'],
      'exactly the caller span must export — no phantom detection-probe span may leak into the trace',
    );
  } finally {
    trace.disable();
  }
});

test('a real provider that is not sampling (valid but non-recording spans) is still detected as shared', () => {
  // A host provider configured with a non-sampling sampler hands back
  // non-recording spans that nonetheless carry a valid, non-zero SpanContext
  // (real trace/span ids). A behavioral probe that only checked isRecording()
  // would misread this as "no real provider"; SpanContext validity keeps it
  // classified correctly. Guards that branch of the probe against removal.
  const provider = new NodeTracerProvider({ sampler: new AlwaysOffSampler() });
  provider.register();
  try {
    const handle = createTracing(resolveConfig({ apiKey: 'test-key' }));
    assert.equal(
      handle.ownsProvider,
      false,
      'a real provider must be detected as shared even when its sampler produces non-recording spans',
    );
  } finally {
    trace.disable();
  }
});

test('the OTel default (no real provider registered) is still detected as private', () => {
  // Positive control: with nothing registered, the API default hands back a
  // non-recording span with the all-zero, invalid INVALID_SPAN_CONTEXT, so
  // the probe must fall through to private mode.
  const handle = createTracing(resolveConfig({ apiKey: 'test-key' }));
  assert.equal(handle.ownsProvider, true);
});
