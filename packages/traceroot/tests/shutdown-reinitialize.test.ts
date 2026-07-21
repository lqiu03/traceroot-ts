import { afterEach, describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { ProxyTracerProvider, trace } from '@opentelemetry/api';
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import { InMemorySpanExporter, SimpleSpanProcessor } from '@opentelemetry/sdk-trace-base';
import { TraceRoot, _resetForTesting } from '../src/traceroot';

// The globally-registered tracer provider is always the OTel API's singleton
// ProxyTracerProvider; its delegate is the concrete NodeTracerProvider that
// initialize() created and register()'d. Reading the delegate proves WHICH
// provider is actually active without reaching into TraceRoot's private module
// state (it exposes no getter for _provider).
function activeProvider() {
  return (trace.getTracerProvider() as ProxyTracerProvider).getDelegate();
}

async function cycle(apiKey: string) {
  TraceRoot.initialize({ apiKey, disableBatch: true });
  const provider = activeProvider();
  await TraceRoot.shutdown();
  return provider;
}

describe('TraceRoot.shutdown() then TraceRoot.initialize() again', () => {
  afterEach(() => {
    _resetForTesting();
  });

  // RED-FIRST regression check for the fix.
  it('re-initializing after shutdown() rewires the OTel global to the new provider', async () => {
    TraceRoot.initialize({ apiKey: 'test-key', disableBatch: true });
    assert.equal(TraceRoot.isInitialized(), true);
    const firstProvider = activeProvider();

    await TraceRoot.shutdown();
    assert.equal(TraceRoot.isInitialized(), false);

    TraceRoot.initialize({ apiKey: 'test-key-2', disableBatch: true });
    assert.equal(TraceRoot.isInitialized(), true);
    const secondProvider = activeProvider();

    // Without the disable() calls in shutdown(), the second register() is
    // silently rejected and activeProvider() still returns the first (dead)
    // provider -- a strict-equal object. The fix makes the second provider
    // genuinely active, so the two references must differ.
    assert.notStrictEqual(secondProvider, firstProvider);
  });

  // RED-FIRST: durability across repeated cycles (each cycle must activate a
  // distinct provider). Unrolled to avoid loops/conditionals in a test.
  it('each shutdown/initialize cycle activates a distinct provider', async () => {
    const a = await cycle('test-key-a');
    const b = await cycle('test-key-b');
    const c = await cycle('test-key-c');
    assert.notStrictEqual(a, b);
    assert.notStrictEqual(b, c);
    assert.notStrictEqual(a, c);
  });

  // GREEN both ways: guards that the new disable() calls don't throw on a
  // never-registered global and don't poison a later initialize().
  it('shutdown() before any initialize() does not throw and a later initialize() still succeeds', async () => {
    await assert.doesNotReject(() => TraceRoot.shutdown());
    assert.equal(TraceRoot.isInitialized(), false);
    TraceRoot.initialize({ apiKey: 'test-key', disableBatch: true });
    assert.equal(TraceRoot.isInitialized(), true);
  });

  // GREEN both ways: the new disable() calls must be idempotent.
  it('calling shutdown() twice in a row is idempotent and does not throw', async () => {
    TraceRoot.initialize({ apiKey: 'test-key', disableBatch: true });
    await assert.doesNotReject(() => TraceRoot.shutdown());
    await assert.doesNotReject(() => TraceRoot.shutdown());
    assert.equal(TraceRoot.isInitialized(), false);
  });

  // GREEN both ways: flush() on a torn-down instance must be a safe no-op.
  it('flush() after shutdown() resolves without throwing', async () => {
    TraceRoot.initialize({ apiKey: 'test-key', disableBatch: true });
    await TraceRoot.shutdown();
    await assert.doesNotReject(() => TraceRoot.flush());
  });
});

describe('TraceRoot.shutdown() does not disable a concurrent OTel consumer', () => {
  afterEach(() => {
    _resetForTesting();
  });

  // RED-FIRST: another OTel consumer in the same process owns the global
  // registration (it registered first, so first-write-wins left TraceRoot's
  // own register() rejected). shutdown()'s trace/context/propagation
  // disable() calls are process-wide singletons — calling them unconditionally
  // would wipe the foreign provider's registration (and context/propagation)
  // out from under it. shutdown() must scope those resets to the case where
  // TraceRoot's provider is still the active global delegate, and otherwise
  // tear down only its own provider.
  it('leaves a foreign provider that owns the global registration active after shutdown()', async () => {
    const foreignExporter = new InMemorySpanExporter();
    const foreignProvider = new NodeTracerProvider();
    foreignProvider.addSpanProcessor(new SimpleSpanProcessor(foreignExporter));
    foreignProvider.register(); // wins the global slot first

    // TraceRoot initializes on top; its register() is rejected (first-write-wins).
    TraceRoot.initialize({ apiKey: 'test-key', disableBatch: true });
    assert.strictEqual(
      (trace.getTracerProvider() as ProxyTracerProvider).getDelegate(),
      foreignProvider,
      'sanity: the foreign provider must still own the global slot after TraceRoot.initialize()',
    );

    await TraceRoot.shutdown();

    // The foreign provider must still be the active global delegate...
    assert.strictEqual(
      (trace.getTracerProvider() as ProxyTracerProvider).getDelegate(),
      foreignProvider,
      'TraceRoot.shutdown() must not disable a foreign provider it never registered',
    );
    // ...and must still actually export through the global tracer.
    const span = trace.getTracer('concurrent-consumer').startSpan('after-traceroot-shutdown');
    span.end();
    assert.deepEqual(
      foreignExporter.getFinishedSpans().map((s) => s.name),
      ['after-traceroot-shutdown'],
      'the foreign consumer must keep tracing through the global provider after TraceRoot.shutdown()',
    );
  });

  // GREEN both ways: the ordinary case — TraceRoot owns the global — must
  // STILL reset it on shutdown so a later initialize() is not rejected. This
  // guards against over-scoping the fix into breaking the reinitialize cycle.
  it('still resets the global when TraceRoot itself owns it, so reinitialize keeps working', async () => {
    TraceRoot.initialize({ apiKey: 'test-key', disableBatch: true });
    const first = (trace.getTracerProvider() as ProxyTracerProvider).getDelegate();
    await TraceRoot.shutdown();
    TraceRoot.initialize({ apiKey: 'test-key-2', disableBatch: true });
    const second = (trace.getTracerProvider() as ProxyTracerProvider).getDelegate();
    assert.notStrictEqual(second, first);
  });
});
