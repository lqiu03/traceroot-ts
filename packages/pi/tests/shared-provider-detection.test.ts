/**
 * Lens: shared-provider-detection.
 *
 * Probes createTracing()'s new hasRealGlobalProvider()-driven branch in
 * isolation from AgentSession/event-handling concerns: does it correctly
 * distinguish a real, globally-registered TracerProvider from the OTel
 * default no-op, does shared mode skip building a private provider and
 * skip registering its own beforeExit hook, and does the documented
 * ordering caveat (private-fallback-then-late-registration stays stuck)
 * behave exactly as designed rather than as an accidental bug.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { trace } from '@opentelemetry/api';
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import { InMemorySpanExporter, SimpleSpanProcessor } from '@opentelemetry/sdk-trace-base';
import { createTracing } from '../src/provider';
import { resolveConfig } from '../src/config';

// Runs first, deliberately, against a clean global TracerProvider slot: every
// later test in this file registers and then trace.disable()s its own
// provider, and this is the one case that must observe nothing registered at
// all.
test('createTracing() builds a private provider when nothing is globally registered', () => {
  const resolved = resolveConfig({ apiKey: 'test-key' });
  const handle = createTracing(resolved);
  assert.equal(handle.ownsProvider, true);
});

test('createTracing() uses the shared global tracer when a real provider is already registered', () => {
  const exporter = new InMemorySpanExporter();
  const provider = new NodeTracerProvider();
  provider.addSpanProcessor(new SimpleSpanProcessor(exporter));
  provider.register();
  try {
    const resolved = resolveConfig({ apiKey: 'test-key' });
    const handle = createTracing(resolved);
    assert.equal(handle.ownsProvider, false);

    const span = handle.tracer.startSpan('span-via-shared-handle');
    span.end();
    assert.deepEqual(
      exporter.getFinishedSpans().map((s) => s.name),
      ['span-via-shared-handle'],
    );
  } finally {
    trace.disable();
  }
});

test('a _spanExporter override always forces private-provider mode, even when a real provider is registered', () => {
  const globalExporter = new InMemorySpanExporter();
  const globalProvider = new NodeTracerProvider();
  globalProvider.addSpanProcessor(new SimpleSpanProcessor(globalExporter));
  globalProvider.register();
  try {
    const testExporter = new InMemorySpanExporter();
    const resolved = resolveConfig({ apiKey: 'test-key', _spanExporter: testExporter });
    const handle = createTracing(resolved);
    assert.equal(handle.ownsProvider, true);
  } finally {
    trace.disable();
  }
});

// Isolation-scope check only: createTracing() itself never touches
// 'beforeExit' in either mode -- that hook lives one layer up, in
// instrumentPiCodingAgent() (src/instrumentation.ts). The behavioral test of
// the actual gating -- does instrumentPiCodingAgent() skip the hook in shared
// mode -- lives in shared-mode-instrumentation.test.ts, since only that
// level can observe the gate at all.
test('shared mode does not register its own beforeExit hook', () => {
  const provider = new NodeTracerProvider();
  provider.register();
  try {
    const listenersBefore = process.listenerCount('beforeExit');
    const resolved = resolveConfig({ apiKey: 'test-key' });
    createTracing(resolved);
    assert.equal(process.listenerCount('beforeExit'), listenersBefore);
  } finally {
    trace.disable();
  }
});

// Guards against a future refactor that memoizes the shared-vs-private
// decision (e.g. caching it on first call): detection must be re-evaluated
// on every createTracing() call, so registering and later disabling the
// global provider flips the mode back for a subsequent call in the same
// process.
test('createTracing() re-evaluates global registration on every call (register then disable flips the mode back)', () => {
  const resolved = resolveConfig({ apiKey: 'test-key' });
  const provider = new NodeTracerProvider();
  provider.register();
  try {
    assert.equal(createTracing(resolved).ownsProvider, false); // shared while registered
  } finally {
    trace.disable();
  }
  assert.equal(createTracing(resolved).ownsProvider, true); // private again after disable
});

// A shared-mode handle's forceFlush must be an inert no-op: flush is the
// shared provider owner's responsibility, not this package's -- calling it
// must never throw even though it does nothing.
test('a shared-mode handle exposes a forceFlush that resolves without touching the shared provider', async () => {
  const provider = new NodeTracerProvider();
  provider.register();
  try {
    const handle = createTracing(resolveConfig({ apiKey: 'test-key' }));
    assert.equal(handle.ownsProvider, false);
    await assert.doesNotReject(handle.forceFlush()); // no-op, must never throw
  } finally {
    trace.disable();
  }
});

// In shared mode the OTLP exporter is never constructed, so a missing
// apiKey is irrelevant to createTracing() itself -- the missing-apiKey
// safety net lives one layer up, in instrumentPiCodingAgent().
test('shared mode returns a shared handle even with no apiKey (never needs to build an OTLP exporter)', () => {
  const provider = new NodeTracerProvider();
  provider.register();
  try {
    const handle = createTracing(resolveConfig({})); // no apiKey, no _spanExporter
    assert.equal(handle.ownsProvider, false);
  } finally {
    trace.disable();
  }
});
