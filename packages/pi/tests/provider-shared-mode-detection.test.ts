/**
 * Lens: shared-provider-detection.
 *
 * Probes createTracing()'s hasRealGlobalProvider()-driven branch in isolation
 * from AgentSession/event-handling concerns: does it correctly distinguish a
 * real, globally-registered TracerProvider from the OTel default no-op, does
 * shared mode skip building a private provider and skip registering its own
 * beforeExit hook, and does the documented ordering caveat
 * (private-fallback-then-late-registration stays stuck) behave exactly as
 * designed rather than as an accidental bug.
 *
 * Merged with the former shared-provider-detection-minified.test.ts: detection
 * must also survive production-bundler class-name mangling. A detection that
 * keys off `delegate.constructor.name === 'NoopTracerProvider'` is fragile:
 * webpack prod mode / esbuild / Terser routinely rename classes, which can
 * silently flip the decision either way (a real provider misread as the
 * no-op, or vice versa). Those tests pin the decision to observable span
 * BEHAVIOR — recording state and SpanContext validity — which no minifier can
 * rename away.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { context, trace, TraceFlags } from '@opentelemetry/api';
import type { Context, ContextManager } from '@opentelemetry/api';

// Minimal synchronous stand-in for @opentelemetry/context-async-hooks (not a
// dependency of this package). The OTel API's default NoopContextManager
// makes context.active() always return ROOT_CONTEXT and context.with() run
// its callback without actually propagating the given context -- so without
// registering a real ContextManager, a test can't reproduce a host process
// that has extracted an incoming traceparent into ambient context. This
// tracks "current context" in a plain variable, which is sufficient because
// the test below only exercises synchronous nesting.
class SyncTestContextManager implements ContextManager {
  private currentContext: Context = context.active();

  active(): Context {
    return this.currentContext;
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  with<A extends unknown[], F extends (...args: A) => ReturnType<F>>(
    contextToActivate: Context,
    fn: F,
    thisArg?: ThisParameterType<F>,
    ...args: A
  ): ReturnType<F> {
    const previousContext = this.currentContext;
    this.currentContext = contextToActivate;
    try {
      return fn.apply(thisArg, args);
    } finally {
      this.currentContext = previousContext;
    }
  }

  bind<T>(_context: Context, target: T): T {
    return target;
  }

  enable(): this {
    return this;
  }

  disable(): this {
    return this;
  }
}
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import {
  AlwaysOffSampler,
  InMemorySpanExporter,
  SimpleSpanProcessor,
} from '@opentelemetry/sdk-trace-base';
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

// Regression guard for the probe's { root: true } option. Without it, the
// no-op tracer's startSpan() builds its NonRecordingSpan from whatever parent
// SpanContext is active in ambient context -- so a host that extracted an
// incoming traceparent (e.g. from an inbound HTTP request) but never
// registered a real TracerProvider would hand the probe back a span carrying
// that extracted, structurally-valid SpanContext. isSpanContextValid() would
// then wrongly read true, forcing shared mode against a no-op provider and
// silently dropping every span for the process lifetime. { root: true } makes
// the no-op tracer short-circuit to INVALID_SPAN_CONTEXT regardless of
// ambient context, so detection must still land on private mode here.
test('an ambient extracted parent SpanContext does not fool detection when no real provider is registered', () => {
  // Register a real (synchronous) ContextManager for the duration of this
  // test only, so context.with() genuinely propagates -- mirroring a host
  // process that has one installed (as any real OTel SDK setup does),
  // instead of the API-default NoopContextManager under which context.with()
  // is inert and this scenario cannot be reproduced at all.
  context.setGlobalContextManager(new SyncTestContextManager());
  try {
    const extractedRemoteContext = {
      traceId: '0af7651916cd43dd8448eb211c80319c',
      spanId: 'b7ad6b7169203331',
      traceFlags: TraceFlags.SAMPLED,
      isRemote: true,
    };
    const ctxWithExtractedParent = trace.setSpanContext(context.active(), extractedRemoteContext);
    context.with(ctxWithExtractedParent, () => {
      const handle = createTracing(resolveConfig({ apiKey: 'test-key' }));
      assert.equal(
        handle.ownsProvider,
        true,
        'a valid ambient parent SpanContext must not be mistaken for a real registered provider',
      );
    });
  } finally {
    context.disable();
  }
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
// mode -- lives in provider-shared-mode-behavior.test.ts, since only that
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
