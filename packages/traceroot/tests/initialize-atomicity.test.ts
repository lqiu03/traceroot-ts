/**
 * Regression coverage for TraceRoot.initialize()/shutdown() global-slot ownership
 * (see traceroot.ts: _releaseGlobalSlots and the wiring-failure rollback):
 *
 *  - initialize() atomicity when an instrumentation-wiring step throws. A
 *    claudeAgentSDK ref missing query() makes the dispatch throw AFTER
 *    _provider.register() won the global trace/context/propagation slots but
 *    BEFORE _isInitialized is set. Without initialize()'s catch-block rollback,
 *    that registration is left orphaned on the global slots with _isInitialized
 *    stuck false, so a retried initialize() silently loses the first-write-wins
 *    race instead of recovering.
 *  - beforeExit listener hygiene across repeated init/shutdown cycles — shutdown()
 *    must remove exactly the process.once('beforeExit', ...) flush hook that
 *    initialize() installed, or each cycle leaks one permanently (and trips Node's
 *    MaxListenersExceededWarning at 11).
 *
 * These drive the REAL initialize() end to end. A local, unroutable baseUrl keeps
 * the OTLP exporter from ever touching the network.
 */
import { afterEach, describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { trace } from '@opentelemetry/api';
import { TraceRoot, _resetForTesting } from '../src/traceroot';

// The concrete provider the OTel global proxy currently delegates to — i.e. the
// provider that spans actually route to right now, regardless of what TraceRoot's
// private _provider field points at. Reading it the same way traceroot.ts's own
// isActiveGlobalDelegate() does, without reaching into module-private state.
function activeProvider(): unknown {
  const p = trace.getTracerProvider() as { getDelegate?: () => unknown };
  return typeof p.getDelegate === 'function' ? p.getDelegate() : p;
}

const BASE = {
  apiKey: 'trk_core_wiring',
  baseUrl: 'http://127.0.0.1:9',
  disableBatch: true as const,
  gitRepo: 'traceroot-ai/traceroot-ts',
  gitRef: 'core-wiring-ref',
};

afterEach(() => {
  _resetForTesting();
  delete process.env.TRACEROOT_API_KEY;
});

describe('initialize() atomicity when an instrumentation-wiring step throws', () => {
  it('a retried initialize() after a wiring throw makes ITS provider the active global delegate', () => {
    delete process.env.TRACEROOT_API_KEY;

    // A claudeAgentSDK ref missing query() is a realistic misconfiguration; the
    // dispatch throws for it.
    let threw = false;
    try {
      TraceRoot.initialize({ ...BASE, instrumentModules: { claudeAgentSDK: {} } });
    } catch {
      threw = true;
    }
    assert.equal(threw, true, 'a claudeAgentSDK missing query() must surface as a throw');

    // The throw happened AFTER _provider.register() won the global slots but
    // BEFORE _isInitialized was set — so TraceRoot reports "not initialized"...
    assert.equal(
      TraceRoot.isInitialized(),
      false,
      'a wiring throw must not leave TraceRoot reporting initialized',
    );
    // ...regression coverage: initialize()'s catch block rolls back that
    // registration (trace/context/propagation .disable(), per the same
    // per-slot ownership checks shutdown() uses) before re-throwing, so no
    // provider is left registered as the live global delegate.
    const rolledBack = activeProvider();

    // Because _isInitialized is still false, the Already-initialized guard does
    // NOT block a retry. A retry builds a second provider and register()s it.
    // Correct behavior: that second provider becomes the active global delegate,
    // so TraceRoot's own flush()/shutdown() (which act on it) match where spans
    // actually route.
    TraceRoot.initialize({ ...BASE, apiKey: 'trk_retry' });
    assert.equal(TraceRoot.isInitialized(), true, 'the retry reports initialized');
    const afterRetry = activeProvider();

    // Regression coverage: without the rollback, OTel global registration is
    // first-write-wins, so a still-registered orphaned provider would win this
    // race and the retry's register() would be silently rejected, leaving
    // TraceRoot's _provider pointing at a provider that is not the one
    // exporting spans and permanently un-releasable by shutdown() (gated on
    // isActiveGlobalDelegate).
    assert.notStrictEqual(
      afterRetry,
      rolledBack,
      'the retried initialize() must install a fresh active provider, not stay bound to a stale one',
    );
  });

  it('shutdown() after a wiring throw fully releases the global so a later initialize() works', async () => {
    delete process.env.TRACEROOT_API_KEY;

    try {
      TraceRoot.initialize({ ...BASE, instrumentModules: { claudeAgentSDK: {} } });
    } catch {
      // expected
    }
    const afterThrow = activeProvider();

    // Whether or not initialize()'s own rollback already released the global
    // slots, shutdown() must be safe to call afterwards and must not leave
    // anything behind that blocks a subsequent initialize().
    await assert.doesNotReject(() => TraceRoot.shutdown());

    TraceRoot.initialize({ ...BASE, apiKey: 'trk_after_shutdown' });
    assert.equal(TraceRoot.isInitialized(), true);
    const clean = activeProvider();
    assert.notStrictEqual(
      clean,
      afterThrow,
      'post-shutdown initialize() must install a fresh provider',
    );
  });
});

describe('beforeExit listener hygiene across init/shutdown cycles', () => {
  it('does not accumulate a beforeExit listener per initialize() that shutdown() never removes', async () => {
    delete process.env.TRACEROOT_API_KEY;
    const before = process.listenerCount('beforeExit');

    // Three full, clean lifecycles. Each successful initialize() registers a
    // process.once('beforeExit', ...) flush hook; shutdown() must remove the one
    // it installed.
    for (let i = 0; i < 3; i++) {
      TraceRoot.initialize({ ...BASE, apiKey: `trk_cycle_${i}` });
      await TraceRoot.shutdown();
    }

    const after = process.listenerCount('beforeExit');
    assert.equal(
      after,
      before,
      `shutdown() should reclaim the beforeExit flush hook it added; leaked ${after - before} listener(s)`,
    );
  });
});
