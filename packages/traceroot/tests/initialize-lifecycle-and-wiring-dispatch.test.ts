/**
 * Tests for the core initialize()/shutdown() lifecycle and the
 * wireInstrumentations() dispatch that TraceRoot.initialize({ instrumentModules })
 * calls into:
 *
 *  - initialize() atomicity when an instrumentation-wiring step throws (the
 *    claudeAgentSDK / openaiAgents paths throw on a misshaped module, unlike the
 *    pi path which warns-and-noops) — regression coverage for a bug where a throw
 *    left the just-registered provider orphaned on the global trace/context/
 *    propagation slots, with _isInitialized stuck false so a retry silently lost
 *    the first-write-wins race instead of recovering.
 *  - the blast radius of one bad instrumentModule on its siblings.
 *  - a double initialize() (no shutdown between) with a different instrumentModules
 *    config — the second config is dropped (documented Already-initialized guard).
 *  - both piCodingAgent AND claudeAgentSDK passed in one initialize() call.
 *  - beforeExit listener hygiene across repeated init/shutdown cycles — regression
 *    coverage for a leak where shutdown() never removed the listener initialize()
 *    installed, so each cycle left one behind permanently.
 *
 * These drive the REAL initialize() end to end (the pi path calls straight into
 * the in-tree instrumentPiCodingAgent(), which patches a fake AgentSession's
 * prototype -- that patch is the observable used to prove pi wiring actually
 * ran). A local, unroutable baseUrl keeps the OTLP exporter from ever touching
 * the network.
 */
import { afterEach, describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { trace } from '@opentelemetry/api';
import { TraceRoot, _resetForTesting } from '../src/traceroot';
import { wireInstrumentations } from '../src/instrumentation';

// The concrete provider the OTel global proxy currently delegates to — i.e. the
// provider that spans actually route to right now, regardless of what TraceRoot's
// private _provider field points at. Reading it the same way traceroot.ts's own
// isActiveGlobalDelegate() does, without reaching into module-private state.
function activeProvider(): unknown {
  const p = trace.getTracerProvider() as { getDelegate?: () => unknown };
  return typeof p.getDelegate === 'function' ? p.getDelegate() : p;
}

// A fake `import * as pi from '@earendil-works/pi-coding-agent'` namespace. The
// real in-tree pi instrumentation patches AgentSession.prototype.prompt when
// wired, so a change in that method's identity is a reliable "pi was actually
// instrumented" signal without having to drive a whole agent turn.
function makePiModule() {
  class FakeAgentSession {
    sessionId = 'core-wiring-sess';
    async prompt(_text: string): Promise<void> {}
    subscribe(_listener: (event: { type: string }) => void): () => void {
      return () => {};
    }
    dispose(): void {}
  }
  return { AgentSession: FakeAgentSession };
}

function piPromptPatched(mod: ReturnType<typeof makePiModule>, original: unknown): boolean {
  return mod.AgentSession.prototype.prompt !== original;
}

// A minimally-valid @anthropic-ai/claude-agent-sdk namespace: a mutable object
// exposing query(). wireClaudeAgentSDKInstrumentation() replaces .query in place.
function makeClaudeModule() {
  return {
    query(_params: unknown): AsyncIterable<unknown> {
      return { async *[Symbol.asyncIterator]() {} };
    },
  };
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
    // dispatch throws for it (unlike the pi path, which warns-and-noops).
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

describe('wireInstrumentations() dispatch — sibling isolation', () => {
  it('throws a clear diagnostic when claudeAgentSDK is passed without query()', () => {
    assert.throws(
      () => wireInstrumentations({ claudeAgentSDK: {} }),
      /does not expose query/,
      'a claudeAgentSDK ref without query() must throw an actionable error',
    );
  });

  it('a throwing claudeAgentSDK aborts the loop before a co-passed piCodingAgent is wired', () => {
    // Documents the blast radius of the atomicity gap above: because
    // wireInstrumentations() wires claudeAgentSDK (line ~226) BEFORE piCodingAgent
    // (line ~232) and does not isolate per-module failures, a bad claudeAgentSDK
    // throws out of the whole loop and a perfectly valid piCodingAgent alongside
    // it is silently left un-instrumented. (Consequence of the same
    // non-atomic wiring the first test flags — not asserting a separate contract.)
    delete process.env.TRACEROOT_API_KEY;
    const pi = makePiModule();
    const originalPrompt = pi.AgentSession.prototype.prompt;

    let threw = false;
    try {
      TraceRoot.initialize({
        ...BASE,
        instrumentModules: { claudeAgentSDK: {}, piCodingAgent: pi },
      });
    } catch {
      threw = true;
    }
    assert.equal(threw, true);
    assert.equal(
      piPromptPatched(pi, originalPrompt),
      false,
      'the valid piCodingAgent was left un-instrumented because the claudeAgentSDK throw aborted the loop',
    );
  });
});

describe('both piCodingAgent and claudeAgentSDK in one initialize() call', () => {
  it('wires both when both are valid, with no interference', () => {
    delete process.env.TRACEROOT_API_KEY;
    const pi = makePiModule();
    const originalPrompt = pi.AgentSession.prototype.prompt;
    const claude = makeClaudeModule();
    const originalQuery = claude.query;

    TraceRoot.initialize({
      ...BASE,
      instrumentModules: { claudeAgentSDK: claude, piCodingAgent: pi },
    });

    assert.equal(TraceRoot.isInitialized(), true);
    assert.notStrictEqual(
      claude.query,
      originalQuery,
      'claudeAgentSDK.query must be wrapped in place',
    );
    assert.equal(
      piPromptPatched(pi, originalPrompt),
      true,
      'piCodingAgent must be instrumented alongside claudeAgentSDK',
    );
  });
});

describe('double initialize() without an intervening shutdown()', () => {
  it("drops the second call's instrumentModules (documented Already-initialized guard)", () => {
    delete process.env.TRACEROOT_API_KEY;

    // First init with no instrumentModules.
    TraceRoot.initialize({ ...BASE });
    assert.equal(TraceRoot.isInitialized(), true);

    // Second init supplies a valid piCodingAgent. The Already-initialized guard
    // returns early, so this config is intentionally NOT applied. This confirms
    // the documented contract (a second initialize() is a warned no-op), rather
    // than the alternative one might expect (the new config being merged/applied).
    const pi = makePiModule();
    const originalPrompt = pi.AgentSession.prototype.prompt;
    const warnings: string[] = [];
    const originalWarn = console.warn;
    console.warn = (...args: unknown[]) => {
      warnings.push(args.join(' '));
    };
    try {
      TraceRoot.initialize({ ...BASE, instrumentModules: { piCodingAgent: pi } });
    } finally {
      console.warn = originalWarn;
    }

    assert.equal(
      piPromptPatched(pi, originalPrompt),
      false,
      "the second initialize()'s instrumentModules must be dropped by the Already-initialized guard",
    );
    assert.ok(
      warnings.some((w) => w.includes('Already initialized')),
      'the dropped second initialize() must warn that it was skipped',
    );
  });
});

describe('beforeExit listener hygiene across init/shutdown cycles', () => {
  it('does not accumulate a beforeExit listener per initialize() that shutdown() never removes', async () => {
    delete process.env.TRACEROOT_API_KEY;
    const before = process.listenerCount('beforeExit');

    // Three full, clean lifecycles. Each successful initialize() registers a
    // process.once('beforeExit', ...) flush hook; shutdown() removes none of them.
    for (let i = 0; i < 3; i++) {
      TraceRoot.initialize({ ...BASE, apiKey: `trk_cycle_${i}` });
      await TraceRoot.shutdown();
    }

    const after = process.listenerCount('beforeExit');
    // BUG (minor): the count grows by one per cycle and is never reclaimed on
    // shutdown, so a long-running process that re-initializes repeatedly leaks
    // beforeExit listeners (and trips Node's MaxListenersExceededWarning at 11).
    assert.equal(
      after,
      before,
      `shutdown() should reclaim the beforeExit flush hook it added; leaked ${after - before} listener(s)`,
    );
  });
});
