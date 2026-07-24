/**
 * Lens: per-slot global ownership in TraceRoot.shutdown().
 *
 * OpenTelemetry's global registry tracks `trace`, `context`, and `propagation`
 * as THREE SEPARATE first-write-wins slots. TraceRoot can win some and lose
 * others: a host process can win the `context` slot on its own — by calling
 * context.setGlobalContextManager() directly, independent of any tracer
 * provider — while TraceRoot still wins `trace` and `propagation`.
 *
 * The old shutdown() gated trace.disable() + context.disable() +
 * propagation.disable() behind ONE check that only verified TraceRoot still
 * owned the `trace` slot. In the scenario below that check is true (TraceRoot
 * does own `trace`), so the old code called context.disable() too — silently
 * destroying a context manager TraceRoot never owned. This proves shutdown()
 * now checks each slot's ownership independently and leaves the foreign
 * context manager intact.
 */
import { afterEach, describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { context, ROOT_CONTEXT } from '@opentelemetry/api';
import type { Context, ContextManager } from '@opentelemetry/api';
import { TraceRoot, _resetForTesting } from '../src/traceroot';

// Reads the ContextManager currently installed in OTel's global `context` slot
// via the same underscore-prefixed-by-convention runtime accessor shutdown()'s
// own ownership predicate uses (the .d.ts doesn't expose it, so bracket access).
function activeContextManager(): unknown {
  return (context as unknown as { _getContextManager?: () => unknown })._getContextManager?.();
}

// A minimal, inert host-owned ContextManager standing in for one a host process
// installed on its own (e.g. a framework registering context propagation before
// TraceRoot ever initialized). It only needs a distinct object identity to be
// observed by reference; its methods are never exercised by this test.
function makeHostContextManager(): ContextManager {
  const manager: ContextManager = {
    active(): Context {
      return ROOT_CONTEXT;
    },
    with<A extends unknown[], F extends (...args: A) => ReturnType<F>>(
      _ctx: Context,
      fn: F,
      thisArg?: ThisParameterType<F>,
      ...args: A
    ): ReturnType<F> {
      return fn.call(thisArg, ...args);
    },
    bind<T>(_ctx: Context, target: T): T {
      return target;
    },
    enable(): ContextManager {
      return manager;
    },
    disable(): ContextManager {
      return manager;
    },
  };
  return manager;
}

describe('TraceRoot.shutdown() does not disable a foreign global context manager', () => {
  afterEach(() => {
    _resetForTesting();
  });

  // RED-FIRST: a host owns ONLY the `context` slot (registered directly, before
  // TraceRoot). TraceRoot.initialize() then wins the still-free `trace` and
  // `propagation` slots but LOSES `context` (first-write-wins). The old
  // combined check keyed solely on the trace-delegate would see "TraceRoot owns
  // the global" and call context.disable() anyway, wiping the host's manager.
  it('leaves a host context manager active when TraceRoot won only trace/propagation', async () => {
    const hostManager = makeHostContextManager();
    const won = context.setGlobalContextManager(hostManager);
    assert.equal(won, true, 'sanity: the host must win the free context slot first');
    assert.strictEqual(
      activeContextManager(),
      hostManager,
      'sanity: the host context manager must own the context slot before initialize()',
    );

    TraceRoot.initialize({ apiKey: 'test-key', disableBatch: true });
    // TraceRoot owns `trace` (nothing else registered a provider) but its own
    // context manager lost the already-taken `context` slot to the host.
    assert.strictEqual(
      activeContextManager(),
      hostManager,
      'sanity: TraceRoot.initialize() must NOT have displaced the host context manager',
    );

    await TraceRoot.shutdown();

    // The core assertion: shutdown() must have left the `context` slot alone,
    // because TraceRoot never owned it. Pre-fix, context.disable() ran and this
    // is no longer the host's manager (it was reset to OTel's noop default).
    assert.strictEqual(
      activeContextManager(),
      hostManager,
      'TraceRoot.shutdown() must not disable a context manager it never registered',
    );
  });
});
