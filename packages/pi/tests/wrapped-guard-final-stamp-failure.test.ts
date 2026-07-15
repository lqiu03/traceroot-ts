/**
 * Lens: rollback when the FINAL wrap-once stamp itself throws.
 *
 * wrapped-guard-setup-failure.test.ts covers a failure DURING the method
 * patching (a throwing `steer` probe). This file covers the complementary
 * case the old code got wrong: every method patch succeeds, and then the very
 * last step — Object.defineProperty(proto, WRAPPED, ...) that marks the
 * prototype wrapped — throws (the prototype was frozen/sealed in between, or a
 * conflicting non-configurable WRAPPED-keyed property already exists).
 *
 * Originally that stamp sat OUTSIDE the try/catch, so its throw escaped
 * rollback: the prototype was left fully patched but UNSTAMPED. The next
 * instrumentPiCodingAgent() call then saw no guard, re-patched the already-
 * patched prompt/steer/followUp/dispose, and every event emitted DUPLICATE
 * spans forever. This proves the stamp now lives inside the try, so its failure
 * both surfaces as a clear install error AND rolls the method patches back to
 * their originals — leaving the prototype exactly as found.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { trace } from '@opentelemetry/api';
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import { instrumentPiCodingAgent } from '../src/instrumentation';

// Same registry key the instrumentation stamps on AgentSession.prototype.
const WRAPPED = Symbol.for('traceroot.pi_coding_agent.wrapped');

test('a throw from the final WRAPPED stamp rolls every method patch back to its original', () => {
  // Shared-mode path (a real global provider registered) so the install does
  // not build a private provider or a beforeExit hook — keeps this focused on
  // the stamp-failure rollback, not flush plumbing.
  const provider = new NodeTracerProvider();
  provider.register();
  try {
    // A fully-patchable AgentSession: prompt/subscribe/steer/followUp/dispose
    // all present, so EVERY method patch below succeeds and the install reaches
    // its final stamping step with the prototype fully patched.
    class FullSession {
      sessionId = 's';
      prompt(): void {}
      subscribe(): () => void {
        return () => {};
      }
      steer(): void {}
      followUp(): void {}
      dispose(): void {}
    }
    const proto = FullSession.prototype as unknown as Record<PropertyKey, unknown>;

    // Booby-trap the final stamp: pre-define WRAPPED as a NON-configurable,
    // non-writable data property whose value is falsy. The falsy value lets the
    // early `if (proto[WRAPPED])` guard pass (so setup proceeds and all method
    // patches apply), but the closing Object.defineProperty(proto, WRAPPED,
    // { value: true, ... }) cannot redefine a non-configurable property to a new
    // value — it throws a TypeError, standing in for a frozen/sealed prototype
    // or a genuine WRAPPED-key collision at that exact final step.
    Object.defineProperty(proto, WRAPPED, {
      value: false,
      configurable: false,
      writable: false,
      enumerable: false,
    });

    const originalPrompt = proto.prompt;
    const originalSteer = proto.steer;
    const originalFollowUp = proto.followUp;
    const originalDispose = proto.dispose;
    const sdk = { AgentSession: FullSession };

    // (a) The stamp failure must surface as a clear, rethrown install error.
    assert.throws(
      () => instrumentPiCodingAgent(sdk, { apiKey: 'test-key' }),
      /failed to install/i,
      'a throw from the final WRAPPED stamp must surface as an instrumentation-install error',
    );

    // (b) Every method patch must be rolled back to its ORIGINAL function, even
    // though all of them succeeded and only the trailing stamp failed. Without
    // rollback covering the stamp, these would remain wrapped while the guard
    // is unset — the exact state that double-instruments on the next call.
    assert.equal(proto.prompt, originalPrompt, 'prompt must be rolled back to the original');
    assert.equal(proto.steer, originalSteer, 'steer must be rolled back to the original');
    assert.equal(proto.followUp, originalFollowUp, 'followUp must be rolled back to the original');
    assert.equal(proto.dispose, originalDispose, 'dispose must be rolled back to the original');
  } finally {
    trace.disable();
  }
});
