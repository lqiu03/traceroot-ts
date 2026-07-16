/**
 * Lens: instrumentPiCodingAgent() install rollback.
 *
 * instrumentPiCodingAgent() stamps AgentSession.prototype with its wrap-once
 * guard. Two distinct failure points along that install path must both leave
 * the prototype exactly as found — never half-patched, never stamped without
 * being fully patched:
 *
 *  - A failure DURING the method patching (a throwing `steer` probe below).
 *    If the wrap-once stamp were applied BEFORE setup finishes and setup then
 *    throws, the prototype is left permanently marked "wrapped" while the SDK
 *    is never actually patched — so every later instrumentPiCodingAgent()
 *    call is silently rejected with "config ignored" instead of the real
 *    setup failure ever surfacing.
 *
 *  - A failure from the FINAL wrap-once stamp itself, after every method
 *    patch has already succeeded (Object.defineProperty(proto, WRAPPED, ...)
 *    throwing because the prototype was frozen/sealed in between, or a
 *    conflicting non-configurable WRAPPED-keyed property already exists).
 *    Originally that stamp sat OUTSIDE the try/catch, so its throw escaped
 *    rollback: the prototype was left fully patched but UNSTAMPED. The next
 *    instrumentPiCodingAgent() call then saw no guard, re-patched the
 *    already-patched prompt/steer/followUp/dispose, and every event emitted
 *    DUPLICATE spans forever.
 *
 * Both tests prove the stamp lands only after setup fully succeeds and lives
 * inside the try: the failure surfaces as a clear error, and a partial
 * install is rolled back rather than left half-applied (which a later retry
 * would then double-wrap).
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { trace } from '@opentelemetry/api';
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import { instrumentPiCodingAgent } from '../src/pi/instrumentation';

// Same registry key the instrumentation stamps on AgentSession.prototype.
const WRAPPED = Symbol.for('traceroot.pi_coding_agent.wrapped');

test('a setup failure after the wrap-once decision does not leave the guard stamped or the prototype half-patched', () => {
  // Register a real global provider so the install takes the shared-mode path
  // (no private provider, no beforeExit hook) — keeps this test focused purely
  // on the guard-ordering/rollback behavior, not the flush-hook plumbing.
  const provider = new NodeTracerProvider();
  provider.register();
  try {
    // An AgentSession whose `steer` getter throws the first time the
    // instrumentation probes `typeof proto.steer === 'function'` during setup,
    // standing in for ANY exception thrown partway through installing the
    // patches. prompt() is patched before that probe, so a naive install
    // leaves prompt half-wrapped when the probe throws.
    class ThrowingSteerSession {
      sessionId = 's';
      prompt(): void {}
      subscribe(): () => void {
        return () => {};
      }
      get steer(): unknown {
        throw new Error('injected setup failure while probing steer');
      }
    }
    const proto = ThrowingSteerSession.prototype as unknown as Record<PropertyKey, unknown>;
    const originalPrompt = proto.prompt;
    const sdk = { AgentSession: ThrowingSteerSession };

    // A mid-setup failure must surface as a clear, rethrown install error —
    // not be swallowed, and not be masked as a later "config ignored".
    assert.throws(
      () => instrumentPiCodingAgent(sdk, {}),
      /failed to install/i,
      'a mid-setup failure must surface as a clear instrumentation-install error',
    );

    // The wrap-once guard must NOT be stamped after a failed install: leaving
    // it true would silently reject every subsequent install attempt.
    assert.equal(
      proto[WRAPPED],
      undefined,
      'the wrap-once guard must not be stamped when install failed partway through',
    );

    // The prototype must be left exactly as found: prompt must be the ORIGINAL
    // method, not a half-installed wrapper. Otherwise a later retry re-wraps an
    // already-wrapped prompt, double-instrumenting every call.
    assert.equal(
      proto.prompt,
      originalPrompt,
      'a failed install must roll the prompt patch back, leaving the prototype unpatched',
    );
  } finally {
    trace.disable();
  }
});

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
      () => instrumentPiCodingAgent(sdk, {}),
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
