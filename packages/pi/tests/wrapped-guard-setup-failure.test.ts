/**
 * Lens: wrap-once guard ordering under a mid-setup failure.
 *
 * instrumentPiCodingAgent() stamps AgentSession.prototype with its wrap-once
 * guard. If that stamp is applied BEFORE the pipeline/patch setup has finished
 * and any step of setup then throws (a bad option, an exotic duck-type check
 * that throws), the prototype is left permanently marked "wrapped" while the
 * SDK is never actually patched — so every later instrumentPiCodingAgent()
 * call is silently rejected with "config ignored" instead of the real setup
 * failure ever surfacing. This proves the stamp lands only after setup fully
 * succeeds, the failure surfaces as a clear error, and a partial install is
 * rolled back rather than left half-applied (which a later retry would then
 * double-wrap).
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { trace } from '@opentelemetry/api';
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import { instrumentPiCodingAgent } from '../src/instrumentation';

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
      () => instrumentPiCodingAgent(sdk, { apiKey: 'test-key' }),
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
