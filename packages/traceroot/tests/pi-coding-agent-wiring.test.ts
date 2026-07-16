/**
 * Wiring-only coverage for instrumentModules.piCodingAgent, now that
 * @traceroot-ai/pi has been folded in-tree (packages/traceroot/src/pi/):
 * initialize() calls instrumentPiCodingAgent() (./src/pi/instrumentation.ts)
 * directly, with no dynamic require(), no optional-peer-package diagnosis,
 * and no apiKey/baseUrl threading (this integration builds no export
 * pipeline of its own — see pi/config.ts's module header).
 *
 * This file replaces the old pi-coding-agent.test.ts (deleted) and
 * pi-broken-transitive-dep.test.ts (deleted), whose subject — the
 * dynamic-`require('@traceroot-ai/pi')` machinery, the "not installed" vs.
 * "installed but broken" MODULE_NOT_FOUND classification, and
 * missing-export detection — no longer exists. It is deliberately narrow:
 * proving the WIRING path (unwrap, dispatch, warn-don't-throw) is correct.
 * Deep capture-toggle span behavior belongs to a later phase's moved pi
 * behavioral suite, and cross-package no-mocks span assertions already live
 * in pi-coding-agent-integration.test.ts.
 */
import { afterEach, describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { TraceRoot, _resetForTesting } from '../src/traceroot';

// A minimal, self-contained fake `import * as pi from
// '@earendil-works/pi-coding-agent'` namespace: a class exposing
// prompt/subscribe/dispose on its prototype, which is exactly the structural
// surface instrumentPiCodingAgent() patches (see src/pi/instrumentation.ts).
// A fresh class per call, since instrumentPiCodingAgent()'s wrap-once guard
// is stamped on AgentSession.prototype itself.
function makePiModule() {
  class FakeAgentSession {
    sessionId = 'wiring-test-sess';
    async prompt(_text: string, _options?: unknown): Promise<void> {}
    subscribe(_listener: (event: { type: string }) => void): () => void {
      return () => {};
    }
    dispose(): void {}
  }
  return { AgentSession: FakeAgentSession };
}

const BASE = {
  apiKey: 'trk_pi_wiring',
  baseUrl: 'http://127.0.0.1:9',
  disableBatch: true as const,
  gitRepo: 'traceroot-ai/traceroot-ts',
  gitRef: 'pi-wiring-ref',
};

afterEach(() => {
  _resetForTesting();
});

describe('instrumentModules.piCodingAgent wiring', () => {
  it('patches AgentSession.prototype.prompt for the bare module ref form', () => {
    const pi = makePiModule();
    const originalPrompt = pi.AgentSession.prototype.prompt;

    TraceRoot.initialize({ ...BASE, instrumentModules: { piCodingAgent: pi } });

    assert.equal(TraceRoot.isInitialized(), true);
    assert.notStrictEqual(
      pi.AgentSession.prototype.prompt,
      originalPrompt,
      'the bare module ref form must patch AgentSession.prototype.prompt',
    );
  });

  it('accepts the { module, config } wrapper form and still patches the prototype', () => {
    const pi = makePiModule();
    const originalPrompt = pi.AgentSession.prototype.prompt;

    TraceRoot.initialize({
      ...BASE,
      instrumentModules: {
        piCodingAgent: { module: pi, config: { captureContent: false } },
      },
    });

    assert.equal(TraceRoot.isInitialized(), true);
    assert.notStrictEqual(
      pi.AgentSession.prototype.prompt,
      originalPrompt,
      'the { module, config } wrapper must be unwrapped and still patch the prototype',
    );
  });

  it('leaves the prototype untouched when piCodingAgent is null', () => {
    const pi = makePiModule();
    const originalPrompt = pi.AgentSession.prototype.prompt;

    // Passing pi only as a bystander module so we have something to assert
    // against; piCodingAgent itself is explicitly null.
    TraceRoot.initialize({ ...BASE, instrumentModules: { piCodingAgent: null } });

    assert.equal(TraceRoot.isInitialized(), true);
    assert.strictEqual(
      pi.AgentSession.prototype.prompt,
      originalPrompt,
      'a null piCodingAgent must never touch any AgentSession prototype',
    );
  });

  it('leaves the prototype untouched when piCodingAgent is absent', () => {
    const pi = makePiModule();
    const originalPrompt = pi.AgentSession.prototype.prompt;

    TraceRoot.initialize({ ...BASE, instrumentModules: {} });

    assert.equal(TraceRoot.isInitialized(), true);
    assert.strictEqual(
      pi.AgentSession.prototype.prompt,
      originalPrompt,
      'an absent piCodingAgent key must never touch any AgentSession prototype',
    );
  });

  it('warns but leaves isInitialized() true for a misshapen module (pi warns, does not throw)', () => {
    // No AgentSession.prototype.prompt/subscribe -- instrumentPiCodingAgent()
    // detects this and warns rather than throwing, deliberately unlike
    // wireClaudeAgentSDKInstrumentation() (which throws on a missing query()).
    const misshapen = { AgentSession: class {} };

    const warnings: string[] = [];
    const originalWarn = console.warn;
    console.warn = (...args: unknown[]) => {
      warnings.push(args.join(' '));
    };
    try {
      TraceRoot.initialize({ ...BASE, instrumentModules: { piCodingAgent: misshapen } });
    } finally {
      console.warn = originalWarn;
    }

    assert.equal(
      TraceRoot.isInitialized(),
      true,
      'a misshapen piCodingAgent must warn-and-noop, not abort initialize()',
    );
    assert.ok(
      warnings.some((w) => w.includes('instrumentation disabled')),
      'expected a warning that instrumentation was disabled for the misshapen module',
    );
  });

  it('wires cleanly alongside another instrumentModules entry in the same call', () => {
    const pi = makePiModule();
    const originalPrompt = pi.AgentSession.prototype.prompt;
    const claude = {
      query(_params: unknown): AsyncIterable<unknown> {
        return { async *[Symbol.asyncIterator]() {} };
      },
    };
    const originalQuery = claude.query;

    TraceRoot.initialize({
      ...BASE,
      instrumentModules: { claudeAgentSDK: claude, piCodingAgent: pi },
    });

    assert.equal(TraceRoot.isInitialized(), true);
    assert.notStrictEqual(
      pi.AgentSession.prototype.prompt,
      originalPrompt,
      'piCodingAgent must still be wired alongside claudeAgentSDK',
    );
    assert.notStrictEqual(
      claude.query,
      originalQuery,
      'claudeAgentSDK must still be wired alongside piCodingAgent',
    );
  });
});
