/**
 * Lens: config-resolution-and-env-vars.
 *
 * Probes src/config.ts's resolveConfig() precedence and normalization rules
 * in isolation: explicit config wins over TRACEROOT_API_KEY/TRACEROOT_HOST_URL
 * env vars, which in turn win over the hosted default; baseUrl's
 * trailing-slash stripping; and the deliberate "" vs undefined distinction
 * for apiKey — plus one end-to-end check that an explicit empty-string
 * apiKey still disables instrumentation exactly like a missing one, proving
 * the safety net lives in instrumentPiCodingAgent() and not resolveConfig()
 * itself.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import type { ExportResult } from '@opentelemetry/core';
import { ExportResultCode } from '@opentelemetry/core';
import type { ReadableSpan, SpanExporter } from '@opentelemetry/sdk-trace-base';
import { resolveConfig } from '../src/config';
import { instrumentPiCodingAgent } from '../src/instrumentation';
import type { AgentEvent } from '../src/types';

// Copied locally — no shared state across test files, matching every other
// tests/*.test.ts file's explicit convention.
class CapturingExporter implements SpanExporter {
  readonly spans: ReadableSpan[] = [];
  export(spans: ReadableSpan[], resultCallback: (result: ExportResult) => void): void {
    this.spans.push(...spans);
    resultCallback({ code: ExportResultCode.SUCCESS });
  }
  async shutdown(): Promise<void> {}
}

// Fresh class per rig, not a shared module-level class — instrumentPiCodingAgent
// patches AgentSession.prototype directly, so reusing one class across tests
// would stack multiple wrap layers onto the same prototype method.
function makeFakeSessionClass() {
  return class FakeAgentSession {
    sessionId = 'sess-1';
    private listeners: Array<(event: AgentEvent) => void> = [];
    async prompt(_text: string, _options?: unknown): Promise<void> {}
    subscribe(listener: (event: AgentEvent) => void): () => void {
      this.listeners.push(listener);
      return () => {
        this.listeners = this.listeners.filter((l) => l !== listener);
      };
    }
    emit(event: AgentEvent): void {
      for (const listener of this.listeners) listener(event);
    }
  };
}

// TRACEROOT_API_KEY/TRACEROOT_HOST_URL are process-global mutable state — every
// test that touches them must save + restore in a finally block so test order
// or accidental parallelism never leaks env vars into unrelated tests/files.
function withEnv(vars: Record<string, string | undefined>, fn: () => void): void {
  const saved: Record<string, string | undefined> = {};
  for (const key of Object.keys(vars)) saved[key] = process.env[key];
  try {
    for (const [key, value] of Object.entries(vars)) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
    fn();
  } finally {
    for (const [key, value] of Object.entries(saved)) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  }
}

test('explicit config.apiKey and config.baseUrl win over TRACEROOT_API_KEY/TRACEROOT_HOST_URL env vars', () => {
  withEnv({ TRACEROOT_API_KEY: 'env-key', TRACEROOT_HOST_URL: 'https://env.example.com' }, () => {
    const resolved = resolveConfig({
      apiKey: 'explicit-key',
      baseUrl: 'https://explicit.example.com',
    });
    assert.equal(resolved.apiKey, 'explicit-key');
    assert.equal(resolved.baseUrl, 'https://explicit.example.com');
  });
});

test('TRACEROOT_API_KEY/TRACEROOT_HOST_URL env vars are used as a fallback when config omits apiKey/baseUrl entirely', () => {
  withEnv({ TRACEROOT_API_KEY: 'env-key', TRACEROOT_HOST_URL: 'https://env.example.com' }, () => {
    const resolved = resolveConfig({});
    assert.equal(resolved.apiKey, 'env-key');
    assert.equal(resolved.baseUrl, 'https://env.example.com');
  });
});

test('with neither config nor env vars set, apiKey resolves to undefined and baseUrl falls back to the hosted default', () => {
  withEnv({ TRACEROOT_API_KEY: undefined, TRACEROOT_HOST_URL: undefined }, () => {
    const resolved = resolveConfig({});
    assert.equal(resolved.apiKey, undefined);
    assert.equal(resolved.baseUrl, 'https://app.traceroot.ai');
  });
});

test('a single trailing slash on baseUrl is stripped', () => {
  const resolved = resolveConfig({ apiKey: 'k', baseUrl: 'https://example.com/' });
  assert.equal(resolved.baseUrl, 'https://example.com');
});

test('multiple trailing slashes on baseUrl are all stripped, not just one', () => {
  const resolved = resolveConfig({ apiKey: 'k', baseUrl: 'https://example.com///' });
  assert.equal(resolved.baseUrl, 'https://example.com');
});

test('a baseUrl with an internal path segment keeps the path and strips only the trailing slash', () => {
  const resolved = resolveConfig({ apiKey: 'k', baseUrl: 'https://example.com/api/' });
  assert.equal(resolved.baseUrl, 'https://example.com/api');
});

test('an explicit empty-string apiKey is NOT treated as missing by resolveConfig itself — ?? only catches null/undefined, not ""', () => {
  withEnv({ TRACEROOT_API_KEY: 'env-key' }, () => {
    const resolved = resolveConfig({ apiKey: '' });
    // This is a documented quirk, not a bug: resolveConfig()'s `??` deliberately
    // treats an explicit "" as a present (if useless) value rather than silently
    // substituting the env var behind the caller's back. The safety net lives one
    // layer up — see the end-to-end test below.
    assert.equal(resolved.apiKey, '');
  });
});

test('end-to-end: an explicit empty-string apiKey disables instrumentation exactly like a fully missing key, so "" never reaches the Authorization header', () => {
  const Session = makeFakeSessionClass();
  const originalPrompt = Session.prototype.prompt;
  const sdk = { AgentSession: Session };

  // instrumentPiCodingAgent()'s own `if (!resolved.apiKey)` guard is a plain
  // falsy check, so it catches "" the same way it catches undefined — the
  // broken `Authorization: Bearer ` header this could otherwise produce (see
  // provider.ts's template string) is never assembled because createTracing()
  // is never even called in this branch.
  const result = instrumentPiCodingAgent(sdk, { apiKey: '' });

  assert.equal(result, sdk);
  assert.equal(
    Session.prototype.prompt,
    originalPrompt,
    'prompt must be untouched — an empty-string apiKey must degrade exactly like a missing one',
  );
});

test('an explicit empty-string baseUrl does not silently collapse to a broken relative URL — it falls back like an unset baseUrl', () => {
  const resolved = resolveConfig({ apiKey: 'k', baseUrl: '' });
  // Unlike apiKey, baseUrl has no downstream falsy guard before it is spliced
  // into the OTLP exporter URL (`${config.baseUrl}/api/v1/public/traces` in
  // provider.ts) — an empty resolved.baseUrl would silently become a relative
  // URL and break every span export with no warning at all.
  assert.equal(resolved.baseUrl, 'https://app.traceroot.ai');
});

test('a slashes-only baseUrl ("///") also falls back to the default, not to an empty string after stripping', () => {
  // A naive truthy-check-before-stripping guard (e.g. `config?.baseUrl || ...`)
  // would let "///" through because it is non-empty pre-strip, then the
  // trailing-slash strip would still collapse it to "" afterward — the same
  // broken-relative-URL failure mode as an explicit "", just reached one step
  // later. The fix must normalize (strip, then check emptiness) before falling
  // through to the next candidate, not check truthiness on the raw string.
  const resolved = resolveConfig({ apiKey: 'k', baseUrl: '///' });
  assert.equal(resolved.baseUrl, 'https://app.traceroot.ai');
});
