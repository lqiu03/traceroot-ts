/**
 * Lens: shared-mode requires no apiKey.
 *
 * Regression for the config-plumbing P0: when a real global TracerProvider is
 * already registered (shared mode), instrumentPiCodingAgent() never builds an
 * OTLP exporter, so it needs no apiKey at all. The missing-apiKey safety net
 * must therefore gate only the PRIVATE-provider path that actually constructs
 * an OTLP exporter, never the shared path. Before the gate was moved after the
 * shared-vs-private decision, a shared-mode call with no apiKey bailed early
 * and left AgentSession.prototype.prompt unpatched — silently producing zero
 * spans even though a fully valid export pipeline was available.
 *
 * This is the exact shape TraceRoot.initialize() hands through: it registers
 * its own provider first, then wires pi in shared mode, and a host that only
 * passed apiKey programmatically (never via TRACEROOT_API_KEY) would otherwise
 * see nothing traced.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { trace } from '@opentelemetry/api';
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import { InMemorySpanExporter, SimpleSpanProcessor } from '@opentelemetry/sdk-trace-base';
import { instrumentPiCodingAgent } from '../src/instrumentation';
import { assistantMessage, attrs, makeFakeSessionClass } from './test-helpers';

// The whole point of this regression is a genuinely-absent apiKey, so pin
// TRACEROOT_API_KEY to unset for the duration regardless of the ambient env.
function withoutApiKeyEnv(fn: () => Promise<void> | void): Promise<void> | void {
  const saved = process.env.TRACEROOT_API_KEY;
  delete process.env.TRACEROOT_API_KEY;
  try {
    return fn();
  } finally {
    if (saved === undefined) delete process.env.TRACEROOT_API_KEY;
    else process.env.TRACEROOT_API_KEY = saved;
  }
}

test('shared mode instruments prompt() and exports its span tree with NO apiKey at all', async () => {
  const exporter = new InMemorySpanExporter();
  const provider = new NodeTracerProvider();
  provider.addSpanProcessor(new SimpleSpanProcessor(exporter));
  provider.register();
  try {
    await withoutApiKeyEnv(async () => {
      const Session = makeFakeSessionClass();
      const sdk = { AgentSession: Session };
      const originalPrompt = Session.prototype.prompt;

      // No apiKey, no _spanExporter — exactly what initialize() forwards in
      // shared mode when neither the option nor TRACEROOT_API_KEY is set.
      instrumentPiCodingAgent(sdk, {});

      assert.notEqual(
        Session.prototype.prompt,
        originalPrompt,
        'prompt() MUST be patched in shared mode even without an apiKey — a real global provider owns export',
      );

      const session = new Session();
      await session.prompt('trace me with no api key');
      session.emit({ type: 'agent_start' });
      session.emit({
        type: 'agent_end',
        messages: [assistantMessage({ content: [{ type: 'text', text: 'traced anyway' }] })],
        willRetry: false,
      });

      const rootSpan = exporter
        .getFinishedSpans()
        .find((s) => attrs(s)['openinference.span.kind'] === 'AGENT');
      assert.ok(
        rootSpan,
        'the AGENT root span must export through the shared provider with no apiKey configured',
      );
      assert.equal(attrs(rootSpan!)['input.value'], 'trace me with no api key');
      assert.equal(attrs(rootSpan!)['output.value'], 'traced anyway');
    });
  } finally {
    trace.disable();
  }
});

// Companion negative control: the PRIVATE path (no global provider registered)
// with no apiKey and no exporter override still cannot build an export
// pipeline, so it must keep degrading to an untouched no-op rather than
// pretending to instrument. Confirms the gate was scoped, not deleted.
test('private mode with no apiKey and no exporter override still degrades to a no-op', () => {
  withoutApiKeyEnv(() => {
    const Session = makeFakeSessionClass();
    const originalPrompt = Session.prototype.prompt;
    const sdk = { AgentSession: Session };

    const result = instrumentPiCodingAgent(sdk, {}); // nothing registered globally -> private

    assert.equal(result, sdk);
    assert.equal(
      Session.prototype.prompt,
      originalPrompt,
      'prompt must stay untouched when no pipeline (shared provider or apiKey) is available',
    );
  });
});
