/**
 * Lens: tracer re-resolution across a shutdown()/initialize() provider swap.
 *
 * Ported from packages/pi/tests/provider-shared-mode-behavior.test.ts, which
 * is otherwise deleted for this in-tree migration (it mostly probed the old
 * standalone package's private-vs-shared-provider mode detection — apiKey,
 * _spanExporter, hasRealGlobalProvider() — none of which exist in this
 * integration; see packages/traceroot/src/pi/config.ts's own header: there
 * is no export-pipeline configuration here at all, since core always
 * guarantees a real, globally-registered OTel provider before
 * instrumentPiCodingAgent() ever runs). This one scenario survives because it
 * proves something still true and still load-bearing: the collapsed
 * tracer-acquisition approach (see instrumentation.ts's
 * createReresolvingTracer) must keep exporting through a provider SWAP, not
 * just through the single provider that was live at wrap time.
 *
 * TraceRoot.shutdown() (and anything else that calls trace.disable()) swaps
 * the OTel API's internal ProxyTracerProvider for a brand-new instance rather
 * than mutating the old one, so a tracer captured once at wrap time and
 * closed over forever would stay bound to the old, now-detached provider —
 * every subsequent span would go dark permanently after a
 * shutdown()/initialize() cycle, with no recovery path (the Symbol.for()
 * wrap-once guard blocks re-instrumenting to pick up a fresh tracer).
 * createReresolvingTracer instead re-resolves through the global `trace`
 * facade on every span-open, so this reproduces the exact repro: register
 * provider A, instrument, run once (lands in A), run the shutdown reset
 * sequence, register provider B, run again on the SAME already-instrumented
 * session (must land in B).
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { context, propagation, trace } from '@opentelemetry/api';
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import { InMemorySpanExporter, SimpleSpanProcessor } from '@opentelemetry/sdk-trace-base';
import { instrumentPiCodingAgent } from '../src/pi/instrumentation';
import { assistantMessage, attrs, makeFakeSessionClass } from './pi-test-helpers';

function registerProvider(): { provider: NodeTracerProvider; exporter: InMemorySpanExporter } {
  const exporter = new InMemorySpanExporter();
  const provider = new NodeTracerProvider();
  provider.addSpanProcessor(new SimpleSpanProcessor(exporter));
  provider.register();
  return { provider, exporter };
}

// Exactly the three process-wide resets TraceRoot.shutdown() performs, after
// tearing down its own provider.
async function runShutdownSequence(provider: NodeTracerProvider): Promise<void> {
  await provider.shutdown();
  trace.disable();
  context.disable();
  propagation.disable();
}

test('an already-instrumented AgentSession keeps exporting across a shutdown()/initialize() cycle', async () => {
  const a = registerProvider();
  let b: { provider: NodeTracerProvider; exporter: InMemorySpanExporter } | undefined;
  try {
    const Session = makeFakeSessionClass();
    const sdk = { AgentSession: Session };
    instrumentPiCodingAgent(sdk, {}); // wraps the prototype once, tracer re-resolved per span

    // First run lands in A.
    const s1 = new Session();
    await s1.prompt('first run');
    s1.emit({ type: 'agent_start' });
    s1.emit({
      type: 'agent_end',
      messages: [assistantMessage({ content: [{ type: 'text', text: 'done A' }] })],
      willRetry: false,
    });
    assert.ok(
      a.exporter.getFinishedSpans().some((s) => attrs(s)['openinference.span.kind'] === 'AGENT'),
      'sanity: the first run must export through provider A',
    );

    // Simulate TraceRoot.shutdown() -> TraceRoot.initialize(): the prototype is
    // still wrapped (Symbol.for() guard blocks re-instrumentation), so recovery
    // depends entirely on the captured tracer re-resolving, not re-wrapping.
    await runShutdownSequence(a.provider);
    b = registerProvider();

    const s2 = new Session();
    await s2.prompt('second run');
    s2.emit({ type: 'agent_start' });
    s2.emit({
      type: 'agent_end',
      messages: [assistantMessage({ content: [{ type: 'text', text: 'done B' }] })],
      willRetry: false,
    });

    const rootB = b.exporter
      .getFinishedSpans()
      .find((s) => attrs(s)['openinference.span.kind'] === 'AGENT');
    assert.ok(
      rootB,
      'the second run must export through provider B after the cycle — no silent black hole',
    );
    assert.equal(attrs(rootB!)['input.value'], 'second run');
  } finally {
    trace.disable();
  }
});
