/**
 * Lens: shared-mode tracer survives a global-provider swap.
 *
 * Regression for the P0 where pi's shared-mode tracer was resolved ONCE at
 * wrap time and closed over forever. TraceRoot.shutdown() (and anything else
 * that calls trace.disable()) swaps the OTel API's internal ProxyTracerProvider
 * for a brand-new instance rather than mutating the old one, so a tracer
 * captured before the swap stays bound to the old, now-detached provider —
 * every subsequent span goes dark, permanently, after a
 * shutdown()/initialize() cycle, with no recovery path.
 *
 * These tests reproduce the plan's exact repro: register provider A, obtain a
 * shared-mode tracer, export a span (lands in A); run shutdown()'s exact
 * disable sequence; register provider B; export a span on the SAME captured
 * handle — it must now route to B.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { context, propagation, trace } from '@opentelemetry/api';
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import { InMemorySpanExporter, SimpleSpanProcessor } from '@opentelemetry/sdk-trace-base';
import { createTracing } from '../src/provider';
import { resolveConfig } from '../src/config';
import { instrumentPiCodingAgent } from '../src/instrumentation';
import { assistantMessage, attrs, makeFakeSessionClass } from './test-helpers';

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

test('a shared-mode tracer handle re-resolves onto a provider registered after a shutdown()/disable() cycle', async () => {
  const a = registerProvider();
  let b: { provider: NodeTracerProvider; exporter: InMemorySpanExporter } | undefined;
  try {
    // Shared mode: no _spanExporter override, a real global provider (A) live.
    const handle = createTracing(resolveConfig({ apiKey: 'test-key' }));
    assert.equal(handle.ownsProvider, false, 'sanity: provider A must put us in shared mode');

    const first = handle.tracer.startSpan('span-before-cycle');
    first.end();
    assert.deepEqual(
      a.exporter.getFinishedSpans().map((s) => s.name),
      ['span-before-cycle'],
      'sanity: the pre-cycle span must land in provider A',
    );

    // TraceRoot.shutdown()'s exact sequence: tear down A, then reset the OTel
    // globals (which swaps the internal ProxyTracerProvider wholesale).
    await runShutdownSequence(a.provider);

    // A fresh provider takes over the global slot, exactly as a subsequent
    // TraceRoot.initialize() would install.
    b = registerProvider();

    // The SAME captured handle from before the cycle must now route to B.
    const second = handle.tracer.startSpan('span-after-cycle');
    second.end();

    assert.deepEqual(
      b.exporter.getFinishedSpans().map((s) => s.name),
      ['span-after-cycle'],
      'the captured shared-mode handle must re-resolve onto provider B after the disable() cycle',
    );
  } finally {
    trace.disable();
  }
});

test('an already-instrumented AgentSession keeps exporting across a shutdown()/initialize() cycle in shared mode', async () => {
  const a = registerProvider();
  let b: { provider: NodeTracerProvider; exporter: InMemorySpanExporter } | undefined;
  try {
    const Session = makeFakeSessionClass();
    const sdk = { AgentSession: Session };
    instrumentPiCodingAgent(sdk, { apiKey: 'test-key' }); // shared mode, wraps the prototype once

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
