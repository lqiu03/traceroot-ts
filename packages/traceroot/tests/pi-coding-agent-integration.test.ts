/**
 * Real cross-package integration: drives the ACTUAL built @traceroot-ai/pi
 * through a REAL TraceRoot.initialize({ instrumentModules }) call — no
 * Module._load interception, no mocked pi export. This is the end-to-end proof
 * that the shared-pipeline wiring works: initialize() lazy-loads the real pi
 * package, threads its own resolved apiKey down to it (with TRACEROOT_API_KEY
 * unset, so the apiKey MUST arrive via config, not the env fallback), pi
 * detects TraceRoot's freshly-registered global provider and runs in shared
 * mode, and the spans pi produces land in TraceRoot's own pipeline WITH
 * TraceRootSpanProcessor's enrichment (environment / git repo / git ref /
 * span path) applied.
 *
 * Every other pi<->traceroot test mocks one side or the other; this one mocks
 * neither. It fails against the pre-shared-pipeline state (where pi silently
 * no-ops because it never receives an apiKey and gates instrumentation on one
 * before the shared-vs-private decision) and passes once the plumbing is in
 * place.
 */
import { afterEach, test } from 'node:test';
import assert from 'node:assert/strict';
import { trace } from '@opentelemetry/api';
import type { ReadableSpan } from '@opentelemetry/sdk-trace-base';
import { InMemorySpanExporter, SimpleSpanProcessor } from '@opentelemetry/sdk-trace-base';
import { TraceRoot, _resetForTesting } from '../src/traceroot';

// A realistically-shaped fake AgentSession module, mirroring the shape pi's
// own tests drive (a private listener array; prompt/subscribe/emit/dispose) —
// the exact structural surface `import * as pi from
// '@earendil-works/pi-coding-agent'` exposes and instrumentPiCodingAgent()
// patches. This is passed straight into instrumentModules.piCodingAgent.
interface FakeAgentEvent {
  type: string;
  [key: string]: unknown;
}
function makePiModule() {
  class FakeAgentSession {
    sessionId = 'integration-sess';
    private listeners: Array<(event: FakeAgentEvent) => void> = [];
    async prompt(_text: string): Promise<void> {}
    subscribe(listener: (event: FakeAgentEvent) => void): () => void {
      this.listeners.push(listener);
      return () => {
        this.listeners = this.listeners.filter((l) => l !== listener);
      };
    }
    emit(event: FakeAgentEvent): void {
      for (const listener of this.listeners) listener(event);
    }
    dispose(): void {
      this.listeners = [];
    }
  }
  return { AgentSession: FakeAgentSession };
}

// Minimal valid AssistantMessage (pi's src/types.ts shape) — enough for pi's
// span builders to read model/provider/usage/content without throwing.
function assistantMessage(text: string): FakeAgentEvent['message'] {
  return {
    role: 'assistant',
    content: [{ type: 'text', text }],
    api: 'anthropic-messages',
    provider: 'anthropic',
    model: 'claude-sonnet-5',
    usage: {
      input: 10,
      output: 5,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 15,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
    stopReason: 'stop',
    timestamp: 0,
  };
}

function attachInMemoryExporterToGlobalProvider(): InMemorySpanExporter {
  // TraceRoot.initialize() register()s a NodeTracerProvider as the OTel global
  // delegate. Reach that same provider and add an in-memory exporter so we can
  // observe exactly what its pipeline exports — the spans have already passed
  // through TraceRootSpanProcessor's onStart enrichment by export time.
  const proxy = trace.getTracerProvider() as { getDelegate?: () => unknown };
  const delegate = (typeof proxy.getDelegate === 'function' ? proxy.getDelegate() : proxy) as {
    addSpanProcessor(processor: SimpleSpanProcessor): void;
  };
  const exporter = new InMemorySpanExporter();
  delegate.addSpanProcessor(new SimpleSpanProcessor(exporter));
  return exporter;
}

const attrOf = (span: ReadableSpan): Record<string, unknown> =>
  span.attributes as Record<string, unknown>;
const spanKind = (span: ReadableSpan): unknown => attrOf(span)['openinference.span.kind'];

afterEach(() => {
  _resetForTesting();
});

test('the real @traceroot-ai/pi wired through TraceRoot.initialize exports enriched spans into TraceRoot own pipeline', async () => {
  // Unset so the apiKey can only reach pi via initialize()'s config plumbing,
  // never the TRACEROOT_API_KEY env fallback — this is the P0 that made pi
  // silently no-op before the fix.
  delete process.env.TRACEROOT_API_KEY;

  const pi = makePiModule();
  const Session = pi.AgentSession;

  TraceRoot.initialize({
    apiKey: 'trk_integration_key',
    // Local, unroutable endpoint: TraceRoot's own OTLP exporter must never
    // POST integration junk to the real backend. Irrelevant to pi, which runs
    // in shared mode and builds no exporter of its own.
    baseUrl: 'http://127.0.0.1:9',
    disableBatch: true,
    environment: 'integration-test',
    gitRepo: 'traceroot-ai/traceroot-ts',
    gitRef: 'integration-ref',
    instrumentModules: { piCodingAgent: pi },
  });
  assert.equal(TraceRoot.isInitialized(), true);

  const captured = attachInMemoryExporterToGlobalProvider();

  // Full prompt() -> agent_start -> (LLM turn) -> agent_end cycle, through the
  // REAL pi instrumentation initialize() just installed on Session.prototype.
  const session = new Session();
  await session.prompt('summarize the repository');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'turn_start' });
  session.emit({ type: 'message_start', message: assistantMessage('working on it') });
  session.emit({ type: 'message_end', message: assistantMessage('working on it') });
  session.emit({ type: 'turn_end', message: assistantMessage('working on it'), toolResults: [] });
  session.emit({
    type: 'agent_end',
    messages: [assistantMessage('the repository has three packages')],
    willRetry: false,
  });

  const spans = captured.getFinishedSpans();
  const rootSpan = spans.find((s) => spanKind(s) === 'AGENT');
  const llmSpan = spans.find((s) => spanKind(s) === 'LLM');

  // The core proof: the real pi package actually produced spans through
  // TraceRoot's shared provider (pre-fix this array is empty — pi no-op'd).
  assert.ok(
    rootSpan,
    'the real @traceroot-ai/pi must export an AGENT root span through TraceRoot shared provider',
  );

  // pi span shape survived the real round trip.
  assert.equal(rootSpan!.name, 'AgentSession.prompt');
  assert.equal(attrOf(rootSpan!)['session.id'], 'integration-sess');
  assert.equal(attrOf(rootSpan!)['input.value'], 'summarize the repository');
  assert.equal(attrOf(rootSpan!)['output.value'], 'the repository has three packages');
  assert.equal(attrOf(rootSpan!)['traceroot.sdk.name'], 'traceroot-pi');

  // TraceRootSpanProcessor enrichment was applied to pi's span on the way out
  // — this is what proves the spans travelled through TraceRoot's OWN pipeline,
  // not a private pi pipeline.
  assert.equal(attrOf(rootSpan!)['deployment.environment'], 'integration-test');
  assert.equal(attrOf(rootSpan!)['traceroot.git.repo'], 'traceroot-ai/traceroot-ts');
  assert.equal(attrOf(rootSpan!)['traceroot.git.ref'], 'integration-ref');
  assert.deepEqual(
    attrOf(rootSpan!)['traceroot.span.path'],
    ['AgentSession.prompt'],
    'the root span must carry TraceRootSpanProcessor span-path enrichment',
  );

  // The LLM child span routed through the same provider and nests under the
  // root, and its span path chains off the root's — proving whole-tree
  // enrichment, not just the root.
  assert.ok(llmSpan, 'the LLM child span must also route through TraceRoot provider');
  assert.equal(
    llmSpan!.parentSpanId,
    rootSpan!.spanContext().spanId,
    'the LLM span must nest under the AGENT root span',
  );
  assert.deepEqual(
    attrOf(llmSpan!)['traceroot.span.path'],
    ['AgentSession.prompt', 'claude-sonnet-5'],
    'the LLM span path must chain off the root span path via TraceRootSpanProcessor',
  );
});
