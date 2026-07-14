/**
 * OTLP tracing pipeline construction.
 *
 * Prefers an already-registered global TracerProvider (e.g. one set up by
 * a host app's TraceRoot.initialize() elsewhere in the same process) over
 * building a private one — see hasRealGlobalProvider() below. Only when no
 * real provider is globally registered does this build and use its own
 * NodeTracerProvider, and even then it is deliberately NEVER globally
 * registered (no `provider.register()`) — this package is an add-on
 * integration a host app opts into explicitly via instrumentPiCodingAgent(),
 * not the app's primary tracer. Registering globally here would risk
 * silently and permanently shutting out a host app's own OTel setup
 * (including traceroot-ts's own core package, if it initializes AFTER this
 * runs) — OTel's global provider registration is first-write-wins, and the
 * losing register() call fails silently. The returned Tracer is used
 * directly.
 */
import { trace } from '@opentelemetry/api';
import { BatchSpanProcessor, SimpleSpanProcessor } from '@opentelemetry/sdk-trace-base';
import type { SpanExporter, SpanProcessor } from '@opentelemetry/sdk-trace-base';
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-proto';
import type { Tracer } from '@opentelemetry/api';
import { SDK_NAME } from './config';
import type { ResolvedPiInstrumentationConfig } from './config';
import { SDK_VERSION } from './package-version';

export interface TracingHandle {
  tracer: Tracer;
  forceFlush: () => Promise<void>;
  /**
   * True only when this handle built and owns a private NodeTracerProvider.
   * Callers use this to decide whether they're responsible for flushing
   * (e.g. registering their own 'beforeExit' hook) — when false, a shared
   * global provider elsewhere in the process owns flush responsibility.
   */
  ownsProvider: boolean;
}

// Detects whether a REAL TracerProvider is already globally registered, as
// opposed to the default no-op the OTel API returns before anyone has
// called .register(). Deliberately duck-typed (checks for a getDelegate
// method and inspects what it returns by constructor NAME) rather than
// `instanceof ProxyTracerProvider` / `instanceof NoopTracerProvider` --
// instanceof fails across a dual-copy @opentelemetry/api dependency-tree
// duplication (two physically separate installs of the same package are
// two different class objects in memory, even with identical source),
// which is a real, empirically-reproduced hazard whenever a host app's
// dependency tree ends up with a second copy of @opentelemetry/api.
// Cross-copy detection still works via duck-typing because
// globalThis-based storage (keyed by
// Symbol.for(`opentelemetry.js.api.<major>`), see @opentelemetry/api's own
// internal/global-utils.ts) is genuinely shared across copies of the same
// major version -- only nominal instanceof checks are defeated by the
// cross-copy class-identity mismatch, not method presence or the object
// actually retrieved.
function hasRealGlobalProvider(): boolean {
  // Cast through `unknown` first: TracerProvider (the real return type of
  // trace.getTracerProvider()) doesn't structurally overlap with this
  // duck-typed shape enough for a direct assertion, even though the actual
  // runtime object (a ProxyTracerProvider) does carry getDelegate().
  const provider = trace.getTracerProvider() as unknown as {
    getDelegate?: () => { constructor: { name: string } };
  };
  if (typeof provider.getDelegate !== 'function') {
    // Not a Proxy-shaped provider at all -- some other, non-default
    // TracerProvider implementation is already active. Treat as real.
    return true;
  }
  const delegate = provider.getDelegate();
  return delegate.constructor.name !== 'NoopTracerProvider';
}

function buildPrivateTracing(config: ResolvedPiInstrumentationConfig): TracingHandle {
  const exporter: SpanExporter =
    config.spanExporterOverride ??
    new OTLPTraceExporter({
      url: `${config.baseUrl}/api/v1/public/traces`,
      headers: {
        Authorization: `Bearer ${config.apiKey}`,
        'x-traceroot-sdk-name': SDK_NAME,
        'x-traceroot-sdk-version': SDK_VERSION,
      },
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      compression: 'gzip' as any,
    });

  // A test-injected exporter never touches the network, so a SimpleSpanProcessor
  // (synchronous, no batching) keeps span assertions deterministic in tests.
  const processor: SpanProcessor = config.spanExporterOverride
    ? new SimpleSpanProcessor(exporter)
    : new BatchSpanProcessor(exporter, {
        maxExportBatchSize: 64,
        maxQueueSize: 1024,
        scheduledDelayMillis: 2000,
        exportTimeoutMillis: 4000,
      });

  const provider = new NodeTracerProvider();
  provider.addSpanProcessor(processor);

  const tracer = provider.getTracer(SDK_NAME, SDK_VERSION);

  // Flushes any pending spans without tearing the provider down — safe to
  // call repeatedly (e.g. once per 'beforeExit' in a long-lived process),
  // unlike shutdown(), which permanently disables further export.
  const forceFlush = async (): Promise<void> => {
    try {
      await provider.forceFlush();
    } catch {
      // A flush failure must never block or crash the host app's exit.
    }
  };

  return { tracer, forceFlush, ownsProvider: true };
}

export function createTracing(config: ResolvedPiInstrumentationConfig): TracingHandle {
  // A test-injected exporter means the caller wants deterministic,
  // synchronous control over exactly where spans land -- always honor it by
  // building a private pipeline, even if a real global provider happens to
  // be registered in the test process. This keeps existing tests (which all
  // pass _spanExporter via makeRig()) unaffected by this change.
  if (!config.spanExporterOverride && hasRealGlobalProvider()) {
    return {
      tracer: trace.getTracer(SDK_NAME, SDK_VERSION),
      forceFlush: async () => {}, // flush is the shared provider owner's responsibility
      ownsProvider: false,
    };
  }
  return buildPrivateTracing(config);
}
