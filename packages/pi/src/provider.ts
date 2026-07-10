/**
 * OTLP tracing pipeline construction.
 *
 * Deliberately NOT globally registered (no `provider.register()`, unlike
 * packages/traceroot/src/traceroot.ts) — this package is an add-on
 * integration a host app opts into explicitly via instrumentPiCodingAgent(),
 * not the app's primary tracer. Registering globally would risk overwriting
 * or conflicting with a host app's own OTel setup (including traceroot-ts's
 * own core package, if also in use). The returned Tracer is used directly.
 */
import { BatchSpanProcessor, SimpleSpanProcessor } from '@opentelemetry/sdk-trace-base';
import type { SpanExporter, SpanProcessor } from '@opentelemetry/sdk-trace-base';
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-proto';
import type { Tracer } from '@opentelemetry/api';
import { SDK_NAME } from './config';
import { SDK_VERSION } from './package-version';
import type { ResolvedPiInstrumentationConfig } from './config';

export interface TracingHandle {
  tracer: Tracer;
  shutdown: () => Promise<void>;
}

export function createTracing(config: ResolvedPiInstrumentationConfig): TracingHandle {
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

  const shutdown = async (): Promise<void> => {
    try {
      await provider.shutdown();
    } catch {
      // A flush/export failure at shutdown must never block or crash the host app's exit.
    }
  };

  return { tracer, shutdown };
}
