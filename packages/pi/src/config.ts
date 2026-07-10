/**
 * Configuration resolution for the Pi coding agent instrumentation.
 *
 * Mirrors the config-resolution contract shared by every traceroot-ts
 * integration: explicit config wins, then env vars, then a default base URL.
 * A missing API key never throws — the caller gets back an uninstrumented
 * sdk and a console warning instead.
 */
import type { SpanExporter } from '@opentelemetry/sdk-trace-base';

export const SDK_NAME = 'traceroot-pi';

const DEFAULT_BASE_URL = 'https://app.traceroot.ai';

export interface PiInstrumentationConfig {
  /** TraceRoot API key. Falls back to TRACEROOT_API_KEY. */
  apiKey?: string;
  /** TraceRoot base URL. Falls back to TRACEROOT_HOST_URL, then the hosted default. */
  baseUrl?: string;
  /** Capture prompt/response text as input.value/output.value on AGENT and LLM spans. Default true. */
  captureContent?: boolean;
  /** Capture tool call args/results as input.value/output.value on TOOL spans. Default true. */
  captureToolIo?: boolean;
  /**
   * Test-only span exporter override, injected before any real OTLP network
   * client is constructed. Not part of the public API contract.
   */
  _spanExporter?: SpanExporter;
}

export interface ResolvedPiInstrumentationConfig {
  apiKey: string | undefined;
  baseUrl: string;
  captureContent: boolean;
  captureToolIo: boolean;
  spanExporterOverride: SpanExporter | undefined;
}

export function resolveConfig(config?: PiInstrumentationConfig): ResolvedPiInstrumentationConfig {
  const apiKey = config?.apiKey ?? process.env.TRACEROOT_API_KEY;
  const rawBaseUrl = config?.baseUrl ?? process.env.TRACEROOT_HOST_URL ?? DEFAULT_BASE_URL;
  const baseUrl = rawBaseUrl.replace(/\/+$/, '');
  const captureContent = config?.captureContent ?? true;
  const captureToolIo = config?.captureToolIo ?? true;
  return {
    apiKey,
    baseUrl,
    captureContent,
    captureToolIo,
    spanExporterOverride: config?._spanExporter,
  };
}
