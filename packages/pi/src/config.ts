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

// Strips trailing slashes and collapses a slashes-only (or empty) string to
// undefined so a degenerate value like "" or "///" falls through to the next
// candidate instead of surviving as a truthy-but-unusable base URL.
function normalizeBaseUrl(value: string | undefined): string | undefined {
  const stripped = value?.replace(/\/+$/, '');
  return stripped ? stripped : undefined;
}

export function resolveConfig(config?: PiInstrumentationConfig): ResolvedPiInstrumentationConfig {
  const apiKey = config?.apiKey ?? process.env.TRACEROOT_API_KEY;
  // baseUrl deliberately does not use a plain `??` chain on the raw strings:
  // unlike apiKey (whose "" is caught by a downstream falsy check before it
  // ever reaches an Authorization header, see instrumentation.ts), baseUrl has
  // no such guard before it is spliced into the OTLP exporter URL — an
  // explicit "" or "///" (or an env var interpolated to "" by a misconfigured
  // shell) must fall through to the next candidate instead of silently
  // producing a broken relative URL.
  const baseUrl =
    normalizeBaseUrl(config?.baseUrl) ??
    normalizeBaseUrl(process.env.TRACEROOT_HOST_URL) ??
    DEFAULT_BASE_URL;
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
