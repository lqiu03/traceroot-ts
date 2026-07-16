/**
 * Configuration resolution for the Pi coding agent instrumentation.
 *
 * Mirrors the config-resolution contract shared by every traceroot-ts
 * integration: explicit config wins, then env vars, then a built-in default.
 * There is no export-pipeline configuration here (no apiKey/baseUrl/exporter
 * override) — this in-tree integration always gets its tracer from the
 * globally-registered OTel provider core sets up, never builds its own.
 */

export const SDK_NAME = 'traceroot-pi';

export interface PiInstrumentationConfig {
  /** Capture prompt/response text as input.value/output.value on AGENT and LLM spans. Default true. */
  captureContent?: boolean;
  /** Capture tool call args/results as input.value/output.value on TOOL spans. Default true. */
  captureToolIo?: boolean;
}

export interface ResolvedPiInstrumentationConfig {
  captureContent: boolean;
  captureToolIo: boolean;
}

export function resolveConfig(config?: PiInstrumentationConfig): ResolvedPiInstrumentationConfig {
  const captureContent = config?.captureContent ?? true;
  const captureToolIo = config?.captureToolIo ?? true;
  return {
    captureContent,
    captureToolIo,
  };
}
