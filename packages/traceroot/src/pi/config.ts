/**
 * Configuration resolution for the Pi coding agent instrumentation.
 *
 * Mirrors the config-resolution contract shared by every traceroot-ts
 * integration: explicit config wins, then env vars, then a built-in default.
 * There is no export-pipeline configuration here (no apiKey/baseUrl/exporter
 * override) — this in-tree integration always gets its tracer from the
 * globally-registered OTel provider core sets up, never builds its own.
 */

// OTel tracer scope name shipped in every exported Pi span — the tracer's
// only remaining use (the old x-traceroot-sdk-name header use died with the
// standalone package). Named distinctly from core's own SDK_NAME
// (processor.ts, 'traceroot-ts') to avoid same-name shadowing between the two.
// Value follows the same scoped convention as the Claude Agent SDK integration
// (claude-agent-sdk.ts's '@traceroot-ai/claude-agent-sdk'), one style for both.
export const TRACER_NAME = '@traceroot-ai/pi-coding-agent';

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
