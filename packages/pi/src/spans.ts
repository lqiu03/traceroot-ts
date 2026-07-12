/**
 * Span construction and attribute mapping for Pi coding agent traces.
 *
 * Attribute triad (matching every other traceroot-ts integration):
 * OpenInference span-kind/input/output (internal, drives UI rendering),
 * standard OTel gen_ai.* semconv, and a traceroot.* namespace for SDK
 * identity. Span path/ids_path (Mastra's live-ancestry feature) is not
 * emitted here — Pi delivers a discrete AgentEvent per lifecycle step, so
 * parent/child relationships are already explicit via OTel Context, not
 * reconstructed from a flat event stream the way Mastra's exporter does.
 */
import { SpanKind, SpanStatusCode } from '@opentelemetry/api';
import type { Context, Span, Tracer } from '@opentelemetry/api';
import { SDK_NAME } from './config';
import { SDK_VERSION } from './package-version';
import { describeToolCallSpan } from './span-name';
import { sliceSurrogateSafe } from './surrogate-safe';
import type { AgentMessage, AssistantMessage } from './types';

// Span path keys intentionally omitted — see file header. Everything else
// mirrors packages/mastra/src/exporter.ts's attribute constant shape.
const TR_ATTRIBUTES = {
  SDK_NAME: 'traceroot.sdk.name',
  SDK_VERSION: 'traceroot.sdk.version',
  COST_TOTAL: 'traceroot.pi.cost.total',
  WILL_RETRY: 'traceroot.pi.will_retry',
  FORCE_CLOSED: 'traceroot.pi.force_closed',
} as const;

// OpenInference semconv keys — internal only, not exposed in public API.
const OI_ATTRIBUTES = {
  SPAN_KIND: 'openinference.span.kind',
  INPUT_VALUE: 'input.value',
  OUTPUT_VALUE: 'output.value',
  SESSION_ID: 'session.id',
} as const;

// gen_ai semconv (standard, used by multiple platforms)
const GEN_AI_ATTRIBUTES = {
  SYSTEM: 'gen_ai.system',
  REQUEST_MODEL: 'gen_ai.request.model',
  RESPONSE_MODEL: 'gen_ai.response.model',
  USAGE_INPUT_TOKENS: 'gen_ai.usage.input_tokens',
  USAGE_OUTPUT_TOKENS: 'gen_ai.usage.output_tokens',
  CACHE_WRITE_INPUT_TOKENS: 'gen_ai.usage.cache_creation_input_tokens',
  CACHE_READ_INPUT_TOKENS: 'gen_ai.usage.cache_read_input_tokens',
  TOOL_NAME: 'gen_ai.tool.name',
  TOOL_CALL_ID: 'gen_ai.tool.call.id',
} as const;

function setAttr(
  span: Span,
  key: string,
  value: string | number | boolean | undefined | null,
): void {
  if (value === undefined || value === null) return;
  span.setAttribute(key, value);
}

function endSpanSafe(span: Span | undefined): void {
  if (!span) return;
  try {
    span.end();
  } catch {
    // Never let a misbehaving OTel exporter/processor crash the host app.
  }
}

// Tool args/results can be arbitrarily large (a big file read, a long shell
// command's stdout) — cap the exported JSON so one tool call can't inflate a
// span's attribute payload without bound. Shares span-name.ts's
// MAX_BASH_NAME cap pattern and the ./surrogate-safe sliceSurrogateSafe
// helper: cut on a UTF-16 code-unit boundary that never splits a surrogate
// pair, which would corrupt the UTF-8 an OTLP/proto collector requires.
const MAX_TOOL_IO_JSON_CHARS = 32 * 1024; // 32 KB of UTF-16 code units

// sliceSurrogateSafe (./surrogate-safe) is the shared boundary-detection cut,
// no marker appended — callers that need a "…[truncated]" suffix
// (truncateJsonSafe) or a bare cap (capOversizedStringValue, used
// mid-serialization) each append what they need on top of it.
function truncateJsonSafe(json: string): string {
  if (json.length <= MAX_TOOL_IO_JSON_CHARS) return json;
  return `${sliceSurrogateSafe(json, MAX_TOOL_IO_JSON_CHARS)}…[truncated]`;
}

// Caps any individual string value to MAX_TOOL_IO_JSON_CHARS as JSON.stringify
// visits it via the replacer parameter — i.e. *during* the tree walk, before
// it is embedded into the growing output string. truncateJsonSafe alone is
// not enough: JSON.stringify(args) fully materializes the serialized payload
// first and only then gets truncated, so a single huge field (the dominant
// real-world case — full file content, long command stdout) is built out to
// its full size in memory before the cap ever applies. Capping per-string
// here means no single field can inflate the intermediate JSON.stringify
// output past this bound, regardless of how large the source value is.
// truncateJsonSafe remains a backstop below for the many-small-fields case,
// where no single field is oversized but the combined JSON still is.
function capOversizedStringValue(_key: string, value: unknown): unknown {
  if (typeof value === 'string' && value.length > MAX_TOOL_IO_JSON_CHARS) {
    return sliceSurrogateSafe(value, MAX_TOOL_IO_JSON_CHARS);
  }
  return value;
}

// JSON.stringify's real runtime return type is `string | undefined`, not the
// `string` TypeScript's lib.es5.d.ts always claims: for a literal `undefined`
// (or a bare function/symbol) at the top level it returns the *value*
// undefined, not the string "undefined". args/result are typed `unknown` and
// plausibly are `undefined` at runtime (a parameterless tool call, or a
// void-returning tool) — short-circuit here so callers get an honest
// `string | undefined` instead of quietly handing truncateJsonSafe something
// whose `.length` access would throw. Mirrors
// packages/traceroot/src/claude-agent-sdk.ts's tryStringify.
function stringifyToolIo(value: unknown): string | undefined {
  if (value === undefined) return undefined;
  return JSON.stringify(value, capOversizedStringValue);
}

function textOf(message: AgentMessage | undefined): string | undefined {
  if (!message) return undefined;
  if (message.role === 'user' && typeof message.content === 'string') return message.content;
  if (message.role !== 'assistant') return undefined;
  const parts = message.content
    .filter((c): c is { type: 'text'; text: string } => {
      return typeof c === 'object' && c !== null && (c as { type?: unknown }).type === 'text';
    })
    .map((c) => c.text);
  return parts.length > 0 ? parts.join('') : undefined;
}

export function openRootSpan(
  tracer: Tracer,
  parentCtx: Context,
  input: { text: string | undefined; sessionId: string | undefined; captureContent: boolean },
): Span {
  const span = tracer.startSpan('AgentSession.prompt', { kind: SpanKind.INTERNAL }, parentCtx);
  setAttr(span, OI_ATTRIBUTES.SPAN_KIND, 'AGENT');
  setAttr(span, OI_ATTRIBUTES.SESSION_ID, input.sessionId);
  setAttr(span, TR_ATTRIBUTES.SDK_NAME, SDK_NAME);
  setAttr(span, TR_ATTRIBUTES.SDK_VERSION, SDK_VERSION);
  if (input.captureContent) setAttr(span, OI_ATTRIBUTES.INPUT_VALUE, input.text);
  return span;
}

export function closeRootSpan(
  span: Span,
  finalMessages: AgentMessage[],
  willRetry: boolean | undefined,
  captureContent: boolean,
): void {
  setAttr(span, TR_ATTRIBUTES.WILL_RETRY, Boolean(willRetry));
  if (captureContent) {
    let lastAssistant: AgentMessage | undefined;
    for (let i = finalMessages.length - 1; i >= 0; i--) {
      if (finalMessages[i].role === 'assistant') {
        lastAssistant = finalMessages[i];
        break;
      }
    }
    setAttr(span, OI_ATTRIBUTES.OUTPUT_VALUE, textOf(lastAssistant));
  }
  endSpanSafe(span);
}

export function openLlmSpan(tracer: Tracer, parentCtx: Context, message: AssistantMessage): Span {
  const span = tracer.startSpan(message.model || 'pi.llm', { kind: SpanKind.CLIENT }, parentCtx);
  setAttr(span, OI_ATTRIBUTES.SPAN_KIND, 'LLM');
  setAttr(span, GEN_AI_ATTRIBUTES.SYSTEM, message.provider);
  setAttr(span, GEN_AI_ATTRIBUTES.REQUEST_MODEL, message.model);
  return span;
}

export function closeLlmSpan(span: Span, message: AssistantMessage, captureContent: boolean): void {
  span.updateName(message.responseModel || message.model || 'pi.llm');
  setAttr(span, GEN_AI_ATTRIBUTES.RESPONSE_MODEL, message.responseModel || message.model);
  setAttr(span, GEN_AI_ATTRIBUTES.USAGE_INPUT_TOKENS, message.usage?.input);
  setAttr(span, GEN_AI_ATTRIBUTES.USAGE_OUTPUT_TOKENS, message.usage?.output);
  setAttr(span, GEN_AI_ATTRIBUTES.CACHE_READ_INPUT_TOKENS, message.usage?.cacheRead);
  setAttr(span, GEN_AI_ATTRIBUTES.CACHE_WRITE_INPUT_TOKENS, message.usage?.cacheWrite);
  setAttr(span, TR_ATTRIBUTES.COST_TOTAL, message.usage?.cost?.total);
  if (captureContent) {
    setAttr(span, OI_ATTRIBUTES.OUTPUT_VALUE, textOf(message));
  }
  if (message.stopReason === 'error' || message.stopReason === 'aborted') {
    span.setStatus({
      code: SpanStatusCode.ERROR,
      message: message.errorMessage || message.stopReason,
    });
  }
  endSpanSafe(span);
}

export function openToolSpan(
  tracer: Tracer,
  parentCtx: Context,
  toolCallId: string,
  toolName: string,
  args: unknown,
  captureToolIo: boolean,
): Span {
  const span = tracer.startSpan(
    describeToolCallSpan(toolName, args),
    { kind: SpanKind.INTERNAL },
    parentCtx,
  );
  setAttr(span, OI_ATTRIBUTES.SPAN_KIND, 'TOOL');
  setAttr(span, GEN_AI_ATTRIBUTES.TOOL_NAME, toolName);
  setAttr(span, GEN_AI_ATTRIBUTES.TOOL_CALL_ID, toolCallId);
  if (captureToolIo) {
    try {
      const serializedArgs = stringifyToolIo(args);
      if (serializedArgs !== undefined) {
        setAttr(span, OI_ATTRIBUTES.INPUT_VALUE, truncateJsonSafe(serializedArgs));
      }
    } catch {
      // args may contain circular refs or BigInt — skip rather than crash.
    }
  }
  return span;
}

export function closeToolSpan(
  span: Span,
  result: unknown,
  isError: boolean,
  captureToolIo: boolean,
): void {
  if (captureToolIo) {
    try {
      const serializedResult = stringifyToolIo(result);
      if (serializedResult !== undefined) {
        setAttr(span, OI_ATTRIBUTES.OUTPUT_VALUE, truncateJsonSafe(serializedResult));
      }
    } catch {
      // result may contain circular refs or BigInt — skip rather than crash.
    }
  }
  if (isError) {
    span.setStatus({ code: SpanStatusCode.ERROR });
  }
  endSpanSafe(span);
}

// Used whenever a span is closed because a later event force-cleaned it up
// rather than its own normal close event arriving (e.g. an abandoned run's
// spans on a fresh agent_start, or a turn_end that never saw its message_end).
// Marks it so an abnormal trace is distinguishable from a clean one in the
// backend, rather than looking identical to a span that closed normally.
export function closeDanglingSpan(span: Span | undefined): void {
  if (!span) return;
  setAttr(span, TR_ATTRIBUTES.FORCE_CLOSED, true);
  endSpanSafe(span);
}
