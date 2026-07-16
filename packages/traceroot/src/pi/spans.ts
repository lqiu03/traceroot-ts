/**
 * Span construction and attribute mapping for Pi coding agent traces.
 *
 * Attribute triad (matching every other traceroot-ts integration):
 * OpenInference span-kind/input/output (internal, drives UI rendering),
 * standard OTel gen_ai.* semconv, and a traceroot.pi.* namespace for
 * retry/force-close markers. SDK identity (traceroot.sdk.name/version) is
 * NOT stamped here — core's TraceRootSpanProcessor.onStart owns it uniformly
 * across every span, matching the Claude Agent SDK integration.
 * Span path/ids_path (Mastra's live-ancestry feature) is not
 * emitted here — Pi delivers a discrete AgentEvent per lifecycle step, so
 * parent/child relationships are already explicit via OTel Context, not
 * reconstructed from a flat event stream the way Mastra's exporter does.
 */
import { SpanKind, SpanStatusCode } from '@opentelemetry/api';
import type { Context, Span } from '@opentelemetry/api';
import { OI_INPUT_VALUE, OI_OUTPUT_VALUE, OI_SPAN_KIND, OI_TRACE_SESSION_ID } from '../constants';
import { describeToolCallSpan } from './span-name';
import { sliceSurrogateSafe } from './surrogate-safe';
import type { AgentMessage, AssistantMessage } from './types';
import type { SpanFactory } from '../reresolving-tracer';

// Span path keys intentionally omitted — see file header. Everything else
// mirrors packages/mastra/src/exporter.ts's attribute constant shape.
const TR_ATTRIBUTES = {
  WILL_RETRY: 'traceroot.pi.will_retry',
  FORCE_CLOSED: 'traceroot.pi.force_closed',
} as const;

// OpenInference semconv keys are imported from ../constants (single source of
// truth shared with claude-agent-sdk.ts), not re-defined locally.

// gen_ai semconv (standard, used by multiple platforms). Pi emits ONLY the
// gen_ai.* family (gen_ai.request.model, gen_ai.response.model,
// gen_ai.usage.*) -- unlike packages/traceroot/src/claude-agent-sdk.ts,
// which emits a mixed family (llm.token_count.* plus its own
// gen_ai.response.model). This is a deliberate divergence, not a bug: the
// backend's otel_transform.py reads pi's gen_ai.* keys directly via its own
// fallback chain, independently of how it reads claude-agent-sdk's
// llm.token_count.* keys. No dual-write is needed here.
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

// Appended by both the mid-serialization budget (makeBudgetedReplacer) and the
// post-hoc backstop (truncateJsonSafe) whenever tool I/O had to be cut, so a
// truncated payload is always distinguishable from one that merely happened to
// end this way. Single source of truth for the marker text.
const TRUNCATION_MARKER = '…[truncated]';

// sliceSurrogateSafe (./surrogate-safe) is the shared boundary-detection cut,
// no marker appended — callers that need the TRUNCATION_MARKER suffix
// (truncateJsonSafe) or a bare cap (makeBudgetedReplacer, used
// mid-serialization) each append what they need on top of it.
function truncateJsonSafe(json: string): string {
  if (json.length <= MAX_TOOL_IO_JSON_CHARS) return json;
  return `${sliceSurrogateSafe(json, MAX_TOOL_IO_JSON_CHARS)}${TRUNCATION_MARKER}`;
}

// Returns a JSON.stringify replacer that enforces a running output-length
// budget as the tree is walked, capping tool I/O *during* serialization rather
// than after the full payload has already been built. Each call gets its own
// fresh closure — the budget is per-serialization, never shared across calls.
//
// Two distinct oversize shapes both have to be deflected before JSON.stringify
// materializes them:
//   - ONE huge field (full file content, long command stdout — the dominant
//     real-world case): capped to at most MAX_TOOL_IO_JSON_CHARS on its own.
//   - MANY small fields (a large grep/find result: an array of thousands of
//     short lines, none individually oversized): the per-field cap alone never
//     fires for any single element, so without a *running* budget the whole
//     multi-hundred-KB payload is built out in full before truncateJsonSafe's
//     post-hoc slice ever runs — O(N) work and memory on the exact input this
//     is most likely to see.
// A single running `remaining` budget covers both: each visited string is
// first capped to the per-field bound, then charged against `remaining`; once
// `remaining` is spent, every subsequent string is cut to '' (sliceSurrogateSafe
// with a zero/negative bound returns ''), so neither shape can push the
// intermediate serialized output past the budget. truncateJsonSafe remains the
// final backstop that appends the marker and enforces the hard length bound.
//
// The budget cannot be strings-only, though: a tool result that is a large
// array of NON-string primitives (thousands of numbers or booleans — a numeric
// grep/find result, a big matrix) has no oversized string for the per-field cap
// to catch, so a strings-only replacer returned every element untouched and
// JSON.stringify materialized the whole payload before truncateJsonSafe's
// post-hoc slice ran — the exact O(N) blowup a running budget exists to
// prevent. So numbers and booleans are charged against `remaining` too (by
// their serialized width), and — critically — a large ARRAY is proactively
// sliced when the replacer first reaches it, BEFORE JSON.stringify walks its
// elements: one element serializes to at least one JSON char, so more than
// `remaining` elements can never fit the budget regardless of content, and
// slicing there (not merely collapsing each scalar one-by-one, which still
// requires enumerating every element) is what keeps a huge array O(budget)
// rather than O(N) to serialize. Once the budget is fully spent, every
// remaining value collapses to its cheapest valid-JSON form so no further
// content is embedded and no large container is walked deeper.
function makeBudgetedReplacer(): (key: string, value: unknown) => unknown {
  let remaining = MAX_TOOL_IO_JSON_CHARS;
  return function budgetedReplacer(_key: string, value: unknown): unknown {
    // Budget spent: collapse every remaining value to the cheapest valid JSON so
    // JSON.stringify neither embeds more content nor keeps walking a large
    // container. null passes through (already the cheapest literal).
    if (remaining <= 0) {
      if (typeof value === 'string') return '';
      if (typeof value === 'number') return 0;
      if (typeof value === 'boolean') return false;
      if (Array.isArray(value)) return [];
      if (value !== null && typeof value === 'object') return {};
      return value;
    }
    if (typeof value === 'string') {
      const capped =
        value.length > MAX_TOOL_IO_JSON_CHARS
          ? sliceSurrogateSafe(value, MAX_TOOL_IO_JSON_CHARS)
          : value;
      if (capped.length <= remaining) {
        remaining -= capped.length;
        return capped;
      }
      // Crosses the running budget: emit only what's left (surrogate-safe) and
      // spend the rest, so every subsequent value is likewise collapsed above.
      const fit = sliceSurrogateSafe(capped, remaining);
      remaining = 0;
      return fit;
    }
    if (typeof value === 'number' || typeof value === 'boolean') {
      // Charge the scalar's serialized width (e.g. "12345", "true") so a large
      // array of non-string primitives drains the budget instead of slipping
      // through uncounted the way it used to.
      remaining -= String(value).length;
      if (remaining < 0) remaining = 0;
      return value;
    }
    if (Array.isArray(value)) {
      if (value.length > remaining) {
        // Cap the array to at most `remaining` elements up front, then append a
        // truncation marker so the cut is self-evident in the payload.
        // JSON.stringify only walks the sliced copy, so a million-element array
        // costs O(budget), not O(N). truncateJsonSafe still enforces the final
        // hard char bound on top.
        return [...value.slice(0, remaining), TRUNCATION_MARKER];
      }
      return value;
    }
    // Plain object (or null): pass through while budget remains — its scalar
    // leaves are charged as the walk reaches them, and any nested array is
    // sliced when the replacer reaches it in turn.
    return value;
  };
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
  return JSON.stringify(value, makeBudgetedReplacer());
}

function textOf(message: AgentMessage | undefined): string | undefined {
  if (!message) return undefined;
  if (message.role === 'user' && typeof message.content === 'string') return message.content;
  if (message.role !== 'assistant') return undefined;
  // A malformed/non-array content (e.g. undefined) is treated as "no text"
  // rather than thrown: callers close a span right after this call, and a
  // throw here would skip that close entirely, leaking the span.
  if (!Array.isArray(message.content)) return undefined;
  const parts = message.content
    .filter((c): c is { type: 'text'; text: string } => {
      return typeof c === 'object' && c !== null && (c as { type?: unknown }).type === 'text';
    })
    .map((c) => c.text);
  return parts.length > 0 ? parts.join('') : undefined;
}

export function openRootSpan(
  tracer: SpanFactory,
  parentCtx: Context,
  input: { text: string | undefined; sessionId: string | undefined; captureContent: boolean },
): Span {
  const span = tracer.startSpan('AgentSession.prompt', { kind: SpanKind.INTERNAL }, parentCtx);
  setAttr(span, OI_SPAN_KIND, 'AGENT');
  setAttr(span, OI_TRACE_SESSION_ID, input.sessionId);
  if (input.captureContent) setAttr(span, OI_INPUT_VALUE, input.text);
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
    const lastAssistant = finalMessages.findLast((m) => m.role === 'assistant');
    setAttr(span, OI_OUTPUT_VALUE, textOf(lastAssistant));
  }
  endSpanSafe(span);
}

export function openLlmSpan(
  tracer: SpanFactory,
  parentCtx: Context,
  message: AssistantMessage,
): Span {
  const span = tracer.startSpan(message.model || 'pi.llm', { kind: SpanKind.CLIENT }, parentCtx);
  setAttr(span, OI_SPAN_KIND, 'LLM');
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
  if (captureContent) {
    setAttr(span, OI_OUTPUT_VALUE, textOf(message));
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
  tracer: SpanFactory,
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
  setAttr(span, OI_SPAN_KIND, 'TOOL');
  setAttr(span, GEN_AI_ATTRIBUTES.TOOL_NAME, toolName);
  setAttr(span, GEN_AI_ATTRIBUTES.TOOL_CALL_ID, toolCallId);
  if (captureToolIo) {
    try {
      const serializedArgs = stringifyToolIo(args);
      if (serializedArgs !== undefined) {
        setAttr(span, OI_INPUT_VALUE, truncateJsonSafe(serializedArgs));
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
        setAttr(span, OI_OUTPUT_VALUE, truncateJsonSafe(serializedResult));
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
