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
  RETRY_COUNT: 'traceroot.pi.retry_count',
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

// Appended by the post-hoc backstop (truncateJsonSafe) whenever tool I/O had
// to be cut, so a truncated payload is always distinguishable from one that
// merely happened to end this way. Single source of truth for the marker
// text.
const TRUNCATION_MARKER = '…[truncated]';

// sliceSurrogateSafe (./surrogate-safe) is the shared boundary-detection cut,
// no marker appended — callers that need the TRUNCATION_MARKER suffix
// (truncateJsonSafe) or a bare cap (capFieldReplacer, used mid-serialization)
// each append what they need on top of it.
function truncateJsonSafe(json: string): string {
  if (json.length <= MAX_TOOL_IO_JSON_CHARS) return json;
  return `${sliceSurrogateSafe(json, MAX_TOOL_IO_JSON_CHARS)}${TRUNCATION_MARKER}`;
}

// Returns a JSON.stringify replacer that caps each individually-oversized
// STRING field as the tree is walked, so one huge field (full file content,
// long command stdout — the dominant real-world shape of oversized tool I/O)
// is never fully materialized before truncateJsonSafe's post-hoc backstop
// runs. Non-string values (numbers, booleans, arrays, objects) pass through
// untouched — there is no running budget across the whole payload, only a
// per-field cap.
//
// Accepted trade-off (see this file's header comment history / the Ask 3a
// simplification): a large ARRAY of many individually-small values (a big
// grep/find result, none of whose elements exceed the cap on their own) now
// transiently serializes in full before truncateJsonSafe slices the final
// string — O(N) work on data that is already fully materialized in memory by
// the time a tool result reaches this function, which is the right trade for
// a coding agent's tool results. The one catastrophic case — a single huge
// string — is still capped up front, before it is embedded in the growing
// output.
function capFieldReplacer(_key: string, value: unknown): unknown {
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
  return JSON.stringify(value, capFieldReplacer);
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

// Stamps the run's output onto the root span WITHOUT ending it — called by
// agent_end, once per attempt (retry/compaction/follow-up continuations all
// share one still-open root under the new prompt()-anchored model; see
// instrumentation.ts's module header). The LAST attempt's call wins, since
// each subsequent agent_end simply overwrites OI_OUTPUT_VALUE with its own
// final assistant message. Ending the span is finalizeRootSpan's job, called
// once when the wrapping prompt() call's own promise settles.
export function stampRootOutput(
  span: Span,
  finalMessages: AgentMessage[],
  captureContent: boolean,
): void {
  if (!captureContent) return;
  const lastAssistant = finalMessages.findLast((m) => m.role === 'assistant');
  setAttr(span, OI_OUTPUT_VALUE, textOf(lastAssistant));
}

// Ends the root span exactly once, when the wrapping prompt() call's own
// returned promise settles (resolve or reject) — see instrumentation.ts's
// proto.prompt wrap. Stamps the observable retry-attempt count (superseding
// the old per-attempt will_retry flag, which could not represent "this
// prompt() call retried N times" across a single trace) and the final OTel
// status, records an exception on failure, then ends the span via
// endSpanSafe so a misbehaving exporter/processor can never crash the host.
export function finalizeRootSpan(
  span: Span,
  retryCount: number,
  status: { code: SpanStatusCode; message?: string },
  error?: unknown,
): void {
  // instrumentation.ts's proto.prompt calls this from inside a detached
  // `.then(onResolve, onReject)` chain that nobody awaits — a throw here
  // would surface as an unhandledRejection capable of crashing the host,
  // exactly the failure mode endSpanSafe (below) already guards span.end()
  // against. setAttribute/setStatus/recordException are NOT otherwise
  // wrapped, so a single misbehaving Span implementation could still throw
  // out of this function before ever reaching endSpanSafe. Same best-effort
  // idiom as endSpanSafe and closeDanglingSpan above: a tracing failure must
  // never destabilize the host app.
  try {
    setAttr(span, TR_ATTRIBUTES.RETRY_COUNT, retryCount);
    span.setStatus(status);
    if (error !== undefined) {
      span.recordException(error instanceof Error ? error : new Error(String(error)));
    }
  } catch {
    // Never let a misbehaving OTel exporter/processor crash the host app.
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
//
// Only the setAttr(FORCE_CLOSED) call is guarded here: a misbehaving Span
// implementation whose setAttribute() throws must not prevent endSpanSafe()
// below from running. Wrapping the whole function body (or omitting the
// try/catch entirely) would let that throw escape before endSpanSafe() is
// ever reached, so the span would never have .end() called on it — and a
// span with no .end() call is never exported at all, not merely "left open".
// endSpanSafe() itself is already best-effort (see its own comment), so the
// span is guaranteed to at least attempt a close either way.
export function closeDanglingSpan(span: Span | undefined): void {
  if (!span) return;
  try {
    setAttr(span, TR_ATTRIBUTES.FORCE_CLOSED, true);
  } catch (err) {
    console.warn(
      '[traceroot-pi] failed to mark a dangling span force_closed (still ending it):',
      err,
    );
  }
  endSpanSafe(span);
}
