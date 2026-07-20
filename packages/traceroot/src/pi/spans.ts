/**
 * Span construction and attribute mapping for Pi coding agent traces.
 *
 * Attribute triad (matching every other traceroot-ts integration):
 * OpenInference span-kind/input/output (drives UI rendering), standard OTel
 * gen_ai.* semconv, and a traceroot.pi.* namespace for retry/force-close
 * markers. SDK identity (traceroot.sdk.name/version) is NOT stamped here —
 * core's TraceRootSpanProcessor.onStart owns it uniformly across every span.
 * Span path/ids_path (Mastra's live-ancestry feature) is not emitted here —
 * Pi delivers a discrete AgentEvent per lifecycle step, so parent/child
 * relationships are already explicit via OTel Context.
 */
import { win32 } from 'node:path';
import { SpanKind, SpanStatusCode } from '@opentelemetry/api';
import type { Context, Span } from '@opentelemetry/api';
import { OI_INPUT_VALUE, OI_OUTPUT_VALUE, OI_SPAN_KIND, OI_TRACE_SESSION_ID } from '../constants';
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
// gen_ai.* family -- unlike claude-agent-sdk.ts, which emits a mixed family
// (llm.token_count.* plus its own gen_ai.response.model). Deliberate
// divergence, not a bug: the backend's otel_transform.py reads pi's gen_ai.*
// keys directly via its own fallback chain, independently of how it reads
// claude-agent-sdk's llm.token_count.* keys. No dual-write is needed here.
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
// span's attribute payload without bound. Cut on a UTF-16 code-unit boundary
// that never splits a surrogate pair (see sliceSurrogateSafe below), which
// would corrupt the UTF-8 an OTLP/proto collector requires.
const MAX_TOOL_IO_JSON_CHARS = 32 * 1024; // 32 KB of UTF-16 code units

// Appended by the post-hoc backstop (capJsonWithMarker) whenever tool I/O had
// to be cut, so a truncated payload is always distinguishable from one that
// merely happened to end this way.
const TRUNCATION_MARKER = '…[truncated]';

/**
 * UTF-16 surrogate-pair-safe truncation boundary check, shared by every cut
 * point in this file so a string is never capped at a UTF-16 code-unit
 * length that splits a surrogate pair — doing so would leave a lone high
 * surrogate in the output and corrupt the UTF-8 an OTLP/proto collector
 * requires. No marker is appended here — callers append what they need
 * (TRUNCATION_MARKER, a bare cap, or an ellipsis) on top of the raw slice.
 *
 * Contract for maxLen <= 0: treated as "cap to nothing" and returns ''.
 * Without this guard, a negative maxLen would fall through to
 * `text.slice(0, cut)` with a negative `cut`, which slices from the END of
 * the string (e.g. 'abcdefghij'.slice(0, -3) === 'abcdefg') — the opposite
 * of capping.
 */
export function sliceSurrogateSafe(text: string, maxLen: number): string {
  if (maxLen <= 0) return '';
  if (text.length <= maxLen) return text;
  let cut = maxLen;
  const code = text.charCodeAt(cut - 1);
  if (code >= 0xd800 && code <= 0xdbff) {
    // High surrogate sitting right at the cut boundary — back off one so we
    // never emit a lone surrogate.
    cut -= 1;
  }
  return text.slice(0, cut);
}

function capJsonWithMarker(json: string): string {
  if (json.length <= MAX_TOOL_IO_JSON_CHARS) return json;
  return `${sliceSurrogateSafe(json, MAX_TOOL_IO_JSON_CHARS)}${TRUNCATION_MARKER}`;
}

// Returns a JSON.stringify replacer that caps each individually-oversized
// STRING field as the tree is walked, so one huge field (full file content,
// long command stdout — the dominant real-world shape of oversized tool I/O)
// is never fully materialized before capJsonWithMarker's post-hoc backstop
// runs. Non-string values pass through untouched — there is no running
// budget across the whole payload, only a per-field cap.
//
// Accepted trade-off: a large ARRAY of many individually-small values (a big
// grep/find result, none of whose elements exceed the cap on their own)
// still transiently serializes in full before capJsonWithMarker slices the
// final string. That's fine — the data is already fully materialized in
// memory by the time a tool result reaches this function. The one
// catastrophic case, a single huge string, is still capped up front.
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
// plausibly are `undefined` at runtime — short-circuit here so callers get an
// honest `string | undefined` instead of handing capJsonWithMarker something
// whose `.length` access would throw. Mirrors claude-agent-sdk.ts's
// tryStringify.
function stringifyToolIo(value: unknown): string | undefined {
  if (value === undefined) return undefined;
  return JSON.stringify(value, capFieldReplacer);
}

function assistantTextOf(message: AgentMessage | undefined): string | undefined {
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
// share one still-open root; see instrumentation.ts's module header). The
// LAST attempt's call wins, since each subsequent agent_end simply overwrites
// OI_OUTPUT_VALUE with its own final assistant message. Ending the span is
// finalizeRootSpan's job, called once when the wrapping prompt() call's own
// promise settles.
export function stampRootOutput(
  span: Span,
  finalMessages: AgentMessage[],
  captureContent: boolean,
): void {
  if (!captureContent) return;
  const lastAssistant = finalMessages.findLast((m) => m.role === 'assistant');
  setAttr(span, OI_OUTPUT_VALUE, assistantTextOf(lastAssistant));
}

// Ends the root span exactly once, when the wrapping prompt() call's own
// returned promise settles (resolve or reject) — see instrumentation.ts's
// proto.prompt wrap. Stamps the observable retry-attempt count and the final
// OTel status, records an exception on failure, then ends the span via
// endSpanSafe so a misbehaving exporter/processor can never crash the host.
export function finalizeRootSpan(
  span: Span,
  retryCount: number,
  status: { code: SpanStatusCode; message?: string },
  error?: unknown,
): void {
  // instrumentation.ts's proto.prompt calls this from inside a detached
  // `.then(onResolve, onReject)` chain that nobody awaits — a throw here
  // would surface as an unhandledRejection capable of crashing the host.
  // setAttribute/setStatus/recordException are NOT otherwise wrapped, so a
  // misbehaving Span implementation could still throw before ever reaching
  // endSpanSafe. Same best-effort idiom as endSpanSafe and closeDanglingSpan.
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
    setAttr(span, OI_OUTPUT_VALUE, assistantTextOf(message));
  }
  if (message.stopReason === 'error' || message.stopReason === 'aborted') {
    span.setStatus({
      code: SpanStatusCode.ERROR,
      message: message.errorMessage || message.stopReason,
    });
  }
  endSpanSafe(span);
}

// Privacy-safe tool span naming, ported from traceroot-pi-extension (the Pi
// CLI extension). Never emits a full file path — basename only, handling
// both / and \ separators — and never emits more than MAX_NAME_SEGMENT_CHARS chars of
// a bash command OR of a path-like argument's basename, truncated without
// splitting a UTF-16 surrogate pair (which would corrupt the UTF-8 an
// OTLP/proto collector requires). The basename cap matters because
// basename() only strips path separators: an argument with none at all (or
// whose final segment is itself huge, e.g. adversarial input) would
// otherwise pass through unbounded.
//
// Truncating a bash command (or an unseparated "path") to 60 chars can still
// leak the start of a pasted secret (e.g. an Authorization header) even when
// captureToolIo is off — deliberate tradeoff: the alternative (no name at
// all) makes traces far less useful, and the full value is only captured as
// a span attribute when captureToolIo is explicitly enabled.
const basename = win32.basename;

const MAX_NAME_SEGMENT_CHARS = 60;

const TOOL_PATH_ARGUMENT_KEYS = [
  'path',
  'file',
  'filePath',
  'file_path',
  'filename',
  'target',
] as const;

function firstPathArgument(args: Record<string, unknown>): string | undefined {
  for (const key of TOOL_PATH_ARGUMENT_KEYS) {
    const value = args[key];
    if (typeof value === 'string' && value) return value;
  }
  return undefined;
}

// Thin wrapper around the sliceSurrogateSafe boundary cut above: appends the
// ellipsis marker this file's span names use, but only when truncation
// actually happened (an untruncated name gets no trailing "…").
function truncateWithEllipsis(text: string, maxLen: number): string {
  if (text.length <= maxLen) return text;
  return `${sliceSurrogateSafe(text, maxLen)}…`;
}

export function describeToolCallSpan(toolName: string, args: unknown): string {
  if (args && typeof args === 'object') {
    const a = args as Record<string, unknown>;
    // Bash calls are checked first and exclusively by their command: tool
    // schemas can attach an incidental path/file/target-like argument
    // alongside `command` (e.g. a cwd or target field), and the generic
    // path-like check below has no bash-specific exemption. Resolving the
    // command here first ensures such an argument never shadows the actual
    // command being run.
    if (toolName === 'bash' && typeof a.command === 'string' && a.command) {
      const cmd = a.command.replace(/\s+/g, ' ').trim();
      if (cmd) return `bash: ${truncateWithEllipsis(cmd, MAX_NAME_SEGMENT_CHARS)}`;
    }
    const pathLike = firstPathArgument(a);
    if (pathLike) {
      const base = basename(pathLike);
      // A non-empty, truthy path-like value can still basename() down to an
      // EMPTY string — e.g. "/", "\\", "///", "C:\\", "C:/": a root or
      // drive-only reference with no filename component to keep. Ordinary,
      // non-adversarial (listing/reading a root directory) — falling through
      // here avoids emitting a dangling "toolName: " with nothing after the
      // colon.
      if (base) {
        return `${toolName}: ${truncateWithEllipsis(base, MAX_NAME_SEGMENT_CHARS)}`;
      }
    }
  }
  return toolName;
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
        setAttr(span, OI_INPUT_VALUE, capJsonWithMarker(serializedArgs));
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
        setAttr(span, OI_OUTPUT_VALUE, capJsonWithMarker(serializedResult));
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
// backend.
//
// Only the setAttr(FORCE_CLOSED) call is guarded here: a misbehaving Span
// implementation whose setAttribute() throws must not prevent endSpanSafe()
// below from running — a span that never has .end() called on it is never
// exported at all, not merely "left open".
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
