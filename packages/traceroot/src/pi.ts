import { win32 } from 'node:path';
import { context, ROOT_CONTEXT, SpanKind, SpanStatusCode, trace } from '@opentelemetry/api';
import type { Context, Span } from '@opentelemetry/api';
import { OI_INPUT_VALUE, OI_OUTPUT_VALUE, OI_SPAN_KIND, OI_TRACE_SESSION_ID } from './constants';
import { SDK_VERSION } from './processor';
import { createReresolvingTracer } from './reresolving-tracer';
import type { SpanFactory } from './reresolving-tracer';

/**
 * Hand-transcribed local mirror of the `@earendil-works/pi-coding-agent` /
 * `@earendil-works/pi-agent-core` surface this package touches.
 *
 * Deliberately NOT imported from the real packages, matching the convention
 * in claude-agent-sdk.ts (hand-rolls a structural ClaudeAgentSDKModule type
 * rather than depending on @anthropic-ai/claude-agent-sdk). This source file
 * imports nothing from Pi's own packages, so the shapes below are easy to
 * re-verify against a new Pi version by hand.
 *
 * Every field here was confirmed against the real published .d.ts for
 * @earendil-works/pi-coding-agent@0.80.6, @earendil-works/pi-agent-core@0.80.6,
 * and @earendil-works/pi-ai@0.80.6, not guessed.
 */

export interface Usage {
  input: number;
  output: number;
  cacheRead: number;
  cacheWrite: number;
  reasoning?: number;
  totalTokens: number;
  cost: {
    input: number;
    output: number;
    cacheRead: number;
    cacheWrite: number;
    total: number;
  };
}

export type StopReason = 'stop' | 'length' | 'toolUse' | 'error' | 'aborted';

export interface AssistantMessage {
  role: 'assistant';
  content: unknown[];
  api: string;
  provider: string;
  model: string;
  responseModel?: string;
  responseId?: string;
  usage: Usage;
  stopReason: StopReason;
  errorMessage?: string;
  timestamp: number;
}

export interface UserMessage {
  role: 'user';
  content: unknown;
  timestamp: number;
}

export interface ToolResultMessage {
  role: 'toolResult';
  toolCallId: string;
  toolName: string;
  content: unknown[];
  details?: unknown;
  isError: boolean;
  timestamp: number;
}

/**
 * `@earendil-works/pi-agent-core`'s real `AgentMessage` is
 * `Message | CustomAgentMessages[keyof CustomAgentMessages]`, and
 * `@earendil-works/pi-coding-agent`'s `dist/core/messages.d.ts` augments
 * `CustomAgentMessages` with four additional roles this package doesn't
 * otherwise model: `bashExecution`, `custom`, `branchSummary`,
 * `compactionSummary`. `sendCustomMessage()` emits a live
 * `message_start`/`message_end` pair with `role: 'custom'` while idle
 * (dist/core/agent-session.js:~1074), so these are reachable at runtime, not
 * merely declared.
 *
 * Every consumer in this package narrows by `.role` before touching any
 * other field, so these four are never dereferenced beyond `.role` here —
 * kept minimal rather than fabricating field shapes this package doesn't
 * act on.
 */
export interface OtherAgentMessage {
  role: 'bashExecution' | 'custom' | 'branchSummary' | 'compactionSummary';
  timestamp?: number;
}

export type AgentMessage = AssistantMessage | UserMessage | ToolResultMessage | OtherAgentMessage;

/**
 * The event union emitted by `AgentSession.subscribe()` and the lower-level
 * `Agent.subscribe()`. `AgentSession.subscribe()` is a strict superset (adds
 * `willRetry` to `agent_end`, plus session-level events this package does
 * not use) — only the shared, raw `AgentEvent` shapes are mirrored here.
 */
export type AgentEvent =
  | { type: 'agent_start' }
  | { type: 'agent_end'; messages: AgentMessage[]; willRetry?: boolean }
  | { type: 'turn_start' }
  | { type: 'turn_end'; message: AgentMessage; toolResults: ToolResultMessage[] }
  | { type: 'message_start'; message: AgentMessage }
  | { type: 'message_update'; message: AgentMessage }
  | { type: 'message_end'; message: AgentMessage }
  | { type: 'tool_execution_start'; toolCallId: string; toolName: string; args: unknown }
  | {
      type: 'tool_execution_update';
      toolCallId: string;
      toolName: string;
      args: unknown;
      partialResult: unknown;
    }
  | {
      type: 'tool_execution_end';
      toolCallId: string;
      toolName: string;
      result: unknown;
      isError: boolean;
    };

export interface PromptOptions {
  images?: unknown[];
  streamingBehavior?: 'steer' | 'followUp';
}

export interface AgentSessionInstance {
  readonly sessionId?: string;
  /**
   * True while a run is actively executing. Confirmed against
   * @earendil-works/pi-coding-agent@0.80.6 (dist/core/agent-session.js:572,
   * `get isStreaming()`) and used by prompt() itself (agent-session.js:812)
   * to decide whether to queue via steer()/followUp() instead of starting a
   * fresh run. instrumentation.ts's proto.prompt wrapper reads this same
   * getter (its own isQueueOnlySteer check) to skip root management entirely
   * for that queue-and-return shape — opening a fresh root or running the
   * overlap sweep for it would force-close the ACTIVE run's still-open root
   * out from under it.
   */
  readonly isStreaming?: boolean;
  prompt(text: string, options?: PromptOptions): Promise<void>;
  /**
   * Queue a steering message while the agent is running — delivered after
   * the current assistant turn finishes its tool calls, before the next LLM
   * call. A standalone public entry point distinct from prompt(text, {
   * streamingBehavior: 'steer' }): a host can call this directly without
   * ever having called prompt() on the session first. Optional here (rather
   * than required, like prompt/subscribe) so a minimal/partial double never
   * disables prompt instrumentation over a missing, unrelated method.
   */
  steer?(text: string, images?: unknown[]): Promise<void>;
  /**
   * Queue a follow-up message to be processed once the agent has no more
   * tool calls or steering messages left. Same standalone-entry-point
   * caveat as steer() above.
   */
  followUp?(text: string, images?: unknown[]): Promise<void>;
  subscribe(listener: (event: AgentEvent) => void): () => void;
  /**
   * Removes every listener registered via subscribe() and disconnects from
   * the underlying Agent. Confirmed against @earendil-works/pi-coding-agent@0.80.6
   * dist/core/agent-session.js: dispose() reassigns the private
   * `_eventListeners` array (the same array subscribe() pushes into and
   * _emit() reads on every dispatch) to a fresh empty array, so no
   * subscribe() listener — including instrumentation.ts's own — is ever
   * invoked again after dispose() runs.
   */
  dispose(): void;
}

export interface AgentSessionConstructor {
  prototype: AgentSessionInstance;
}

/** Structural shape of the imported `@earendil-works/pi-coding-agent` module namespace. */
export interface PiCodingAgentModule {
  AgentSession?: AgentSessionConstructor;
  [key: string]: unknown;
}

/**
 * Configuration resolution for the Pi coding agent instrumentation.
 *
 * Explicit config wins, then a built-in default. Unlike core
 * (traceroot.ts's TRACEROOT_* variables), no env vars are read here.
 * There is no export-pipeline configuration here (no apiKey/baseUrl/exporter
 * override) — this in-tree integration always gets its tracer from the
 * globally-registered OTel provider core sets up, never builds its own.
 */

// OTel tracer scope name shipped in every exported Pi span. Named distinctly
// from core's own SDK_NAME (processor.ts, 'traceroot-ts') to avoid same-name
// shadowing. Follows the same scoped convention as the Claude Agent SDK
// integration (claude-agent-sdk.ts's '@traceroot-ai/claude-agent-sdk').
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

/**
 * Patch layer: wraps AgentSession.prototype.prompt once per module namespace
 * (wrap-once guarded, matching claude-agent-sdk.ts's idiom) and builds the
 * entire span tree from a single session.subscribe() listener registered
 * once per session instance.
 *
 * No AsyncLocalStorage: an AgentSession processes one run at a time, so a
 * plain per-session state object (closed over per subscribe() call) is
 * sufficient to correlate events.
 *
 * ── One trace per prompt() call ───────────────────────────────────────────
 * The root AGENT span is anchored on the wrapped prompt() call's own promise
 * window — opened at entry, closed when that promise settles — mirroring
 * claude-agent-sdk.ts's one-root-per-query() idiom. This is deliberately NOT
 * anchored on agent_start/agent_end: verified against the real, installed
 * @earendil-works/pi-coding-agent@0.80.6 (dist/core/agent-session.js),
 * AgentSession.prompt() awaits _runAgentPrompt(), whose
 * `while (await this._handlePostAgentRun()) { await this.agent.continue(); }`
 * loop performs EVERY retry/compaction/follow-up continuation inside the
 * awaited promise, with each attempt's own agent_end handler completing
 * before prompt() resolves. So a single root spanning the whole prompt()
 * promise captures every continuation attempt as a child LLM/tool span
 * instead of pi's internal retry/compaction/follow-up control flow leaking
 * into trace *structure* as multiple sibling traces. See proto.prompt below
 * for the open/close mechanics and the DECIDED boundary policies.
 *
 * Cleanup: instrumentPiCodingAgent() never unsubscribes its own listener.
 * Verified against @earendil-works/pi-coding-agent@0.80.6
 * (dist/core/agent-session.js): AgentSession.subscribe(listener) pushes onto
 * a private `_eventListeners` array; `_emit(event)` reads that array fresh on
 * every call rather than a captured snapshot. AgentSession.dispose()
 * reassigns `this._eventListeners = []`, so every subsequent `_emit()` call —
 * ours and every other caller's — iterates zero listeners. There is no
 * separate "session ended" event to hook for cleanup, and none is needed.
 */

// Symbol.for(), not a module-scoped Symbol(): stamped onto
// AgentSession.prototype, which two independently-loaded copies of this
// package (e.g. a monorepo hoisting/dedup failure) could both reach. A
// process-wide registry key means both copies see the same symbol, so a
// second instrumentPiCodingAgent() call is rejected with a console.warn
// instead of silently double-instrumenting every session forever. Do not
// flip this back to a bare Symbol() without first building multiplexing
// support for two configs sharing one patched prototype.
const WRAPPED = Symbol.for('traceroot.pi_coding_agent.wrapped');

interface SessionSpanState {
  rootSpan: Span | undefined;
  rootCtx: Context | undefined;
  llmSpan: Span | undefined;
  llmCtx: Context | undefined;
  toolSpans: Map<string, Span>;
  // Observable retry-attempt count for the CURRENT prompt() window, reset to
  // 0 when proto.prompt opens a fresh root and incremented once per
  // agent_end{willRetry:true}. Stamped onto the root as
  // traceroot.pi.retry_count when the window's prompt() promise settles (see
  // finalizeRootSpan in spans.ts).
  retryCount: number;
}

// Force-closes every span left open by an abandoned run: every open tool
// span, then the LLM span, and — only when explicitly requested — the root
// span. Shared by agent_start, turn_end, agent_end, proto.prompt (the
// OVERLAP SAFETY sweep), and dispose(). Safe to call unconditionally: every
// step is already a no-op on undefined/empty, so callers never need their own
// guard before calling this, and one span's close failure never aborts the
// rest of the sweep.
function sweepDanglingSpans(
  state: SessionSpanState,
  options: { includeRoot?: boolean } = {},
): void {
  for (const [, span] of state.toolSpans) {
    closeDanglingSpan(span);
  }
  state.toolSpans.clear();
  closeDanglingSpan(state.llmSpan);
  state.llmSpan = undefined;
  if (options.includeRoot) {
    closeDanglingSpan(state.rootSpan);
    state.rootSpan = undefined;
  }
}

// Shared by proto.prompt/proto.steer/proto.followUp below: each of those
// three independently-callable entry points can be a session's first
// interaction, so each must guarantee session.subscribe() has been called
// exactly once for that session, and a SessionSpanState created for it,
// before delegating to the real method. proto.prompt relies on this —
// `sessionSpanState.get(this)` immediately after calling this is guaranteed
// non-undefined.
function ensureSubscribed(
  session: AgentSessionInstance,
  tracer: SpanFactory,
  config: ResolvedPiInstrumentationConfig,
  subscribedSessions: WeakSet<AgentSessionInstance>,
  sessionSpanState: WeakMap<AgentSessionInstance, SessionSpanState>,
): void {
  if (subscribedSessions.has(session)) return;
  subscribedSessions.add(session);
  attachSpanListener(session, tracer, config, sessionSpanState);
}

export function instrumentPiCodingAgent(sdk: unknown, config?: PiInstrumentationConfig): unknown {
  const mod = sdk as PiCodingAgentModule;

  const resolved = resolveConfig(config);

  // The wrap-once guard is stamped on AgentSession.prototype, never on `mod`
  // itself: when `sdk` is obtained via `import * as pi from "..."` (required
  // for an ESM-only package like @earendil-works/pi-coding-agent), `mod` is
  // an ES module namespace exotic object — the spec makes those permanently
  // non-extensible, so `Object.defineProperty(mod, ...)` always throws
  // TypeError. AgentSession.prototype is an ordinary, extensible object, and
  // it's the thing actually being patched below.
  const proto = mod?.AgentSession?.prototype as
    | (AgentSessionInstance & { [WRAPPED]?: boolean })
    | undefined;
  if (typeof proto?.prompt !== 'function' || typeof proto?.subscribe !== 'function') {
    console.warn(
      '[traceroot-pi] AgentSession.prototype.prompt/subscribe not found — instrumentation disabled.',
    );
    return sdk;
  }

  if (proto[WRAPPED]) {
    console.warn(
      "[traceroot-pi] instrumentPiCodingAgent() was already called for this sdk — this call's config is ignored.",
    );
    return sdk;
  }

  // core's TraceRoot.initialize() guarantees a real, globally registered OTel
  // TracerProvider is already in place before this function runs, so the
  // tracer below can re-resolve that global provider on every span-open
  // (see createReresolvingTracer).
  const tracer = createReresolvingTracer(TRACER_NAME, SDK_VERSION);

  const subscribedSessions = new WeakSet<AgentSessionInstance>();
  // attachSpanListener() below creates one SessionSpanState per session and
  // closes over it for its own subscribe() callback, but
  // AgentSession.prototype.dispose (patched once, below) has no closure over
  // any particular session — it needs this out-of-band map to reach
  // whichever session's SessionSpanState (if any) it was called on.
  const sessionSpanState = new WeakMap<AgentSessionInstance, SessionSpanState>();

  // Install every prototype method patch, then — only then — stamp the
  // wrap-once guard, so a mid-setup failure can't leave the prototype
  // permanently marked "wrapped" while never actually patched. Each patch
  // records an undo on `rollback`, restoring the prototype exactly to how it
  // was found before rethrowing a clear install error.
  const rollback: Array<() => void> = [];
  try {
    // Process-exit flushing is core's responsibility (see traceroot.ts's own
    // process.once('beforeExit', ...)) — no flush hook to register here.

    const originalPrompt = proto.prompt;
    proto.prompt = function (this: AgentSessionInstance, text, options) {
      ensureSubscribed(this, tracer, resolved, subscribedSessions, sessionSpanState);

      // MID-STREAM STEER: verified against @earendil-works/pi-coding-agent@0.80.6
      // (dist/core/agent-session.js) — when a run is ALREADY streaming and the
      // caller passes options.streamingBehavior, prompt() injects into the
      // CURRENTLY-running run's queue and returns early rather than starting a
      // new run. Must be detected and delegated straight through BEFORE any
      // root management below, or opening a fresh root here (or the OVERLAP
      // SAFETY sweep further down) would force-close the ACTIVE run's
      // still-open root mid-flight. The only early-return path that skips
      // root management entirely — the "/command" and 'input'-hook returns
      // below are non-streaming, so their trivial root (policy 2) is
      // unaffected.
      const isQueueOnlySteer = this.isStreaming === true && !!options?.streamingBehavior;
      if (isQueueOnlySteer) {
        return originalPrompt.call(this, text, options);
      }

      // Guaranteed non-undefined: ensureSubscribed() either found an existing
      // entry or attachSpanListener() just created one for `this` session.
      const state = sessionSpanState.get(this) as SessionSpanState;

      // OVERLAP SAFETY: a previous prompt() window's root is still open here.
      // Not the mid-stream steer case above (already detected and returned
      // before this point) — this is a last-resort safety net for a
      // genuinely-new, non-streaming prompt() call that races a still-open
      // prior window. Force-close that stale window (root included) rather
      // than silently overwriting state.rootSpan and leaking it unended
      // forever (a span that never has .end() called on it is never exported).
      if (state.rootSpan) {
        sweepDanglingSpans(state, { includeRoot: true });
        // sweepDanglingSpans() only clears state.llmSpan, not the parenting
        // context that goes with it — left stale, a tool_execution_start
        // racing in right after this sweep would wrongly parent under the
        // now-force-closed LLM span instead of falling back to the fresh
        // root about to be opened below.
        state.llmCtx = undefined;
      }

      // context.active(), matching claude-agent-sdk.ts's own query() span —
      // lets a host that wraps prompt() in its own span nest this trace
      // under it.
      const parentCtx = context.active();
      const rootSpan = openRootSpan(tracer, parentCtx, {
        text: typeof text === 'string' ? text : undefined,
        sessionId: this.sessionId,
        captureContent: resolved.captureContent,
      });
      state.rootSpan = rootSpan;
      state.rootCtx = trace.setSpan(parentCtx, rootSpan);
      state.retryCount = 0;

      // Ends and clears THIS window's root exactly once, guarded by identity
      // (state.rootSpan === rootSpan) so a mid-run dispose() or a later
      // overlapping prompt() call's own OVERLAP SAFETY sweep — either of
      // which may already have force-closed and cleared it — is never
      // double-ended or clobbers a DIFFERENT, newer window's root.
      const finalize = (
        status: { code: SpanStatusCode; message?: string },
        error?: unknown,
      ): void => {
        if (state.rootSpan !== rootSpan) return;
        state.rootSpan = undefined;
        state.rootCtx = undefined;
        // claude-agent-sdk.ts parity (see its own endInFlight()): a
        // rejection (or an early settle racing a crashed attempt) can leave
        // a tool/LLM span from the in-flight attempt never closed by its own
        // normal event. Force-close those BEFORE ending the root so they
        // still export instead of being silently dropped forever. includeRoot
        // is omitted: finalizeRootSpan below is THIS window's own root close.
        sweepDanglingSpans(state);
        state.llmCtx = undefined;
        finalizeRootSpan(rootSpan, state.retryCount, status, error);
      };

      // If this call throws synchronously (before ever returning a promise —
      // e.g. a validation failure inside pi's own prompt() before the agent
      // loop starts), it never reached agent_start: finalize the root as
      // ERROR right here and rethrow, matching claude-agent-sdk.ts's own
      // sync-throw handling in wrapQuery.
      let result: Promise<void>;
      try {
        result = originalPrompt.call(this, text, options);
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        finalize({ code: SpanStatusCode.ERROR, message }, err);
        throw err;
      }
      // A SEPARATE .then chain (not a replacement of `result`) so the host
      // still sees the original resolution/rejection unmodified. Attaching
      // this handler is enough to mark `result`'s rejection as handled for
      // Node's unhandledRejection detection without re-throwing.
      //
      // Four boundary policies:
      //  1. A bypass run (e.g. sendCustomMessage) never opens a root — see
      //     agent_start below.
      //  2. A synchronous early-return prompt() call (handled "/command", an
      //     'input' hook) resolves with no children — a trivial, OK root.
      //  3. A rejection (or the sync-throw path above) finalizes the root
      //     ERROR with the rejection recorded as an exception.
      //  4. A queue-only steer/followUp call returns before this point
      //     (isQueueOnlySteer above) and never reaches finalize() at all.
      result.then(
        () => finalize({ code: SpanStatusCode.OK }),
        (err: unknown) => {
          const message = err instanceof Error ? err.message : String(err);
          finalize({ code: SpanStatusCode.ERROR, message }, err);
        },
      );
      return result;
    };
    rollback.push(() => {
      proto.prompt = originalPrompt;
    });

    // steer() and followUp() are standalone public SDK entry points, not
    // wrappers of prompt() — a host can call them directly without ever
    // calling prompt() on that session first. Without this patch, a session
    // whose first interaction was steer()/followUp() would never get
    // session.subscribe() called, so every AgentEvent would silently produce
    // zero spans. Neither opens a root span itself (only proto.prompt does),
    // so a run triggered purely by steer()/followUp() is a BYPASS run under
    // the rootless boundary policy: see agent_start below. Patched
    // defensively (only when present), like dispose() below.
    if (typeof proto.steer === 'function') {
      const originalSteer = proto.steer;
      proto.steer = function (this: AgentSessionInstance, text: string, images?: unknown[]) {
        ensureSubscribed(this, tracer, resolved, subscribedSessions, sessionSpanState);
        return originalSteer.call(this, text, images);
      };
      rollback.push(() => {
        proto.steer = originalSteer;
      });
    }
    if (typeof proto.followUp === 'function') {
      const originalFollowUp = proto.followUp;
      proto.followUp = function (this: AgentSessionInstance, text: string, images?: unknown[]) {
        ensureSubscribed(this, tracer, resolved, subscribedSessions, sessionSpanState);
        return originalFollowUp.call(this, text, images);
      };
      rollback.push(() => {
        proto.followUp = originalFollowUp;
      });
    }

    // Patched once here (guarded by the same WRAPPED check/stamp above that
    // guards proto.prompt), never per-session: dispose() is a single shared
    // prototype method, same as prompt(). Only patched when dispose actually
    // exists as a function — defensive, so a minimal/partial double never
    // disables prompt instrumentation over a missing, unrelated method.
    if (typeof proto.dispose === 'function') {
      const originalDispose = proto.dispose;
      proto.dispose = function (this: AgentSessionInstance): void {
        // The real dispose() only reassigns the SDK's private
        // _eventListeners array — it does nothing to spans this session's
        // listener already opened. If a host calls dispose() mid-run,
        // force-close any still-open rootSpan/llmSpan/toolSpans here first,
        // exactly like agent_start's own sweep, then delegate to the real
        // dispose().
        const state = sessionSpanState.get(this);
        if (state) {
          try {
            sweepDanglingSpans(state, { includeRoot: true });
          } catch (err) {
            // A misbehaving span/exporter must never make dispose() throw —
            // the host still needs its session torn down even if this
            // best-effort tracing cleanup failed.
            console.warn(
              '[traceroot-pi] failed to force-close in-flight spans during dispose() (a span may leak):',
              err,
            );
          } finally {
            state.rootSpan = undefined;
            state.rootCtx = undefined;
            state.llmSpan = undefined;
            state.llmCtx = undefined;
            // Avoids re-sweeping already-ended spans if dispose() is ever
            // called twice on the same session.
            sessionSpanState.delete(this);
            // subscribedSessions gates the "already subscribed?" check in
            // proto.prompt/steer/followUp above. dispose() only clears
            // _eventListeners — it doesn't make the session instance
            // unusable, and nothing stops a host calling prompt()/steer()/
            // followUp() again after dispose(). Without this delete, a
            // reused session would find subscribedSessions.has(this) still
            // true and silently never re-subscribe.
            subscribedSessions.delete(this);
          }
        }
        return originalDispose.call(this);
      };
      rollback.push(() => {
        proto.dispose = originalDispose;
      });
    }

    // Setup fully succeeded: only now is it correct to mark this prototype
    // wrapped. This MUST be the last statement INSIDE the try: if
    // Object.defineProperty throws (e.g. a frozen/sealed prototype), that
    // failure has to unwind the method patches already applied, like any
    // other mid-setup failure. Left outside the try, the prototype would
    // stay patched but UNSTAMPED, so the next instrumentPiCodingAgent() call
    // would re-patch already-patched methods and emit duplicate spans
    // forever.
    Object.defineProperty(proto, WRAPPED, { value: true, enumerable: false });
  } catch (err) {
    // Undo every patch already applied, newest first, so a failed install
    // leaves AgentSession.prototype exactly as it was found rather than
    // half-patched — then surface the failure loudly instead of leaving a
    // silently-broken, "wrapped"-but-uninstrumented prototype behind.
    for (let i = rollback.length - 1; i >= 0; i--) {
      rollback[i]();
    }
    throw new Error('[traceroot-pi] failed to install instrumentation on AgentSession.prototype', {
      cause: err,
    });
  }

  return sdk;
}

function attachSpanListener(
  session: AgentSessionInstance,
  tracer: SpanFactory,
  config: ResolvedPiInstrumentationConfig,
  sessionSpanState: WeakMap<AgentSessionInstance, SessionSpanState>,
): void {
  const state: SessionSpanState = {
    rootSpan: undefined,
    rootCtx: undefined,
    llmSpan: undefined,
    llmCtx: undefined,
    toolSpans: new Map(),
    retryCount: 0,
  };
  // Reachable from AgentSession.prototype.dispose so a mid-run dispose() can
  // force-close whatever this session's listener callback below left open.
  sessionSpanState.set(session, state);

  session.subscribe((event: AgentEvent) => {
    try {
      handleEvent(event, tracer, config, state);
    } catch (err) {
      // A handler throw must never reach Pi's event dispatcher — that would
      // crash or destabilize the host app's agent loop over a tracing bug.
      console.warn(
        '[traceroot-pi] instrumentation handler failed (a span may be missing or incomplete):',
        err,
      );
    }
  });
}

function handleEvent(
  event: AgentEvent,
  tracer: SpanFactory,
  config: ResolvedPiInstrumentationConfig,
  state: SessionSpanState,
): void {
  switch (event.type) {
    case 'agent_start': {
      // Sweep dangling LLM/tool spans from a crashed prior ATTEMPT within
      // the same prompt() window (e.g. the loop restarted a retry/
      // compaction/follow-up continuation without its own agent_end ever
      // firing for the previous attempt). Deliberately WITHOUT includeRoot:
      // the root now belongs to the enclosing prompt() call's promise
      // window, not to any one attempt, so agent_start must never close it —
      // doing so would end the trace early and orphan every later
      // continuation attempt's spans onto a fresh, disconnected root.
      sweepDanglingSpans(state);
      state.llmSpan = undefined;
      state.llmCtx = undefined;
      // Boundary policy 1 (rootless bypass): if state.rootSpan is undefined
      // here, this run never went through the wrapped prompt() call at all
      // (e.g. sendCustomMessage({triggerTurn:true}), or a steer()/followUp()-
      // only session with no enclosing prompt()). Do NOT synthesize a root
      // for it — its LLM/tool children fall back to ROOT_CONTEXT below
      // and form a parentless mini-trace instead of a fabricated, input-less
      // AGENT span.
      break;
    }
    case 'message_start': {
      if (event.message.role !== 'assistant') return;
      // A second message_start with no intervening message_end/turn_end
      // means the previous LLM span was abandoned (e.g. a stream error
      // skipped straight to a new message) — force-close it first so it
      // still exports instead of having its state.llmSpan slot silently
      // overwritten below.
      closeDanglingSpan(state.llmSpan);
      // Falls back to ROOT_CONTEXT, never context.active(): the latter is
      // whatever the host process's own OTel context manager happens to have
      // ambiently active right now, unrelated to this Pi session. Parenting
      // under it would risk cross-trace contamination in a multi-tenant host.
      const parentCtx = state.rootCtx ?? ROOT_CONTEXT;
      state.llmSpan = openLlmSpan(tracer, parentCtx, event.message);
      state.llmCtx = trace.setSpan(parentCtx, state.llmSpan);
      break;
    }
    case 'message_end': {
      if (event.message.role !== 'assistant') return;
      if (state.llmSpan) closeLlmSpan(state.llmSpan, event.message, config.captureContent);
      state.llmSpan = undefined;
      // llmCtx stays alive on purpose: tool_execution_start/end for this
      // turn's tool calls fire after message_end but before turn_end, and
      // must still parent under this (now-ended) LLM span rather than
      // falling back to the root span.
      break;
    }
    case 'turn_end': {
      // Normally message_end already closed and cleared state.llmSpan before
      // turn_end fires. If it didn't (e.g. a stream error cut the turn short),
      // force-close it here instead of leaking it. Tool calls are
      // turn-scoped too: any tool span still open when the turn ends must be
      // force-closed here as well. Root span is deliberately left untouched —
      // turn_end isn't session end (nor even attempt end).
      sweepDanglingSpans(state);
      state.llmCtx = undefined;
      break;
    }
    case 'tool_execution_start': {
      // A tool_execution_start for a toolCallId that is already open (no
      // intervening tool_execution_end) would otherwise have its Map slot
      // silently overwritten below, losing the abandoned span rather than
      // merely leaving it open — force-close it first so it still surfaces.
      const existing = state.toolSpans.get(event.toolCallId);
      closeDanglingSpan(existing);
      // See message_start's comment: fall back to ROOT_CONTEXT, never the
      // ambient context.active().
      const parentCtx = state.llmCtx ?? state.rootCtx ?? ROOT_CONTEXT;
      const span = openToolSpan(
        tracer,
        parentCtx,
        event.toolCallId,
        event.toolName,
        event.args,
        config.captureToolIo,
      );
      state.toolSpans.set(event.toolCallId, span);
      break;
    }
    case 'tool_execution_end': {
      const span = state.toolSpans.get(event.toolCallId);
      if (span) closeToolSpan(span, event.result, event.isError, config.captureToolIo);
      state.toolSpans.delete(event.toolCallId);
      break;
    }
    case 'agent_end': {
      // Defensive cleanup for any tool/LLM span this attempt left dangling,
      // mirroring turn_end's identical sweep. The root span is NOT part of
      // this sweep: agent_end no longer owns closing the root at all (that
      // is proto.prompt's job, on the enclosing promise settling) — it only
      // stamps this attempt's output onto whatever root is currently open.
      sweepDanglingSpans(state);
      state.llmCtx = undefined;
      if (state.rootSpan) {
        // The LAST attempt's call wins: a retry/compaction/follow-up
        // continuation's own later agent_end simply overwrites
        // OI_OUTPUT_VALUE with ITS final assistant message, so the trace's
        // output always reflects the prompt() call's true final result
        // rather than an intermediate attempt's.
        stampRootOutput(state.rootSpan, event.messages, config.captureContent);
      }
      // Stamped onto the root as traceroot.pi.retry_count once the enclosing
      // prompt() call's own promise settles (see finalizeRootSpan in spans.ts).
      if (event.willRetry) {
        state.retryCount += 1;
      }
      break;
    }
    default:
      break;
  }
}
