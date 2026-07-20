import { win32 } from 'node:path';
import { context, ROOT_CONTEXT, SpanKind, SpanStatusCode, trace } from '@opentelemetry/api';
import type { Context, Span } from '@opentelemetry/api';
import { OI_INPUT_VALUE, OI_OUTPUT_VALUE, OI_SPAN_KIND, OI_TRACE_SESSION_ID } from './constants';
import { SDK_VERSION } from './processor';
import { createReresolvingTracer } from './reresolving-tracer';
import type { SpanFactory } from './reresolving-tracer';

/**
 * Hand-transcribed local mirror of the `@earendil-works/pi-coding-agent` /
 * `@earendil-works/pi-agent-core` surface this package touches (not imported
 * from the real packages, mirroring claude-agent-sdk.ts). Verified against
 * the published .d.ts for @earendil-works/pi-coding-agent@0.80.6,
 * pi-agent-core@0.80.6, and pi-ai@0.80.6 — this file's one provenance note.
 *
 * One AGENT root span per prompt() call: opened on entry to
 * AgentSession.prototype.prompt, closed only when that call's own returned
 * promise settles, so every retry/compaction/follow-up continuation lands
 * as a child of the same root.
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

// The real `AgentMessage` union is wider (`bashExecution`/`custom`/
// `branchSummary`/`compactionSummary` roles); every consumer here narrows
// by `.role` first, so these four are kept minimal.
export interface OtherAgentMessage {
  role: 'bashExecution' | 'custom' | 'branchSummary' | 'compactionSummary';
  timestamp?: number;
}

export type AgentMessage = AssistantMessage | UserMessage | ToolResultMessage | OtherAgentMessage;

// Shapes shared by `AgentSession.subscribe()` and `Agent.subscribe()`; the
// former also adds `willRetry` to `agent_end` plus events unused here.
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
  /** True while a run is actively executing; prompt() reads this to route through steer()/followUp() instead of starting a fresh run (see isQueueOnlySteer below). */
  readonly isStreaming?: boolean;
  prompt(text: string, options?: PromptOptions): Promise<void>;
  /** Queue a steering message mid-run, a standalone entry point a host can call without ever having called prompt() first. Optional (unlike prompt/subscribe) so a minimal/partial double never disables prompt instrumentation over a missing, unrelated method — same reasoning for followUp() below. */
  steer?(text: string, images?: unknown[]): Promise<void>;
  /** Queue a follow-up message, processed once the agent has no more tool calls or steering messages left. Same standalone shape as steer(). */
  followUp?(text: string, images?: unknown[]): Promise<void>;
  subscribe(listener: (event: AgentEvent) => void): () => void;
  /**
   * Removes every listener registered via subscribe(): dispose() reassigns
   * the private `_eventListeners` array (the same array subscribe() pushes
   * into and _emit() reads on every dispatch) to a fresh empty array, so no
   * listener — including this file's own — is invoked again after dispose().
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

// No env-var fallback or export-pipeline config — always uses core's provider.
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

// Attribute triad, matching every traceroot-ts integration: OpenInference
// span-kind/input/output, OTel gen_ai.* semconv, traceroot.pi.* retry/
// force-close markers. Span path/ids_path is skipped as unneeded here.
const TR_ATTRIBUTES = {
  RETRY_COUNT: 'traceroot.pi.retry_count',
  FORCE_CLOSED: 'traceroot.pi.force_closed',
} as const;

// Pi emits only the gen_ai.* family — unlike claude-agent-sdk.ts's mixed set.
// Safe and deliberate: the backend's otel_transform.py reads these gen_ai.*
// keys directly via its own fallback chain, so no dual-write of llm.token_count.*
// is needed here.
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

// Caps exported tool JSON so one call (a big file read, long stdout) can't
// inflate a span's attribute payload without bound.
const MAX_TOOL_IO_JSON_CHARS = 32 * 1024; // 32 KB of UTF-16 code units

// Appended whenever capJsonWithMarker cuts, distinguishing truncation from a
// payload that merely happened to end this way.
const TRUNCATION_MARKER = '…[truncated]';

/**
 * UTF-16 surrogate-pair-safe truncation, shared by every cut point in this
 * file: never caps at a code-unit length that splits a surrogate pair, which
 * would leave a lone high surrogate and corrupt the UTF-8 an OTLP/proto
 * collector requires. maxLen <= 0 returns '' — a negative maxLen would
 * otherwise slice from the END of the string instead of capping.
 */
export function sliceSurrogateSafe(text: string, maxLen: number): string {
  if (maxLen <= 0) return '';
  if (text.length <= maxLen) return text;
  let cut = maxLen;
  const code = text.charCodeAt(cut - 1);
  if (code >= 0xd800 && code <= 0xdbff) {
    // Lone high surrogate at the boundary — back off one.
    cut -= 1;
  }
  return text.slice(0, cut);
}

function capJsonWithMarker(json: string): string {
  if (json.length <= MAX_TOOL_IO_JSON_CHARS) return json;
  return `${sliceSurrogateSafe(json, MAX_TOOL_IO_JSON_CHARS)}${TRUNCATION_MARKER}`;
}

// Caps each oversized STRING field while walking the tree, so one huge
// field (full file content, long stdout) never fully materializes first.
function capFieldReplacer(_key: string, value: unknown): unknown {
  if (typeof value === 'string' && value.length > MAX_TOOL_IO_JSON_CHARS) {
    return sliceSurrogateSafe(value, MAX_TOOL_IO_JSON_CHARS);
  }
  return value;
}

// JSON.stringify's real return type is `string | undefined`, not the
// `string` TypeScript claims — a top-level undefined returns the value
// undefined, not "undefined". Short-circuit before capJsonWithMarker's `.length`.
function stringifyToolIo(value: unknown): string | undefined {
  if (value === undefined) return undefined;
  return JSON.stringify(value, capFieldReplacer);
}

function assistantTextOf(message: AgentMessage | undefined): string | undefined {
  if (!message) return undefined;
  if (message.role === 'user' && typeof message.content === 'string') return message.content;
  if (message.role !== 'assistant') return undefined;
  // Malformed/non-array content is treated as "no text", not thrown — a
  // throw here would skip the caller's span close, leaking the span.
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

// Stamps output onto the root span WITHOUT ending it. Continuations share
// one still-open root, so each later agent_end overwrites OI_OUTPUT_VALUE
// and the LAST attempt wins; ending the span is finalizeRootSpan's job.
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
// returned promise settles. Stamps retry count and final OTel status,
// records an exception on failure, then ends via endSpanSafe.
export function finalizeRootSpan(
  span: Span,
  retryCount: number,
  status: { code: SpanStatusCode; message?: string },
  error?: unknown,
): void {
  // Called from a detached `.then()` chain nobody awaits — a throw here
  // would surface as an unhandledRejection, so guard these too.
  try {
    setAttr(span, TR_ATTRIBUTES.RETRY_COUNT, retryCount);
    span.setStatus(status);
    if (error !== undefined) {
      span.recordException(error instanceof Error ? error : new Error(String(error)));
    }
  } catch {
    // Same best-effort guard as endSpanSafe.
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

// Privacy-safe tool span naming: never a full file path, commands/basenames
// capped to MAX_NAME_SEGMENT_CHARS (accepted even though truncation can
// still leak a pasted secret's start — no name at all is far less useful).
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

// Appends "…" only when truncation actually happened.
function truncateWithEllipsis(text: string, maxLen: number): string {
  if (text.length <= maxLen) return text;
  return `${sliceSurrogateSafe(text, maxLen)}…`;
}

export function describeToolCallSpan(toolName: string, args: unknown): string {
  if (args && typeof args === 'object') {
    const a = args as Record<string, unknown>;
    // Checked first, exclusively by command — an incidental path/target
    // argument (e.g. cwd) must not shadow it.
    if (toolName === 'bash' && typeof a.command === 'string' && a.command) {
      const cmd = a.command.replace(/\s+/g, ' ').trim();
      if (cmd) return `bash: ${truncateWithEllipsis(cmd, MAX_NAME_SEGMENT_CHARS)}`;
    }
    const pathLike = firstPathArgument(a);
    if (pathLike) {
      const base = basename(pathLike);
      // Can still basename() to '' (e.g. "/") — fall through instead of "toolName: ".
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

// Used when a span is force-closed by a later event instead of its own
// normal close, so an abnormal trace is distinguishable from a clean one.
// Only the setAttr call is guarded, so a throw there can't skip endSpanSafe.
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

// Patch layer: wraps prompt once per module namespace and builds the span
// tree from one session.subscribe() listener per session. No
// AsyncLocalStorage needed — an AgentSession runs one prompt at a time.

// Symbol.for(), not a module-scoped Symbol(): stamped onto
// AgentSession.prototype, which two independently-loaded copies of this
// package (e.g. a monorepo hoisting/dedup failure) could both reach, so a
// second instrumentPiCodingAgent() call is rejected via console.warn instead
// of silently double-instrumenting forever. Do not flip this back to a bare
// Symbol() without first building multiplexing support for two configs
// sharing one patched prototype.
const WRAPPED = Symbol.for('traceroot.pi_coding_agent.wrapped');

interface SessionSpanState {
  rootSpan: Span | undefined;
  rootCtx: Context | undefined;
  llmSpan: Span | undefined;
  llmCtx: Context | undefined;
  toolSpans: Map<string, Span>;
  // Reset by proto.prompt; stamped as traceroot.pi.retry_count on settle.
  retryCount: number;
}

// Force-closes spans left open by an abandoned run, root only when requested.
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

// Guarantees subscribe() ran once — proto.prompt/steer/followUp can each independently be a session's first call.
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

  // Stamped on AgentSession.prototype, never on `mod`: an ESM `import * as
  // pi from "..."` namespace object is permanently non-extensible per spec,
  // so `Object.defineProperty(mod, ...)` would always throw TypeError.
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

  // Re-resolves core's globally registered TracerProvider on every span-open.
  const tracer = createReresolvingTracer(TRACER_NAME, SDK_VERSION);

  const subscribedSessions = new WeakSet<AgentSessionInstance>();
  // Out-of-band, since dispose() (below) has no closure over a session.
  const sessionSpanState = new WeakMap<AgentSessionInstance, SessionSpanState>();

  const rollback: Array<() => void> = [];
  try {
    const originalPrompt = proto.prompt;
    proto.prompt = function (this: AgentSessionInstance, text, options) {
      ensureSubscribed(this, tracer, resolved, subscribedSessions, sessionSpanState);

      // When a run is ALREADY streaming and the caller passes
      // options.streamingBehavior, prompt() injects into the CURRENTLY-running
      // run's queue and returns early instead of starting a new run. Must be
      // detected and delegated straight through BEFORE any root management
      // below, or opening a fresh root here (or the OVERLAP SAFETY sweep
      // further down) would force-close the ACTIVE run's still-open root.
      const isQueueOnlySteer = this.isStreaming === true && !!options?.streamingBehavior;
      if (isQueueOnlySteer) {
        return originalPrompt.call(this, text, options);
      }

      // Non-undefined: ensureSubscribed() above created or found this entry.
      const state = sessionSpanState.get(this) as SessionSpanState;

      // OVERLAP SAFETY: a genuinely-new call raced a still-open prior
      // window's root — force-close it (and its stale parenting context)
      // rather than silently overwriting state.rootSpan and leaking it.
      if (state.rootSpan) {
        sweepDanglingSpans(state, { includeRoot: true });
        state.llmCtx = undefined;
      }

      // context.active(), matching claude-agent-sdk.ts, lets a host that
      // wraps prompt() in its own span nest this trace under it.
      const parentCtx = context.active();
      const rootSpan = openRootSpan(tracer, parentCtx, {
        text: typeof text === 'string' ? text : undefined,
        sessionId: this.sessionId,
        captureContent: resolved.captureContent,
      });
      state.rootSpan = rootSpan;
      state.rootCtx = trace.setSpan(parentCtx, rootSpan);
      state.retryCount = 0;

      // Guarded by identity so a mid-run dispose() or a later prompt()'s own
      // sweep, which may have already force-closed it, never double-ends.
      const finalize = (
        status: { code: SpanStatusCode; message?: string },
        error?: unknown,
      ): void => {
        if (state.rootSpan !== rootSpan) return;
        state.rootSpan = undefined;
        state.rootCtx = undefined;
        // A rejection can leave the in-flight attempt's spans unclosed.
        sweepDanglingSpans(state);
        state.llmCtx = undefined;
        finalizeRootSpan(rootSpan, state.retryCount, status, error);
      };

      // A synchronous throw here never reached agent_start: finalize ERROR and rethrow.
      let result: Promise<void>;
      try {
        result = originalPrompt.call(this, text, options);
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        finalize({ code: SpanStatusCode.ERROR, message }, err);
        throw err;
      }
      // Four boundary policies:
      //  1. A bypass run (e.g. sendCustomMessage) never opens a root — see
      //     agent_start below.
      //  2. A synchronous early-return prompt() call (handled "/command", an
      //     'input' hook) resolves with no children — a trivial, OK root.
      //  3. A rejection (or the sync-throw path above) finalizes the root
      //     ERROR with the rejection recorded as an exception.
      //  4. A queue-only steer/followUp call returns before this point
      //     (isQueueOnlySteer above) and never reaches finalize() at all.
      // A SEPARATE .then chain, not a replacement of `result`: the host still
      // awaits the original promise and sees its resolution/rejection unchanged.
      // (Returning result.then(...) instead would swallow rejections from the
      // host, since the onReject handler here does not rethrow.) Attaching this
      // handler also marks result's rejection as handled for Node's
      // unhandledRejection detection.
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

    // Standalone entry points, not prompt() wrappers — without this a session
    // whose first interaction was one of them never gets subscribe() called.
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

    // Patched once, never per-session; only when dispose exists (defensive).
    if (typeof proto.dispose === 'function') {
      const originalDispose = proto.dispose;
      proto.dispose = function (this: AgentSessionInstance): void {
        // The real dispose() only clears _eventListeners, so force-close any still-open spans first.
        const state = sessionSpanState.get(this);
        if (state) {
          try {
            sweepDanglingSpans(state, { includeRoot: true });
          } catch (err) {
            // Must never make dispose() throw — the host still needs teardown.
            console.warn(
              '[traceroot-pi] failed to force-close in-flight spans during dispose() (a span may leak):',
              err,
            );
          } finally {
            state.rootSpan = undefined;
            state.rootCtx = undefined;
            state.llmSpan = undefined;
            state.llmCtx = undefined;
            sessionSpanState.delete(this);
            // Else a reused session would never re-subscribe (still "has" it).
            subscribedSessions.delete(this);
          }
        }
        return originalDispose.call(this);
      };
      rollback.push(() => {
        proto.dispose = originalDispose;
      });
    }

    // Only mark wrapped after every patch succeeded, else a re-entrant call
    // would find an unstamped-but-patched prototype and double-emit spans.
    Object.defineProperty(proto, WRAPPED, { value: true, enumerable: false });
  } catch (err) {
    // Undo every patch, newest first, leaving the prototype exactly as found.
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
  sessionSpanState.set(session, state);

  session.subscribe((event: AgentEvent) => {
    try {
      handleEvent(event, tracer, config, state);
    } catch (err) {
      // Must never reach Pi's event dispatcher and destabilize its agent loop.
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
      // Sweep dangling spans from a crashed prior attempt. WITHOUT includeRoot
      // — the root belongs to the enclosing prompt() call, not one attempt.
      sweepDanglingSpans(state);
      state.llmSpan = undefined;
      state.llmCtx = undefined;
      // Policy 1: an undefined state.rootSpan means this run never went
      // through the wrapped prompt() call — do not synthesize a root.
      break;
    }
    case 'message_start': {
      if (event.message.role !== 'assistant') return;
      // A second message_start with no intervening message_end means the
      // previous LLM span was abandoned — force-close before its slot is overwritten.
      closeDanglingSpan(state.llmSpan);
      // Falls back to ROOT_CONTEXT, never context.active(): the latter is
      // whatever the host process's own OTel context manager happens to have
      // ambiently active right now, unrelated to this Pi session — parenting
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
      // llmCtx stays alive on purpose: this turn's tool_execution_start/end
      // fire after message_end but before turn_end, and must still parent
      // under the now-ended LLM span rather than the root.
      break;
    }
    case 'turn_end': {
      // If a stream error cut the turn short before message_end closed
      // llmSpan, sweep force-closes it (and tool spans). Root is untouched.
      sweepDanglingSpans(state);
      state.llmCtx = undefined;
      break;
    }
    case 'tool_execution_start': {
      // An already-open toolCallId would have its Map slot silently overwritten — force-close first.
      const existing = state.toolSpans.get(event.toolCallId);
      closeDanglingSpan(existing);
      // Same ROOT_CONTEXT fallback as message_start above.
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
      // Mirrors turn_end's sweep. Root is NOT swept (closing it is
      // proto.prompt's job) — this only stamps output onto whatever is open.
      sweepDanglingSpans(state);
      state.llmCtx = undefined;
      if (state.rootSpan) {
        stampRootOutput(state.rootSpan, event.messages, config.captureContent);
      }
      if (event.willRetry) {
        state.retryCount += 1;
      }
      break;
    }
    default:
      break;
  }
}
