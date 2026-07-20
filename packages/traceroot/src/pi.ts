import { win32 } from 'node:path';
import { context, ROOT_CONTEXT, SpanKind, SpanStatusCode, trace } from '@opentelemetry/api';
import type { Context, Span } from '@opentelemetry/api';
import { OI_INPUT_VALUE, OI_OUTPUT_VALUE, OI_SPAN_KIND, OI_TRACE_SESSION_ID } from './constants';
import { SDK_VERSION } from './processor';
import { createReresolvingTracer } from './reresolving-tracer';
import type { SpanFactory } from './reresolving-tracer';

/**
 * Hand-transcribed local mirror of @earendil-works/pi-coding-agent / pi-agent-core, verified against their published .d.ts at 0.80.6 (mirrors claude-agent-sdk.ts).
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

// The real AgentMessage union is wider; every consumer here narrows by `.role` first, so these four suffice.
export interface OtherAgentMessage {
  role: 'bashExecution' | 'custom' | 'branchSummary' | 'compactionSummary';
  timestamp?: number;
}

export type AgentMessage = AssistantMessage | UserMessage | ToolResultMessage | OtherAgentMessage;

// Shapes shared by AgentSession.subscribe() and Agent.subscribe(); the former also adds `willRetry` to agent_end.
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
  /** True while a run is actively executing; prompt() routes through steer()/followUp() instead of starting fresh (see isQueueOnlySteer). */
  readonly isStreaming?: boolean;
  prompt(text: string, options?: PromptOptions): Promise<void>;
  /** Queue a steering message mid-run. Optional so a partial double doesn't disable instrumentation over one missing method. */
  steer?(text: string, images?: unknown[]): Promise<void>;
  /** Queue a follow-up message, processed once the agent has no more tool calls or steering messages left. */
  followUp?(text: string, images?: unknown[]): Promise<void>;
  subscribe(listener: (event: AgentEvent) => void): () => void;
  /** dispose() reassigns the private _eventListeners array to a fresh empty array, so no listener (including this file's own) fires again after dispose(). */
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

// Attribute triad matching other traceroot-ts integrations: OpenInference span-kind, OTel gen_ai.*, traceroot.pi.* retry/force-close markers.
const TR_ATTRIBUTES = {
  RETRY_COUNT: 'traceroot.pi.retry_count',
  FORCE_CLOSED: 'traceroot.pi.force_closed',
} as const;

// Pi emits only the gen_ai.* family; the backend reads these directly, no dual-write needed.
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

// Caps exported tool JSON so one call can't inflate a span's attribute payload without bound.
const MAX_TOOL_IO_JSON_CHARS = 32 * 1024; // 32 KB of UTF-16 code units

// Appended whenever capJsonWithMarker cuts, so truncation is distinguishable from coincidence.
const TRUNCATION_MARKER = '…[truncated]';

/** Surrogate-pair-safe truncation: never splits a pair, which would corrupt the UTF-8 an OTLP/proto collector requires. */
export function sliceSurrogateSafe(text: string, maxLen: number): string {
  if (maxLen <= 0) return '';
  if (text.length <= maxLen) return text;
  let cut = maxLen;
  const code = text.charCodeAt(cut - 1);
  if (code >= 0xd800 && code <= 0xdbff) {
    cut -= 1;
  }
  return text.slice(0, cut);
}

function capJsonWithMarker(json: string): string {
  if (json.length <= MAX_TOOL_IO_JSON_CHARS) return json;
  return `${sliceSurrogateSafe(json, MAX_TOOL_IO_JSON_CHARS)}${TRUNCATION_MARKER}`;
}

// Caps each oversized string field while walking the tree, so one huge field never fully materializes first.
function capFieldReplacer(_key: string, value: unknown): unknown {
  if (typeof value === 'string' && value.length > MAX_TOOL_IO_JSON_CHARS) {
    return sliceSurrogateSafe(value, MAX_TOOL_IO_JSON_CHARS);
  }
  return value;
}

// JSON.stringify can return undefined for a top-level undefined value, despite its `string` type — short-circuit before capJsonWithMarker's .length.
function stringifyToolIo(value: unknown): string | undefined {
  if (value === undefined) return undefined;
  return JSON.stringify(value, capFieldReplacer);
}

function assistantTextOf(message: AgentMessage | undefined): string | undefined {
  if (!message) return undefined;
  if (message.role === 'user' && typeof message.content === 'string') return message.content;
  if (message.role !== 'assistant') return undefined;
  // Malformed/non-array content is treated as "no text", not thrown, to avoid skipping the caller's span close.
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

// Stamps output onto the root span without ending it; continuations share one root, so the last agent_end wins.
export function stampRootOutput(
  span: Span,
  finalMessages: AgentMessage[],
  captureContent: boolean,
): void {
  if (!captureContent) return;
  const lastAssistant = finalMessages.findLast((m) => m.role === 'assistant');
  setAttr(span, OI_OUTPUT_VALUE, assistantTextOf(lastAssistant));
}

// Ends the root span exactly once, when the wrapping prompt() call's own returned promise settles.
export function finalizeRootSpan(
  span: Span,
  retryCount: number,
  status: { code: SpanStatusCode; message?: string },
  error?: unknown,
): void {
  // Called from a detached .then() chain nobody awaits, so guard against unhandledRejection here too.
  try {
    setAttr(span, TR_ATTRIBUTES.RETRY_COUNT, retryCount);
    span.setStatus(status);
    if (error !== undefined) {
      span.recordException(error instanceof Error ? error : new Error(String(error)));
    }
  } catch {
    // best-effort, same as endSpanSafe
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

// Privacy-safe tool span naming: never a full file path, commands/basenames capped to MAX_NAME_SEGMENT_CHARS.
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

function truncateWithEllipsis(text: string, maxLen: number): string {
  if (text.length <= maxLen) return text;
  return `${sliceSurrogateSafe(text, maxLen)}…`;
}

export function describeToolCallSpan(toolName: string, args: unknown): string {
  if (args && typeof args === 'object') {
    const a = args as Record<string, unknown>;
    // Checked first, exclusively by command, so an incidental path/target argument (e.g. cwd) can't shadow it.
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

// Used when a span is force-closed by a later event; only the setAttr call is guarded so a throw there can't skip endSpanSafe.
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

// Symbol.for(), not a bare Symbol(): two independently-loaded copies of this package must see the same registry key to detect a second instrumentPiCodingAgent() call — do not change this without adding multiplexing support.
const WRAPPED = Symbol.for('traceroot.pi_coding_agent.wrapped');

interface SessionSpanState {
  rootSpan: Span | undefined;
  rootCtx: Context | undefined;
  llmSpan: Span | undefined;
  llmCtx: Context | undefined;
  toolSpans: Map<string, Span>;
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
  state.llmCtx = undefined;
  if (options.includeRoot) {
    closeDanglingSpan(state.rootSpan);
    state.rootSpan = undefined;
    state.rootCtx = undefined;
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

  // Stamped on AgentSession.prototype, never on `mod`: an ESM namespace object is permanently non-extensible, so defineProperty(mod, ...) would throw.
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

      // Queue-only steer: if already streaming with streamingBehavior set, delegate straight through before any root management, or we'd force-close the active run's still-open root.
      const isQueueOnlySteer = this.isStreaming === true && !!options?.streamingBehavior;
      if (isQueueOnlySteer) {
        return originalPrompt.call(this, text, options);
      }

      // Non-undefined: ensureSubscribed() above created or found this entry.
      const state = sessionSpanState.get(this) as SessionSpanState;

      // OVERLAP SAFETY: a new call raced a still-open prior root — force-close it rather than silently leaking it.
      if (state.rootSpan) {
        sweepDanglingSpans(state, { includeRoot: true });
      }

      // context.active(), matching claude-agent-sdk.ts, lets a host span nest this trace under it.
      const parentCtx = context.active();
      const rootSpan = openRootSpan(tracer, parentCtx, {
        text: typeof text === 'string' ? text : undefined,
        sessionId: this.sessionId,
        captureContent: resolved.captureContent,
      });
      state.rootSpan = rootSpan;
      state.rootCtx = trace.setSpan(parentCtx, rootSpan);
      state.retryCount = 0;

      // Guarded by identity so a mid-run dispose() or a later sweep, which may have already force-closed it, never double-ends. Derives the status message from the error, if any.
      const finalize = (code: SpanStatusCode, error?: unknown): void => {
        if (state.rootSpan !== rootSpan) return;
        state.rootSpan = undefined;
        state.rootCtx = undefined;
        // A rejection can leave the in-flight attempt's spans unclosed.
        sweepDanglingSpans(state);
        const message =
          error === undefined ? undefined : error instanceof Error ? error.message : String(error);
        finalizeRootSpan(rootSpan, state.retryCount, { code, message }, error);
      };

      // A synchronous throw here never reached agent_start: finalize ERROR and rethrow.
      let result: Promise<void>;
      try {
        result = originalPrompt.call(this, text, options);
      } catch (err) {
        finalize(SpanStatusCode.ERROR, err);
        throw err;
      }
      // Separate .then() chain, not a return of result.then(...): that would swallow the rejection since onReject here doesn't rethrow.
      result.then(
        () => finalize(SpanStatusCode.OK),
        (err: unknown) => finalize(SpanStatusCode.ERROR, err),
      );
      return result;
    };
    rollback.push(() => {
      proto.prompt = originalPrompt;
    });

    // Standalone entry points, not prompt() wrappers — needed so a session's first call being one of these still triggers subscribe(). Both just subscribe then delegate, so they share one wrapper.
    const patchSubscribeTrigger = (name: 'steer' | 'followUp'): void => {
      const original = proto[name];
      if (typeof original !== 'function') return;
      proto[name] = function (this: AgentSessionInstance, text: string, images?: unknown[]) {
        ensureSubscribed(this, tracer, resolved, subscribedSessions, sessionSpanState);
        return original.call(this, text, images);
      };
      rollback.push(() => {
        proto[name] = original;
      });
    };
    patchSubscribeTrigger('steer');
    patchSubscribeTrigger('followUp');

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
            // sweepDanglingSpans({ includeRoot: true }) above already nulled every span/ctx field; the state is dropped here regardless.
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

    // Only mark wrapped after every patch succeeded, else a re-entrant call could double-emit spans.
    Object.defineProperty(proto, WRAPPED, { value: true, enumerable: false });
  } catch (err) {
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
      // Sweep dangling spans from a crashed prior attempt; root is untouched (owned by prompt()).
      sweepDanglingSpans(state);
      // An undefined state.rootSpan means this run bypassed the wrapped prompt() call — do not synthesize one.
      break;
    }
    case 'message_start': {
      if (event.message.role !== 'assistant') return;
      // A second message_start with no intervening message_end means the previous LLM span was abandoned — force-close it first.
      closeDanglingSpan(state.llmSpan);
      // Falls back to ROOT_CONTEXT, never context.active(): the latter is the host's ambient OTel context, unrelated to this session, and would risk cross-trace contamination.
      const parentCtx = state.rootCtx ?? ROOT_CONTEXT;
      state.llmSpan = openLlmSpan(tracer, parentCtx, event.message);
      state.llmCtx = trace.setSpan(parentCtx, state.llmSpan);
      break;
    }
    case 'message_end': {
      if (event.message.role !== 'assistant') return;
      if (state.llmSpan) closeLlmSpan(state.llmSpan, event.message, config.captureContent);
      state.llmSpan = undefined;
      // llmCtx stays alive on purpose: this turn's tool_execution events fire after message_end but before turn_end, and still parent under the ended LLM span.
      break;
    }
    case 'turn_end': {
      // If a stream error cut the turn short before message_end closed llmSpan, sweep force-closes it (and tool spans); root is untouched.
      sweepDanglingSpans(state);
      break;
    }
    case 'tool_execution_start': {
      // An already-open toolCallId would have its Map slot silently overwritten — force-close first.
      const existing = state.toolSpans.get(event.toolCallId);
      closeDanglingSpan(existing);
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
      // Mirrors turn_end's sweep; root is not swept (closing it is proto.prompt's job) — this only stamps output onto whatever is open.
      sweepDanglingSpans(state);
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
