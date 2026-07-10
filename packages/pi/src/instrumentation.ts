/**
 * Patch layer: wraps AgentSession.prototype.prompt once per module namespace
 * (wrap-once guarded, matching packages/traceroot/src/claude-agent-sdk.ts's
 * verified real idiom) and builds the entire span tree from a single
 * session.subscribe() listener registered once per session instance.
 *
 * No AsyncLocalStorage: an AgentSession processes one run at a time, so a
 * plain per-session state object (closed over per subscribe() call) is
 * sufficient to correlate events — matching the CLI extension's own
 * "explicit state, never ambient context" convention.
 *
 * Cleanup: instrumentPiCodingAgent() never unsubscribes its own listener.
 * AgentSession.dispose() already removes ALL listeners (including ours)
 * when the host app is done with a session — there is no separate
 * "session ended" event to hook for cleanup.
 */
import { context, trace } from '@opentelemetry/api';
import type { Context, Span, Tracer } from '@opentelemetry/api';
import { resolveConfig } from './config';
import type { PiInstrumentationConfig, ResolvedPiInstrumentationConfig } from './config';
import { createTracing } from './provider';
import {
  closeDanglingSpan,
  closeLlmSpan,
  closeRootSpan,
  closeToolSpan,
  openLlmSpan,
  openRootSpan,
  openToolSpan,
} from './spans';
import type { AgentEvent, AgentSessionInstance, PiCodingAgentModule } from './types';

const WRAPPED = Symbol.for('traceroot.pi_coding_agent.wrapped');

interface SessionSpanState {
  rootSpan: Span | undefined;
  rootCtx: Context | undefined;
  llmSpan: Span | undefined;
  llmCtx: Context | undefined;
  toolSpans: Map<string, Span>;
}

export function instrumentPiCodingAgent(sdk: unknown, config?: PiInstrumentationConfig): unknown {
  const mod = sdk as PiCodingAgentModule & { [WRAPPED]?: boolean };

  const resolved = resolveConfig(config);
  if (!resolved.apiKey) {
    console.warn(
      '[traceroot-pi] TRACEROOT_API_KEY (or config.apiKey) is not set — instrumentation disabled.',
    );
    return sdk;
  }

  const proto = mod?.AgentSession?.prototype;
  if (typeof proto?.prompt !== 'function' || typeof proto?.subscribe !== 'function') {
    console.warn(
      '[traceroot-pi] AgentSession.prototype.prompt/subscribe not found — instrumentation disabled.',
    );
    return sdk;
  }

  if (mod[WRAPPED]) return sdk;
  Object.defineProperty(mod, WRAPPED, { value: true, enumerable: false });

  const { tracer } = createTracing(resolved);

  const pendingInput = new WeakMap<AgentSessionInstance, string>();
  const subscribedSessions = new WeakSet<AgentSessionInstance>();

  const originalPrompt = proto.prompt;
  proto.prompt = function (this: AgentSessionInstance, text, options) {
    if (typeof text === 'string') pendingInput.set(this, text);
    if (!subscribedSessions.has(this)) {
      subscribedSessions.add(this);
      attachSpanListener(this, tracer, resolved, pendingInput);
    }
    return originalPrompt.call(this, text, options);
  };

  return sdk;
}

function attachSpanListener(
  session: AgentSessionInstance,
  tracer: Tracer,
  config: ResolvedPiInstrumentationConfig,
  pendingInput: WeakMap<AgentSessionInstance, string>,
): void {
  const state: SessionSpanState = {
    rootSpan: undefined,
    rootCtx: undefined,
    llmSpan: undefined,
    llmCtx: undefined,
    toolSpans: new Map(),
  };

  session.subscribe((event: AgentEvent) => {
    try {
      handleEvent(event, session, tracer, config, pendingInput, state);
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
  session: AgentSessionInstance,
  tracer: Tracer,
  config: ResolvedPiInstrumentationConfig,
  pendingInput: WeakMap<AgentSessionInstance, string>,
  state: SessionSpanState,
): void {
  switch (event.type) {
    case 'agent_start': {
      const parentCtx = context.active();
      state.rootSpan = openRootSpan(tracer, parentCtx, {
        text: pendingInput.get(session),
        sessionId: session.sessionId,
        captureContent: config.captureContent,
      });
      state.rootCtx = trace.setSpan(parentCtx, state.rootSpan);
      break;
    }
    case 'message_start': {
      if (event.message.role !== 'assistant') return;
      const parentCtx = state.rootCtx ?? context.active();
      state.llmSpan = openLlmSpan(tracer, parentCtx, event.message);
      state.llmCtx = trace.setSpan(parentCtx, state.llmSpan);
      break;
    }
    case 'message_end': {
      if (event.message.role !== 'assistant') return;
      if (state.llmSpan) closeLlmSpan(state.llmSpan, event.message);
      state.llmSpan = undefined;
      // llmCtx stays alive on purpose: tool_execution_start/end for this
      // turn's tool calls fire after message_end but before turn_end, and
      // must still parent under this (now-ended) LLM span rather than
      // falling back to the root span.
      break;
    }
    case 'turn_end': {
      state.llmCtx = undefined;
      break;
    }
    case 'tool_execution_start': {
      const parentCtx = state.llmCtx ?? state.rootCtx ?? context.active();
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
      for (const span of state.toolSpans.values()) closeDanglingSpan(span);
      state.toolSpans.clear();
      closeDanglingSpan(state.llmSpan);
      state.llmSpan = undefined;
      state.llmCtx = undefined;
      if (state.rootSpan)
        closeRootSpan(state.rootSpan, event.messages, event.willRetry, config.captureContent);
      state.rootSpan = undefined;
      state.rootCtx = undefined;
      pendingInput.delete(session);
      break;
    }
    default:
      break;
  }
}
