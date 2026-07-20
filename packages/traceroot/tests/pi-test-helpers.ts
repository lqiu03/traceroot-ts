/**
 * Shared fixtures for packages/traceroot/tests/pi-*.test.ts.
 *
 * CONTRACT: prompt()'s returned promise stays pending until emit() sees an
 * agent_end with willRetry not true (or resolvePrompt()/rejectPrompt()/
 * shouldReject settles it). Tests MUST NOT `await session.prompt(text)`
 * before emitting that call's events — it will deadlock:
 *
 *   const done = session.prompt('text');   // NOT awaited yet
 *   session.emit({ type: 'agent_start' });
 *   session.emit({ type: 'agent_end', ..., willRetry: false });
 *   await done;                            // now resolves; root is finalized
 *
 * `await done` is required before asserting on the root span: it is
 * finalized in a `.then()` on this same promise, registered before the
 * promise is returned to the caller, so it always runs before `await done`
 * resumes.
 */
import { trace } from '@opentelemetry/api';
import type { ExportResult } from '@opentelemetry/core';
import { ExportResultCode } from '@opentelemetry/core';
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import { SimpleSpanProcessor } from '@opentelemetry/sdk-trace-base';
import type { ReadableSpan, SpanExporter } from '@opentelemetry/sdk-trace-base';
import {
  instrumentPiCodingAgent,
  type AgentEvent,
  type AssistantMessage,
  type PiInstrumentationConfig,
  type PromptOptions,
} from '../src/pi';

/** Captures spans exported by a real global TracerProvider (see makeRig). */
export class CapturingExporter implements SpanExporter {
  readonly spans: ReadableSpan[] = [];
  export(spans: ReadableSpan[], resultCallback: (result: ExportResult) => void): void {
    this.spans.push(...spans);
    resultCallback({ code: ExportResultCode.SUCCESS });
  }
  async shutdown(): Promise<void> {}
}

// Fresh FakeAgentSession class per rig/test (instrumentPiCodingAgent patches
// the prototype directly). `shouldReject`, if true for prompt()'s text,
// rejects before agent_start fires (early-return validation path).
export function makeFakeSessionClass(shouldReject?: (text: string) => boolean) {
  return class FakeAgentSession {
    sessionId = 'sess-1';
    disposed = false;
    // Mirrors AgentSession.prototype.isStreaming.
    isStreaming = false;
    private listeners: Array<(event: AgentEvent) => void> = [];
    private pending: { resolve: () => void; reject: (err: unknown) => void } | undefined;

    async prompt(text: string, options?: PromptOptions): Promise<void> {
      if (shouldReject?.(text)) {
        throw new Error(`validation failed for: ${text}`);
      }
      // Queues into the active run; must not touch `this.pending`, which
      // belongs to the earlier prompt() call.
      if (this.isStreaming && options?.streamingBehavior) {
        return;
      }
      return new Promise<void>((resolve, reject) => {
        this.pending = { resolve, reject };
      });
    }
    subscribe(listener: (event: AgentEvent) => void): () => void {
      this.listeners.push(listener);
      return () => {
        this.listeners = this.listeners.filter((l) => l !== listener);
      };
    }
    emit(event: AgentEvent): void {
      for (const listener of this.listeners) listener(event);
      if (event.type === 'agent_end' && !event.willRetry && this.pending) {
        const { resolve } = this.pending;
        this.pending = undefined;
        resolve();
      }
    }
    // Settles as SUCCESS with no agent_end (mirrors a handled "/command").
    resolvePrompt(): void {
      if (!this.pending) return;
      const { resolve } = this.pending;
      this.pending = undefined;
      resolve();
    }
    // Mirrors an internal error rejected mid-loop (unlike shouldReject's
    // pre-flight failure, which happens before the loop starts).
    rejectPrompt(err: unknown): void {
      if (!this.pending) return;
      const { reject } = this.pending;
      this.pending = undefined;
      reject(err);
    }
    dispose(): void {
      this.disposed = true;
      this.listeners = [];
    }
  };
}

// Fresh capture/provider/session, wired through instrumentPiCodingAgent().
// trace.disable() must run first: the instrumentation re-resolves its
// tracer through the global `trace` facade rather than owning a provider,
// so a stale registration would leak spans across rigs.
export function makeRig(config: PiInstrumentationConfig = {}): {
  capture: CapturingExporter;
  Session: ReturnType<typeof makeFakeSessionClass>;
} {
  trace.disable();
  const capture = new CapturingExporter();
  const provider = new NodeTracerProvider();
  // Synchronous export on span.end(); spans land in capture.spans
  // immediately, with no batching delay.
  provider.addSpanProcessor(new SimpleSpanProcessor(capture));
  provider.register();
  const Session = makeFakeSessionClass();
  const sdk = { AgentSession: Session };
  instrumentPiCodingAgent(sdk, config);
  return { capture, Session };
}

// Placeholder usage/cost numbers. Tests asserting on specific figures must
// pass an explicit `usage` override rather than relying on this default.
export function assistantMessage(overrides: Partial<AssistantMessage> = {}): AssistantMessage {
  return {
    role: 'assistant',
    content: [{ type: 'text', text: 'done' }],
    api: 'anthropic-messages',
    provider: 'anthropic',
    model: 'claude-sonnet-5',
    usage: {
      input: 1,
      output: 1,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 2,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
    stopReason: 'stop',
    timestamp: 0,
    ...overrides,
  } as AssistantMessage;
}

export function attrs(span: ReadableSpan): Record<string, unknown> {
  return span.attributes as Record<string, unknown>;
}
