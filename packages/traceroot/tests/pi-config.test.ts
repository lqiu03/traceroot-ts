/**
 * Lens: config resolution (packages/traceroot/src/pi/config.ts).
 *
 * Split out of packages/pi/tests/config-resolution.test.ts: this in-tree
 * integration's PiInstrumentationConfig no longer has apiKey/baseUrl fields
 * (see config.ts's own header — there is no export-pipeline configuration
 * here at all, since this integration always gets its tracer from the
 * globally-registered OTel provider core sets up, never builds its own). So
 * every apiKey/baseUrl/env-var-precedence/empty-apiKey-disables-instrumentation
 * test from the original file is deleted along with those fields — there is
 * nothing left for them to probe. What remains: resolveConfig()'s
 * captureContent/captureToolIo defaults, and instrumentPiCodingAgent()'s
 * config-snapshot immutability (adapted to the surviving fields).
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { trace } from '@opentelemetry/api';
import type { ExportResult } from '@opentelemetry/core';
import { ExportResultCode } from '@opentelemetry/core';
import { SimpleSpanProcessor } from '@opentelemetry/sdk-trace-base';
import type { ReadableSpan, SpanExporter } from '@opentelemetry/sdk-trace-base';
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import { resolveConfig } from '../src/pi/config';
import { instrumentPiCodingAgent } from '../src/pi/instrumentation';
import type { AgentEvent, AssistantMessage } from '../src/pi/types';

// Copied locally per-file, matching every other tests/*.test.ts in this
// package — no shared module-level exporter/session state across files.
class CapturingExporter implements SpanExporter {
  readonly spans: ReadableSpan[] = [];
  export(spans: ReadableSpan[], resultCallback: (result: ExportResult) => void): void {
    this.spans.push(...spans);
    resultCallback({ code: ExportResultCode.SUCCESS });
  }
  async shutdown(): Promise<void> {}
}

// Fresh class per rig, not a shared module-level class — instrumentPiCodingAgent
// patches AgentSession.prototype directly, so reusing one class across tests
// would stack multiple wrap layers onto the same prototype method.
function makeFakeSessionClass() {
  return class FakeAgentSession {
    sessionId = 'sess-1';
    private listeners: Array<(event: AgentEvent) => void> = [];
    // prompt()'s returned promise settles only once its final agent_end
    // fires (willRetry !== true) — mirrors the real SDK; see
    // pi-test-helpers.ts's module header for the full rationale.
    private pending: { resolve: () => void; reject: (err: unknown) => void } | undefined;
    async prompt(_text: string, _options?: unknown): Promise<void> {
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
  };
}

function assistantMessage(overrides: Partial<AssistantMessage> = {}): AssistantMessage {
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

function attrs(span: ReadableSpan): Record<string, unknown> {
  return span.attributes as Record<string, unknown>;
}

function registerCapturingProvider(capture: CapturingExporter): void {
  trace.disable();
  const provider = new NodeTracerProvider();
  provider.addSpanProcessor(new SimpleSpanProcessor(capture));
  provider.register();
}

test('resolveConfig() defaults captureContent and captureToolIo to true when omitted', () => {
  const resolved = resolveConfig();
  assert.equal(resolved.captureContent, true);
  assert.equal(resolved.captureToolIo, true);

  const resolvedFromEmptyObject = resolveConfig({});
  assert.equal(resolvedFromEmptyObject.captureContent, true);
  assert.equal(resolvedFromEmptyObject.captureToolIo, true);
});

test('resolveConfig() respects explicit false overrides for captureContent and captureToolIo independently', () => {
  const contentOff = resolveConfig({ captureContent: false });
  assert.equal(contentOff.captureContent, false);
  assert.equal(contentOff.captureToolIo, true, 'captureToolIo must keep its own default');

  const toolIoOff = resolveConfig({ captureToolIo: false });
  assert.equal(toolIoOff.captureContent, true, 'captureContent must keep its own default');
  assert.equal(toolIoOff.captureToolIo, false);

  const bothOff = resolveConfig({ captureContent: false, captureToolIo: false });
  assert.equal(bothOff.captureContent, false);
  assert.equal(bothOff.captureToolIo, false);
});

test('instrumentPiCodingAgent() snapshots the config object at call time — mutating captureContent on the caller-owned object after the call returns has no effect on already-instrumented behavior', async () => {
  const capture = new CapturingExporter();
  const Session = makeFakeSessionClass();
  const sdk = { AgentSession: Session };

  // A plain mutable object, exactly as a caller might build and later reuse
  // or mutate it (e.g. a shared config object edited elsewhere in the host
  // app).
  const config = { captureContent: true };

  registerCapturingProvider(capture);
  instrumentPiCodingAgent(sdk, config);

  // Mutate the field AFTER instrumentPiCodingAgent() has already returned.
  // resolveConfig() must have copied the primitive value out (not retained a
  // live reference to `config`).
  config.captureContent = false;

  const session = new Session();
  const done = session.prompt('sensitive prompt text');
  session.emit({ type: 'agent_start' });
  session.emit({
    type: 'agent_end',
    messages: [assistantMessage({ content: [{ type: 'text', text: 'sensitive reply' }] })],
    willRetry: false,
  });
  await done;

  assert.equal(capture.spans.length, 1);
  const [rootSpan] = capture.spans;
  assert.equal(
    attrs(rootSpan!)['input.value'],
    'sensitive prompt text',
    'captureContent must still resolve to its call-time value (true), not the post-call ' +
      'mutation to false — resolveConfig() must copy primitives by value, not hold a live ' +
      'reference to the caller-owned config object',
  );
  assert.equal(attrs(rootSpan!)['output.value'], 'sensitive reply');
});
