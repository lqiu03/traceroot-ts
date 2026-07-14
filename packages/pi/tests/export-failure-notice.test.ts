/**
 * Regression test: a private-mode export that FAILS must surface a
 * console.warn notice, not vanish.
 *
 * When pi builds its own private pipeline, a failing OTLP export (bad base
 * URL / API key, unreachable backend, 4xx/5xx) is reported by the exporter as
 * ExportResultCode.FAILED and the spans are dropped. OTel's own signal for
 * that is diag.warn, which is a NO-OP by default in standalone pi (it never
 * calls diag.setLogger) and is suppressed even under TraceRoot's default
 * logLevel of 'error'. So today the single most common "my traces never
 * arrived" failure is completely silent. pi must surface it through the same
 * console.warn channel it already uses for every other operational warning.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { ExportResultCode } from '@opentelemetry/core';
import type { ExportResult } from '@opentelemetry/core';
import type { ReadableSpan, SpanExporter } from '@opentelemetry/sdk-trace-base';
import { instrumentPiCodingAgent } from '../src/instrumentation';
import { assistantMessage, makeFakeSessionClass } from './test-helpers';
import type { AgentSessionInstance } from '../src/types';

class FailingExporter implements SpanExporter {
  exportCalls = 0;
  export(spans: ReadableSpan[], resultCallback: (result: ExportResult) => void): void {
    this.exportCalls += spans.length;
    resultCallback({ code: ExportResultCode.FAILED, error: new Error('simulated export failure') });
  }
  async shutdown(): Promise<void> {}
}

function instrument(exporter: SpanExporter): AgentSessionInstance {
  const Session = makeFakeSessionClass();
  const sdk = { AgentSession: Session };
  instrumentPiCodingAgent(sdk, { apiKey: 'test-key', _spanExporter: exporter });
  const session = new Session() as unknown as AgentSessionInstance;
  // prompt() is what registers pi's session.subscribe() listener (the patched
  // prototype subscribes synchronously before delegating), so without it the
  // emitted events below reach no handler and no span is ever produced. The
  // returned promise resolves immediately and already carries a .catch handler.
  void session.prompt('warm-up');
  return session;
}

function withCapturedWarn<T>(body: () => T): { result: T; warnings: string[] } {
  const warnings: string[] = [];
  const originalWarn = console.warn;
  console.warn = (...args: unknown[]) => {
    warnings.push(args.map((a) => String(a)).join(' '));
  };
  try {
    return { result: body(), warnings };
  } finally {
    console.warn = originalWarn;
  }
}

// A minimal run: agent_start opens the root span, agent_end closes it, and the
// SimpleSpanProcessor (used whenever _spanExporter is injected) exports that
// span synchronously on end — so the FAILED result is observed by the time
// agent_end returns.
function driveOneRun(session: AgentSessionInstance): void {
  (session as { emit: (e: unknown) => void }).emit({ type: 'agent_start' });
  (session as { emit: (e: unknown) => void }).emit({
    type: 'agent_end',
    messages: [assistantMessage()],
    willRetry: false,
  });
}

test('surfaces a console.warn when the exporter reports a FAILED export result', () => {
  const exporter = new FailingExporter();
  const session = instrument(exporter);

  const { warnings } = withCapturedWarn(() => {
    driveOneRun(session);
  });

  assert.ok(exporter.exportCalls > 0, 'the span should have been handed to the exporter');
  assert.ok(
    warnings.some((w) => w.includes('failed to export') && w.includes('span')),
    'a failed export must surface a console.warn notice instead of vanishing into OTel diag',
  );
  assert.ok(
    warnings.some((w) => w.includes('simulated export failure')),
    'the underlying export error should be surfaced in the warning',
  );
});

test('bounds repeated export-failure warnings so a broken backend cannot flood the console', () => {
  const exporter = new FailingExporter();
  const session = instrument(exporter);

  const totalRuns = 15;
  const { warnings } = withCapturedWarn(() => {
    for (let i = 0; i < totalRuns; i++) driveOneRun(session);
  });

  const failureWarnings = warnings.filter((w) => w.includes('failed to export'));
  assert.equal(exporter.exportCalls, totalRuns, 'every run should still reach the exporter');
  assert.ok(
    failureWarnings.length < totalRuns,
    'repeated identical export failures must be bounded, not emitted once per batch forever',
  );
  assert.ok(
    warnings.some((w) => w.includes('suppressed')),
    'the last emitted warning should note that further warnings are suppressed',
  );
});
