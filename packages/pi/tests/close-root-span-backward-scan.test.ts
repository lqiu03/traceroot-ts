/**
 * Lens: closeRootSpan's search for the last assistant message (src/spans.ts)
 * — it must scan backward directly rather than copying the whole
 * finalMessages array and reversing it before find() ever runs. Mirrors
 * spans-truncation.test.ts's last test in spirit ('a single huge string tool
 * argument is capped during serialization, not only after the fact'): the
 * *final output* of a copy+reverse+find and a backward for-loop is
 * byte-identical, so asserting only on the exported attribute can't tell a
 * fixed implementation apart from the wasteful one. Instead this spies on
 * Array.prototype.reverse to observe *how* the search happened — a
 * full-length reverse() call is exactly the bug: it means the entire history
 * was copied and reversed up front, even in the common case where the very
 * last message already is the assistant reply.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { SimpleSpanProcessor } from '@opentelemetry/sdk-trace-base';
import type { ExportResult } from '@opentelemetry/core';
import { ExportResultCode } from '@opentelemetry/core';
import type { ReadableSpan, SpanExporter } from '@opentelemetry/sdk-trace-base';
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import { closeRootSpan } from '../src/spans';
import type { AgentMessage, AssistantMessage, UserMessage } from '../src/types';

// Copied locally — no shared state across test files, matching every other
// *.test.ts file's explicit convention in this package.
class CapturingExporter implements SpanExporter {
  readonly spans: ReadableSpan[] = [];
  export(spans: ReadableSpan[], resultCallback: (result: ExportResult) => void): void {
    this.spans.push(...spans);
    resultCallback({ code: ExportResultCode.SUCCESS });
  }
  async shutdown(): Promise<void> {}
}

// closeRootSpan only needs a real Span (setAttribute/end), not the full
// instrumentPiCodingAgent event pipeline — a direct NodeTracerProvider +
// SimpleSpanProcessor rig (same shape as src/provider.ts's createTracing,
// minus the OTLP exporter) is the most direct way to unit-test this
// function in isolation.
function makeTracer() {
  const capture = new CapturingExporter();
  const provider = new NodeTracerProvider();
  provider.addSpanProcessor(new SimpleSpanProcessor(capture));
  const tracer = provider.getTracer('close-root-span-backward-scan-test');
  return { tracer, capture };
}

function userMessage(text: string): UserMessage {
  return { role: 'user', content: text, timestamp: 0 };
}

function assistantMessage(text: string): AssistantMessage {
  return {
    role: 'assistant',
    content: [{ type: 'text', text }],
    api: 'anthropic-messages',
    provider: 'anthropic',
    model: 'claude-sonnet-5',
    usage: {
      input: 10,
      output: 5,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 15,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
    stopReason: 'stop',
    timestamp: 0,
  };
}

function attrs(span: ReadableSpan): Record<string, unknown> {
  return span.attributes as Record<string, unknown>;
}

test('closeRootSpan finds the last assistant message without copying+reversing the full history', () => {
  const { tracer, capture } = makeTracer();
  const span = tracer.startSpan('AgentSession.prompt');

  // Long-running session shape from the finding: many turns accumulated,
  // and the very last message already is the assistant reply — the case a
  // backward scan resolves in one comparison.
  const finalMessages: AgentMessage[] = [];
  for (let i = 0; i < 5000; i++) {
    finalMessages.push(userMessage(`turn ${i}`));
  }
  finalMessages.push(assistantMessage('final answer'));

  const originalReverse = Array.prototype.reverse;
  let reverseCalledOnFullHistory = false;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  Array.prototype.reverse = function reverseSpy(this: any[]) {
    if (this.length === finalMessages.length) {
      reverseCalledOnFullHistory = true;
    }
    return originalReverse.call(this);
  };

  try {
    closeRootSpan(span, finalMessages, false, true);
  } finally {
    Array.prototype.reverse = originalReverse;
  }

  assert.equal(
    reverseCalledOnFullHistory,
    false,
    'closeRootSpan must not copy and reverse the entire finalMessages array to find the last assistant message',
  );

  const exported = capture.spans[0];
  assert.ok(exported, 'expected the root span to have been exported');
  assert.equal(attrs(exported)['output.value'], 'final answer');
});

test('closeRootSpan still finds the last assistant message when later messages are not assistant messages', () => {
  const { tracer, capture } = makeTracer();
  const span = tracer.startSpan('AgentSession.prompt');

  // The assistant reply is buried, not the final entry — a correct backward
  // scan must keep walking past the trailing non-assistant messages instead
  // of stopping at the wrong one.
  const finalMessages: AgentMessage[] = [
    userMessage('question'),
    assistantMessage('the real answer'),
    {
      role: 'toolResult',
      toolCallId: 't1',
      toolName: 'bash',
      content: [],
      isError: false,
      timestamp: 0,
    },
  ];

  closeRootSpan(span, finalMessages, false, true);

  const exported = capture.spans[0];
  assert.ok(exported, 'expected the root span to have been exported');
  assert.equal(attrs(exported)['output.value'], 'the real answer');
});
