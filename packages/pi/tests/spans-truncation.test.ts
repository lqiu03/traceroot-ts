/**
 * Lens: truncateJsonSafe's boundary logic (src/spans.ts), exercised through
 * the real openToolSpan/closeToolSpan call path rather than as a standalone
 * unit test of the helper — a future refactor that stops calling
 * truncateJsonSafe from either function must fail these tests, not just a
 * test of the helper in isolation. Mirrors span-name.test.ts's surrogate-pair
 * truncation test in spirit (see 'does not split a surrogate pair at the
 * truncation boundary' there) but drives it via the tool span attributes.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import type { ExportResult } from '@opentelemetry/core';
import { ExportResultCode } from '@opentelemetry/core';
import type { ReadableSpan, SpanExporter } from '@opentelemetry/sdk-trace-base';
import { instrumentPiCodingAgent } from '../src/instrumentation';
import type { AgentEvent, AssistantMessage } from '../src/types';

// Mirrors src/spans.ts's MAX_TOOL_IO_JSON_CHARS (not exported — hardcoded
// here the same way span-name.test.ts hardcodes MAX_BASH_NAME as 60).
const MAX_TOOL_IO_JSON_CHARS = 32 * 1024;

// Mirrors src/spans.ts's truncateJsonSafe marker text exactly (single U+2026
// ellipsis, not three ASCII dots).
const TRUNCATION_MARKER = '…[truncated]';

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

// Fresh class per rig, not a shared module-level class — instrumentPiCodingAgent
// patches AgentSession.prototype directly, so reusing one class across tests
// would stack multiple wrap layers onto the same prototype method.
function makeRig(config: { captureToolIo?: boolean } = {}) {
  const capture = new CapturingExporter();

  class FakeAgentSession {
    sessionId = 'sess-1';
    private listeners: Array<(event: AgentEvent) => void> = [];
    async prompt(_text: string, _options?: unknown): Promise<void> {}
    subscribe(listener: (event: AgentEvent) => void): () => void {
      this.listeners.push(listener);
      return () => {
        this.listeners = this.listeners.filter((l) => l !== listener);
      };
    }
    emit(event: AgentEvent): void {
      for (const listener of this.listeners) listener(event);
    }
  }

  const sdk = { AgentSession: FakeAgentSession };
  instrumentPiCodingAgent(sdk, { apiKey: 'test-key', _spanExporter: capture, ...config });

  return { capture, Session: FakeAgentSession };
}

function assistantMessage(overrides: Partial<AssistantMessage> = {}): AssistantMessage {
  return {
    role: 'assistant',
    content: [{ type: 'text', text: 'done' }],
    api: 'anthropic-messages',
    provider: 'anthropic',
    model: 'claude-sonnet-5',
    usage: {
      input: 100,
      output: 20,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 120,
      cost: { input: 0.001, output: 0.0006, cacheRead: 0, cacheWrite: 0, total: 0.0016 },
    },
    stopReason: 'stop',
    timestamp: 0,
    ...overrides,
  } as AssistantMessage;
}

function attrs(span: ReadableSpan): Record<string, unknown> {
  return span.attributes as Record<string, unknown>;
}

// Drives one tool call (args in, result out) through the full instrumented
// event sequence — agent_start through agent_end — and returns the exported
// TOOL span. This is the "real call path" the task asks for: it goes through
// instrumentPiCodingAgent's event handler, which calls openToolSpan/
// closeToolSpan itself, rather than calling those functions directly.
async function runToolCall(
  args: unknown,
  result: unknown,
  config: { captureToolIo?: boolean } = {},
): Promise<ReadableSpan> {
  const { capture, Session } = makeRig(config);
  const session = new Session();

  await session.prompt('run a tool');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'message_start', message: assistantMessage() });
  session.emit({ type: 'message_end', message: assistantMessage() });
  session.emit({ type: 'tool_execution_start', toolCallId: 't1', toolName: 'bash', args });
  session.emit({
    type: 'tool_execution_end',
    toolCallId: 't1',
    toolName: 'bash',
    result,
    isError: false,
  });
  session.emit({ type: 'turn_end', message: assistantMessage(), toolResults: [] });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });

  const toolSpan = capture.spans.find(
    (s) => (s.attributes as Record<string, unknown>)['gen_ai.tool.call.id'] === 't1',
  );
  assert.ok(toolSpan, 'expected a tool span to have been exported');
  return toolSpan!;
}

// JSON.stringify overhead for a single-key { data: '' } object, computed at
// runtime rather than hardcoded — derives the exact prefix/suffix length
// (`{"data":"` + `"}`) so payload builders below stay correct even if this
// ever changes, instead of silently miscalculating.
const JSON_WRAPPER_OVERHEAD = JSON.stringify({ data: '' }).length;

// Builds { data: <filler> } whose JSON.stringify(...) is exactly totalLength
// UTF-16 code units long.
function payloadOfExactJsonLength(totalLength: number, filler = 'x'): { data: string } {
  const fillerLength = totalLength - JSON_WRAPPER_OVERHEAD;
  assert.ok(fillerLength >= 0, 'requested JSON length is smaller than the wrapper overhead');
  return { data: filler.repeat(fillerLength) };
}

// Builds { data: ... } whose JSON.stringify(...) places a surrogate pair (an
// emoji) astride the exact truncateJsonSafe cut boundary: the high surrogate
// lands at character index MAX_TOOL_IO_JSON_CHARS - 1 (0-indexed), i.e.
// exactly the last character truncateJsonSafe would otherwise keep.
function payloadWithSurrogateAtBoundary(): { data: string } {
  const emoji = '\u{1F600}'; // 2 UTF-16 code units (a surrogate pair).
  const prefixLen = JSON_WRAPPER_OVERHEAD - '"}'.length; // chars before the string value starts: `{"data":"`.
  const fillerLength = MAX_TOOL_IO_JSON_CHARS - 1 - prefixLen;
  return { data: 'x'.repeat(fillerLength) + emoji + 'tail' };
}

test('a tool arg/result under the cap passes through openToolSpan/closeToolSpan unchanged', async () => {
  const args = { command: 'echo hello', note: 'small arg' };
  const result = { content: [{ type: 'text', text: 'hello' }] };
  const toolSpan = await runToolCall(args, result);

  assert.equal(attrs(toolSpan)['input.value'], JSON.stringify(args));
  assert.equal(attrs(toolSpan)['output.value'], JSON.stringify(result));
  assert.ok(!String(attrs(toolSpan)['input.value']).includes(TRUNCATION_MARKER));
  assert.ok(!String(attrs(toolSpan)['output.value']).includes(TRUNCATION_MARKER));
});

test('a tool arg/result exactly at the cap boundary passes through unchanged (<=, not <)', async () => {
  const args = payloadOfExactJsonLength(MAX_TOOL_IO_JSON_CHARS);
  const result = payloadOfExactJsonLength(MAX_TOOL_IO_JSON_CHARS, 'y');
  assert.equal(JSON.stringify(args).length, MAX_TOOL_IO_JSON_CHARS, 'sanity check on args length');
  assert.equal(
    JSON.stringify(result).length,
    MAX_TOOL_IO_JSON_CHARS,
    'sanity check on result length',
  );

  const toolSpan = await runToolCall(args, result);

  assert.equal(attrs(toolSpan)['input.value'], JSON.stringify(args));
  assert.equal(attrs(toolSpan)['output.value'], JSON.stringify(result));
  assert.equal((attrs(toolSpan)['input.value'] as string).length, MAX_TOOL_IO_JSON_CHARS);
  assert.equal((attrs(toolSpan)['output.value'] as string).length, MAX_TOOL_IO_JSON_CHARS);
  assert.ok(!(attrs(toolSpan)['input.value'] as string).includes(TRUNCATION_MARKER));
  assert.ok(!(attrs(toolSpan)['output.value'] as string).includes(TRUNCATION_MARKER));
});

test('a tool arg/result over the cap is truncated with the marker appended, total length within bound', async () => {
  const args = payloadOfExactJsonLength(MAX_TOOL_IO_JSON_CHARS + 1000);
  const result = payloadOfExactJsonLength(MAX_TOOL_IO_JSON_CHARS + 500, 'y');

  const toolSpan = await runToolCall(args, result);

  const inputValue = attrs(toolSpan)['input.value'] as string;
  const outputValue = attrs(toolSpan)['output.value'] as string;

  assert.ok(inputValue.endsWith(TRUNCATION_MARKER), 'input.value must end with the marker');
  assert.ok(outputValue.endsWith(TRUNCATION_MARKER), 'output.value must end with the marker');

  // Neither payload has a surrogate at the boundary (plain ASCII filler), so
  // the cut lands exactly at MAX_TOOL_IO_JSON_CHARS and the total length is
  // bounded by MAX + the marker's own length — nowhere near the untruncated
  // (MAX + 1000 / MAX + 500) length.
  assert.ok(
    inputValue.length <= MAX_TOOL_IO_JSON_CHARS + TRUNCATION_MARKER.length,
    'input.value total length must stay within MAX + marker bound',
  );
  assert.ok(
    outputValue.length <= MAX_TOOL_IO_JSON_CHARS + TRUNCATION_MARKER.length,
    'output.value total length must stay within MAX + marker bound',
  );
  assert.equal(inputValue.length, MAX_TOOL_IO_JSON_CHARS + TRUNCATION_MARKER.length);
  assert.equal(outputValue.length, MAX_TOOL_IO_JSON_CHARS + TRUNCATION_MARKER.length);
  assert.equal(
    inputValue,
    JSON.stringify(args).slice(0, MAX_TOOL_IO_JSON_CHARS) + TRUNCATION_MARKER,
  );
  assert.equal(
    outputValue,
    JSON.stringify(result).slice(0, MAX_TOOL_IO_JSON_CHARS) + TRUNCATION_MARKER,
  );
});

test('a UTF-16 surrogate pair astride the exact truncation boundary is not split into a lone surrogate', async () => {
  const args = payloadWithSurrogateAtBoundary();
  const result = payloadWithSurrogateAtBoundary();
  const rawArgsJson = JSON.stringify(args);
  assert.ok(
    rawArgsJson.length > MAX_TOOL_IO_JSON_CHARS,
    'sanity check: payload must actually exceed the cap to exercise truncation',
  );
  // Sanity check the fixture actually puts a high surrogate exactly at the
  // cut boundary (charCodeAt(MAX - 1)) — otherwise this test would pass
  // trivially without ever exercising the surrogate-safe backoff branch.
  const boundaryCode = rawArgsJson.charCodeAt(MAX_TOOL_IO_JSON_CHARS - 1);
  assert.ok(
    boundaryCode >= 0xd800 && boundaryCode <= 0xdbff,
    'fixture must place a high surrogate exactly at index MAX_TOOL_IO_JSON_CHARS - 1',
  );

  const toolSpan = await runToolCall(args, result);
  const inputValue = attrs(toolSpan)['input.value'] as string;
  const outputValue = attrs(toolSpan)['output.value'] as string;

  for (const value of [inputValue, outputValue]) {
    assert.ok(value.endsWith(TRUNCATION_MARKER), 'value must end with the marker');
    const body = value.slice(0, -TRUNCATION_MARKER.length);
    // The backoff must have dropped the whole emoji (and everything after
    // it), landing one code unit short of MAX_TOOL_IO_JSON_CHARS.
    assert.equal(body.length, MAX_TOOL_IO_JSON_CHARS - 1);
    const lastCode = body.charCodeAt(body.length - 1);
    assert.ok(
      lastCode < 0xd800 || lastCode > 0xdbff,
      'the last code unit before the marker must not be an unpaired high surrogate',
    );
    assert.ok(!body.includes('\u{1F600}'), 'the emoji itself must have been dropped, not split');
  }
});

test('captureToolIo: false still bypasses truncateJsonSafe entirely (no marker, no attribute)', async () => {
  const args = payloadOfExactJsonLength(MAX_TOOL_IO_JSON_CHARS + 1000);
  const toolSpan = await runToolCall(args, {}, { captureToolIo: false });

  assert.equal(attrs(toolSpan)['input.value'], undefined);
  assert.equal(attrs(toolSpan)['output.value'], undefined);
});
