/**
 * Lens: src/spans.ts's stringifyToolIo (capFieldReplacer) / truncateJsonSafe
 * boundary logic, exercised through the real openToolSpan/closeToolSpan call
 * path rather than as a standalone unit test of the helpers — a future
 * refactor that stops calling truncateJsonSafe from either function must fail
 * these tests, not just a test of the helper in isolation. Mirrors
 * span-name.test.ts's surrogate-pair truncation test in spirit (see 'does not
 * split a surrogate pair at the truncation boundary' there) but drives it via
 * the tool span attributes.
 *
 * capFieldReplacer only caps an individually-oversized STRING field as
 * JSON.stringify visits it — a single huge string field (a big file read,
 * long command stdout, the dominant real-world shape) is capped mid-walk,
 * before it is embedded in the growing output. It deliberately carries no
 * running budget across the whole payload (that machinery — mid-serialization
 * array slicing and scalar-width charging — was removed as part of the Ask 3a
 * simplification): a large ARRAY of many individually-small values (a big
 * grep/find output, or a numeric/boolean array) now transiently serializes in
 * full before truncateJsonSafe's post-hoc backstop slices the final string —
 * an accepted O(N) trade on already-materialized data. The tests below for
 * that shape assert the surviving final char-cap/marker invariant rather than
 * the removed mid-walk behavior; see each test's own comment.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { trace } from '@opentelemetry/api';
import type { ExportResult } from '@opentelemetry/core';
import { ExportResultCode } from '@opentelemetry/core';
import { SimpleSpanProcessor } from '@opentelemetry/sdk-trace-base';
import type { ReadableSpan, SpanExporter } from '@opentelemetry/sdk-trace-base';
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import { instrumentPiCodingAgent } from '../src/pi/instrumentation';
import type { AgentEvent, AssistantMessage } from '../src/pi/types';

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
  }

  const sdk = { AgentSession: FakeAgentSession };
  // Real global provider per rig, not a private exporter injection —
  // PiInstrumentationConfig no longer has an apiKey/_spanExporter escape
  // hatch (see packages/traceroot/src/pi/config.ts); the in-tree
  // integration always re-resolves its tracer through the OTel API's
  // global `trace` facade. trace.disable() first clears any prior rig's
  // registration so this file's 11 sequential makeRig() calls stay
  // isolated from one another (see pi-test-helpers.ts's makeRig() for the
  // full rationale, mirrored here since this file keeps its own local rig).
  trace.disable();
  const provider = new NodeTracerProvider();
  provider.addSpanProcessor(new SimpleSpanProcessor(capture));
  provider.register();
  instrumentPiCodingAgent(sdk, config);

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

  const done = session.prompt('run a tool');
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
  await done;

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

// A single huge string field (e.g. a big file read, or long command stdout —
// the dominant real-world shape of oversized tool I/O) must never be fully
// materialized by JSON.stringify before truncateJsonSafe's post-hoc cap runs.
// truncateJsonSafe alone already bounds the *final* attribute either way — a
// naive JSON.stringify(args) then slice(0, MAX) produces byte-identical
// output to a properly-capped serialization for this single-field shape, so
// asserting only on the final attribute's length can't tell a fixed
// implementation apart from a broken one here. Instead this spies on the
// global JSON.stringify to observe *how* serialization happened: a bare
// JSON.stringify(args) call (no replacer) covering the huge value is exactly
// the bug — it means the full string was embedded in the output before any
// cap applied. A fixed implementation must instead pass a replacer that caps
// the oversized string as JSON.stringify visits it, before it's embedded.
test('a single huge string tool argument is capped during serialization, not only after the fact', async () => {
  const HUGE_LEN = 500 * 1024; // 500 KB — far past MAX_TOOL_IO_JSON_CHARS.
  const hugeValue = 'A'.repeat(HUGE_LEN);
  const args = { fileContent: hugeValue };

  const originalStringify = JSON.stringify;
  let sawBareStringifyOfHugeValue = false;
  let sawReplacerCapTheHugeValue = false;

  type Replacer = (this: unknown, key: string, value: unknown) => unknown;
  JSON.stringify = ((
    value: unknown,
    replacer?: Replacer | (string | number)[] | null,
    space?: string | number,
  ) => {
    const containsHugeValue =
      typeof value === 'object' &&
      value !== null &&
      Object.values(value as Record<string, unknown>).includes(hugeValue);

    if (containsHugeValue && typeof replacer !== 'function') {
      // The bug: an object containing the huge string was handed to
      // JSON.stringify with no replacer, so the full 500KB string gets
      // embedded in the serialized output before truncateJsonSafe ever runs.
      sawBareStringifyOfHugeValue = true;
    }

    if (typeof replacer === 'function') {
      const wrapped: Replacer = function wrapped(key, val) {
        const capped = replacer.call(this, key, val);
        if (val === hugeValue && typeof capped === 'string' && capped.length < hugeValue.length) {
          sawReplacerCapTheHugeValue = true;
        }
        return capped;
      };
      return originalStringify(value, wrapped, space);
    }
    return originalStringify(value, replacer as (string | number)[] | null | undefined, space);
  }) as typeof JSON.stringify;

  try {
    const toolSpan = await runToolCall(args, {});
    const inputValue = attrs(toolSpan)['input.value'] as string;

    assert.ok(
      !sawBareStringifyOfHugeValue,
      'the huge string argument must never be handed to a bare JSON.stringify(args) call with no replacer — that fully materializes it before truncation can cap it',
    );
    assert.ok(
      sawReplacerCapTheHugeValue,
      "openToolSpan's serialization must cap an oversized string value via JSON.stringify's replacer as it is visited, before it is embedded in the growing output",
    );
    assert.ok(
      inputValue.length <= MAX_TOOL_IO_JSON_CHARS + TRUNCATION_MARKER.length,
      'exported input.value must stay bounded even for a 500KB argument',
    );
    assert.ok(inputValue.endsWith(TRUNCATION_MARKER));
  } finally {
    JSON.stringify = originalStringify;
  }
});

// A parameterless tool call (event.args === undefined) or a void-returning
// tool (event.result === undefined) are both plausible at runtime — AgentEvent
// types args/result as `unknown`, and captureToolIo defaults to true. Per
// spec, JSON.stringify(undefined, replacer) returns the *value* undefined,
// not the string "undefined" — a gap TypeScript's lib.es5.d.ts papers over by
// always typing JSON.stringify's return as `string`. Handing that straight to
// truncateJsonSafe(...) throws a TypeError on `.length`, silently swallowed
// by openToolSpan/closeToolSpan's catch — whose comment claims it exists only
// for "circular refs or BigInt", which this isn't. This spies on the global
// JSON.stringify the same way the huge-string test above does, but asserts
// the opposite: a correct implementation must recognize a literal `undefined`
// itself and never hand it to JSON.stringify in the first place, exactly like
// packages/traceroot/src/claude-agent-sdk.ts's tryStringify already does.
test('undefined args/result (parameterless tool call / void-returning tool) never reach a bare JSON.stringify(undefined) call, and produce no input.value/output.value attribute', async () => {
  const originalStringify = JSON.stringify;
  let stringifyCalledWithUndefined = false;

  type Replacer = (this: unknown, key: string, value: unknown) => unknown;
  JSON.stringify = ((
    value: unknown,
    replacer?: Replacer | (string | number)[] | null,
    space?: string | number,
  ) => {
    if (value === undefined) stringifyCalledWithUndefined = true;
    return originalStringify(value, replacer as (string | number)[] | null | undefined, space);
  }) as typeof JSON.stringify;

  try {
    const toolSpan = await runToolCall(undefined, undefined);

    assert.ok(
      !stringifyCalledWithUndefined,
      'stringifyToolIo must recognize a literal undefined value itself and short-circuit before ever calling JSON.stringify(undefined, ...) — JSON.stringify(undefined, replacer) returns the value undefined (not a string), and handing that to truncateJsonSafe throws a TypeError that gets silently swallowed under a misleading "circular refs or BigInt" comment',
    );
    assert.equal(
      attrs(toolSpan)['input.value'],
      undefined,
      'a parameterless tool call has nothing meaningful to capture as input.value',
    );
    assert.equal(
      attrs(toolSpan)['output.value'],
      undefined,
      'a void-returning tool has nothing meaningful to capture as output.value',
    );
  } finally {
    JSON.stringify = originalStringify;
  }
});

// REPHRASED for Ask 3a (was: 'a large array of many small strings is capped
// by a running serialization budget, not fully materialized then sliced').
// That assertion pinned the now-intentionally-dropped mid-serialization
// budget (proactive array slicing / scalar-width charging) — capFieldReplacer
// only caps individually-oversized strings, so a large array of many
// individually-small strings is now fully materialized by JSON.stringify
// before truncateJsonSafe's post-hoc backstop slices the final string (the
// accepted O(N) trade documented in this file's header). What must still hold
// is the final char-cap/marker invariant, and that the truncated body is a
// verbatim prefix of the real serialization (not corrupted or re-derived).
test('a large array of many small strings still exports a bounded, marked-truncated input.value', async () => {
  const LINE_LEN = 512;
  // 200 * 512 = ~100 KB total, far past the 32 KB cap — yet each individual
  // line (512 chars) is well under the cap, so the per-field cap never fires
  // for any of them.
  const LINE_COUNT = 200;
  const line = 'x'.repeat(LINE_LEN);
  const args = { matches: Array.from({ length: LINE_COUNT }, () => line) };
  const rawJson = JSON.stringify(args);
  assert.ok(
    rawJson.length > MAX_TOOL_IO_JSON_CHARS,
    'sanity check: the raw payload dwarfs the cap',
  );

  const toolSpan = await runToolCall(args, {});
  const inputValue = attrs(toolSpan)['input.value'] as string;

  assert.ok(
    inputValue.endsWith(TRUNCATION_MARKER),
    'the oversized array result must still be marked truncated',
  );
  assert.equal(inputValue.length, MAX_TOOL_IO_JSON_CHARS + TRUNCATION_MARKER.length);
  assert.equal(
    inputValue,
    rawJson.slice(0, MAX_TOOL_IO_JSON_CHARS) + TRUNCATION_MARKER,
    'the truncated body must be a verbatim prefix of the full serialization',
  );
});

// REPHRASED for Ask 3a (was: 'a large array of numbers is bounded by a
// running budget, not fully walked then sliced'). Same reasoning as the
// small-strings test above: a large array of numbers has no oversized string
// element, so capFieldReplacer leaves it untouched and it is fully walked by
// JSON.stringify before truncateJsonSafe's post-hoc backstop slices the final
// string. The surviving requirement is the final char-cap/marker invariant.
test('a large array of numbers still exports a bounded, marked-truncated input.value', async () => {
  const ELEMENT_COUNT = 500_000;
  const args = { values: Array.from({ length: ELEMENT_COUNT }, (_v, i) => i) };
  const rawJson = JSON.stringify(args);
  assert.ok(
    rawJson.length > MAX_TOOL_IO_JSON_CHARS,
    'sanity check: the raw payload dwarfs the cap',
  );

  const toolSpan = await runToolCall(args, {});
  const inputValue = attrs(toolSpan)['input.value'] as string;

  assert.ok(
    inputValue.endsWith(TRUNCATION_MARKER),
    'the oversized numeric array result must still be marked truncated',
  );
  assert.equal(inputValue.length, MAX_TOOL_IO_JSON_CHARS + TRUNCATION_MARKER.length);
  assert.equal(
    inputValue,
    rawJson.slice(0, MAX_TOOL_IO_JSON_CHARS) + TRUNCATION_MARKER,
    'the truncated body must be a verbatim prefix of the full serialization',
  );
});

// REPHRASED for Ask 3a (was: 'a large array of booleans is likewise bounded
// by the running budget'). Same reasoning as the two tests above.
test('a large array of booleans still exports a bounded, marked-truncated input.value', async () => {
  const ELEMENT_COUNT = 500_000;
  const args = { flags: Array.from({ length: ELEMENT_COUNT }, (_v, i) => i % 2 === 0) };
  const rawJson = JSON.stringify(args);
  assert.ok(
    rawJson.length > MAX_TOOL_IO_JSON_CHARS,
    'sanity check: the raw payload dwarfs the cap',
  );

  const toolSpan = await runToolCall(args, {});
  const inputValue = attrs(toolSpan)['input.value'] as string;

  assert.ok(
    inputValue.endsWith(TRUNCATION_MARKER),
    'the oversized boolean array result must still be marked truncated',
  );
  assert.equal(inputValue.length, MAX_TOOL_IO_JSON_CHARS + TRUNCATION_MARKER.length);
  assert.equal(
    inputValue,
    rawJson.slice(0, MAX_TOOL_IO_JSON_CHARS) + TRUNCATION_MARKER,
    'the truncated body must be a verbatim prefix of the full serialization',
  );
});

test('a flat object with a huge number of SHORT-valued keys still exports a bounded input.value (the post-hoc backstop must hold the line)', async () => {
  // capFieldReplacer only caps individually-oversized STRING values — none of
  // this object's values are oversized, and its keys are never inspected at
  // all, so a many-keyed object (a word-count map, a file->stat dictionary)
  // is fully materialized by JSON.stringify. The exported attribute must
  // nevertheless respect the hard MAX + marker bound via truncateJsonSafe's
  // post-hoc backstop.
  const KEY_COUNT = 50_000;
  const args: Record<string, string> = {};
  for (let i = 0; i < KEY_COUNT; i++) {
    args[`key_${i}`] = 'v';
  }
  assert.ok(
    JSON.stringify(args).length > MAX_TOOL_IO_JSON_CHARS * 3,
    'sanity check: the raw payload dwarfs the cap',
  );

  const toolSpan = await runToolCall(args, { ok: true });
  const inputValue = attrs(toolSpan)['input.value'] as string;

  assert.ok(
    inputValue.length <= MAX_TOOL_IO_JSON_CHARS + TRUNCATION_MARKER.length,
    `the exported input.value must stay within the MAX + marker bound (got ${inputValue.length})`,
  );
  assert.ok(inputValue.endsWith(TRUNCATION_MARKER), 'the cut must carry the truncation marker');
});
