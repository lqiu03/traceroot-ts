/**
 * Lens: the running serialization budget in src/spans.ts must cap NON-string
 * primitives too, not only strings.
 *
 * tool-io-budget-truncation.test.ts covers a large array of small STRINGS. This
 * file covers the complementary hole: a tool result that is a large array of
 * NUMBERS (or booleans) — a numeric grep/find result, a big matrix. No element
 * is a string, so the old strings-only replacer returned every element
 * untouched and JSON.stringify materialized the entire multi-hundred-KB payload
 * before truncateJsonSafe's post-hoc slice ever ran — the exact O(N) blowup the
 * mid-walk budget exists to prevent.
 *
 * Asserting on the final attribute alone cannot tell a fixed implementation
 * from a broken one — truncateJsonSafe bounds the FINAL string either way. So,
 * like tool-io-budget-truncation.test.ts, this spies on JSON.stringify to
 * observe HOW serialization happened: it counts how many of the array's numeric
 * elements the replacer is invoked with. A fixed implementation slices the huge
 * array up front and walks only ~budget elements; the broken one walks every
 * one of them before any cap applies.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { assistantMessage, attrs, makeRig } from './test-helpers';

// Mirrors src/spans.ts's MAX_TOOL_IO_JSON_CHARS (not exported — hardcoded here
// the same way the sibling truncation tests hardcode it).
const MAX_TOOL_IO_JSON_CHARS = 32 * 1024;
const TRUNCATION_MARKER = '…[truncated]';

// Drives one tool call (args in) through the full instrumented event sequence
// so serialization goes through openToolSpan exactly as production does.
async function runToolCall(
  args: unknown,
): Promise<import('@opentelemetry/sdk-trace-base').ReadableSpan> {
  const { capture, Session } = makeRig();
  const session = new Session();
  await session.prompt('run a tool');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'message_start', message: assistantMessage() });
  session.emit({ type: 'message_end', message: assistantMessage() });
  session.emit({ type: 'tool_execution_start', toolCallId: 't1', toolName: 'grep', args });
  session.emit({
    type: 'tool_execution_end',
    toolCallId: 't1',
    toolName: 'grep',
    result: {},
    isError: false,
  });
  session.emit({ type: 'turn_end', message: assistantMessage(), toolResults: [] });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
  const toolSpan = capture.spans.find((s) => attrs(s)['gen_ai.tool.call.id'] === 't1');
  assert.ok(toolSpan, 'expected a tool span to have been exported');
  return toolSpan!;
}

test('a large array of numbers is bounded by a running budget, not fully walked then sliced', async () => {
  // 500_000 numbers: far past the 32 KB char budget, yet no element is a string,
  // so the old per-string cap never fired for any of them. This is the numeric
  // grep/find shape the old post-hoc slice let through fully materialized.
  const ELEMENT_COUNT = 500_000;
  const args = { values: Array.from({ length: ELEMENT_COUNT }, (_v, i) => i) };

  const originalStringify = JSON.stringify;
  let numericElementVisits = 0;

  type Replacer = (this: unknown, key: string, value: unknown) => unknown;
  JSON.stringify = ((
    value: unknown,
    replacer?: Replacer | (string | number)[] | null,
    space?: string | number,
  ) => {
    if (typeof replacer === 'function') {
      const wrapped: Replacer = function wrapped(key, val) {
        // Count every numeric element the replacer is actually invoked with.
        // The bug walks all ELEMENT_COUNT of them; the fix slices the array up
        // front so only ~budget elements are ever visited.
        if (typeof val === 'number') numericElementVisits += 1;
        return replacer.call(this, key, val);
      };
      return originalStringify(value, wrapped, space);
    }
    return originalStringify(value, replacer as (string | number)[] | null | undefined, space);
  }) as typeof JSON.stringify;

  try {
    const toolSpan = await runToolCall(args);
    const inputValue = attrs(toolSpan)['input.value'] as string;

    // The core proof: serialization must NOT visit every one of the 500k
    // elements. A fixed replacer slices the array to at most `remaining`
    // elements before JSON.stringify walks it, so the visit count is bounded by
    // the budget (plus a small slack), independent of the input size.
    assert.ok(
      numericElementVisits <= MAX_TOOL_IO_JSON_CHARS + 2,
      `serialization must bound the number of array elements it walks to ~the ` +
        `${MAX_TOOL_IO_JSON_CHARS}-char budget (visited ${numericElementVisits} of ${ELEMENT_COUNT}) — ` +
        'the old strings-only replacer left every numeric element untouched, letting JSON.stringify ' +
        'fully materialize the whole payload before truncateJsonSafe sliced it',
    );
    // The final attribute must still be a bounded, marked-as-truncated string.
    assert.ok(
      inputValue.endsWith(TRUNCATION_MARKER),
      'the oversized numeric array result must still be marked truncated',
    );
    assert.ok(
      inputValue.length <= MAX_TOOL_IO_JSON_CHARS + TRUNCATION_MARKER.length,
      'the exported input.value must stay within the MAX + marker bound',
    );
  } finally {
    JSON.stringify = originalStringify;
  }
});

test('a large array of booleans is likewise bounded by the running budget', async () => {
  const ELEMENT_COUNT = 500_000;
  const args = { flags: Array.from({ length: ELEMENT_COUNT }, (_v, i) => i % 2 === 0) };

  const originalStringify = JSON.stringify;
  let booleanElementVisits = 0;

  type Replacer = (this: unknown, key: string, value: unknown) => unknown;
  JSON.stringify = ((
    value: unknown,
    replacer?: Replacer | (string | number)[] | null,
    space?: string | number,
  ) => {
    if (typeof replacer === 'function') {
      const wrapped: Replacer = function wrapped(key, val) {
        if (typeof val === 'boolean') booleanElementVisits += 1;
        return replacer.call(this, key, val);
      };
      return originalStringify(value, wrapped, space);
    }
    return originalStringify(value, replacer as (string | number)[] | null | undefined, space);
  }) as typeof JSON.stringify;

  try {
    const toolSpan = await runToolCall(args);
    const inputValue = attrs(toolSpan)['input.value'] as string;

    assert.ok(
      booleanElementVisits <= MAX_TOOL_IO_JSON_CHARS + 2,
      `serialization must bound boolean-element walking to ~the budget (visited ` +
        `${booleanElementVisits} of ${ELEMENT_COUNT})`,
    );
    assert.ok(
      inputValue.length <= MAX_TOOL_IO_JSON_CHARS + TRUNCATION_MARKER.length,
      'the exported input.value must stay within the MAX + marker bound',
    );
  } finally {
    JSON.stringify = originalStringify;
  }
});
