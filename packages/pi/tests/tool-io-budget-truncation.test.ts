/**
 * Lens: the running serialization budget in src/spans.ts's stringifyToolIo.
 *
 * spans-truncation.test.ts already covers the SINGLE-huge-string shape (one
 * oversized field capped mid-walk by the per-string cap). This file covers the
 * complementary shape the per-string cap misses entirely: a tool result that
 * is a large ARRAY of many individually-small strings (a big grep/find
 * output). No single element exceeds the cap, so the old per-string replacer
 * left every one of them untouched and JSON.stringify materialized the whole
 * multi-hundred-KB payload before truncateJsonSafe's post-hoc slice ever ran.
 *
 * Asserting on the final attribute alone cannot tell a fixed implementation
 * from a broken one here — truncateJsonSafe bounds the FINAL string either way.
 * So, exactly like spans-truncation.test.ts's single-huge-string test, this
 * spies on JSON.stringify to observe HOW serialization happened: it sums the
 * bytes of the array's line strings that the replacer actually embeds. A fixed
 * implementation stops embedding once its running budget is spent; the buggy
 * one embeds all of them before any cap applies.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { assistantMessage, attrs, makeRig } from './test-helpers';

// Mirrors src/spans.ts's MAX_TOOL_IO_JSON_CHARS (not exported — hardcoded here
// the same way spans-truncation.test.ts hardcodes it).
const MAX_TOOL_IO_JSON_CHARS = 32 * 1024;
const TRUNCATION_MARKER = '…[truncated]';

// Drives one tool call (args in) through the full instrumented event sequence
// so serialization goes through openToolSpan exactly as production does, not a
// direct call to the (unexported) helper.
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

test('a large array of many small strings is capped by a running serialization budget, not fully materialized then sliced', async () => {
  const LINE_LEN = 512;
  // 200 * 512 = ~100 KB total, far past the 32 KB cap — yet each individual
  // line (512 chars) is well under the cap, so the per-string cap never fires
  // for any of them. This is exactly the grep/find shape the old post-hoc
  // slice let through fully materialized.
  const LINE_COUNT = 200;
  const line = 'x'.repeat(LINE_LEN);
  const args = { matches: Array.from({ length: LINE_COUNT }, () => line) };

  const originalStringify = JSON.stringify;
  let embeddedLineChars = 0;

  type Replacer = (this: unknown, key: string, value: unknown) => unknown;
  JSON.stringify = ((
    value: unknown,
    replacer?: Replacer | (string | number)[] | null,
    space?: string | number,
  ) => {
    if (typeof replacer === 'function') {
      const wrapped: Replacer = function wrapped(key, val) {
        const out = replacer.call(this, key, val);
        // Count only the bytes of OUR line strings that the replacer actually
        // embeds (returns as a non-empty string). The bug embeds all
        // LINE_COUNT of them; the fix stops once the running budget is spent
        // and returns '' (or a capped remainder) for the rest.
        if (val === line && typeof out === 'string') embeddedLineChars += out.length;
        return out;
      };
      return originalStringify(value, wrapped, space);
    }
    return originalStringify(value, replacer as (string | number)[] | null | undefined, space);
  }) as typeof JSON.stringify;

  try {
    const toolSpan = await runToolCall(args);
    const inputValue = attrs(toolSpan)['input.value'] as string;

    assert.ok(
      embeddedLineChars <= MAX_TOOL_IO_JSON_CHARS + LINE_LEN,
      `serialization must stop embedding line content once the ${MAX_TOOL_IO_JSON_CHARS}-char ` +
        `budget is spent (embedded ${embeddedLineChars} of ${LINE_LEN * LINE_COUNT} total chars) — ` +
        'the old per-string cap left every small element untouched, letting JSON.stringify fully ' +
        'materialize the whole payload before truncateJsonSafe sliced it',
    );
    // The final attribute must still be a bounded, marked-as-truncated string.
    assert.ok(
      inputValue.endsWith(TRUNCATION_MARKER),
      'the oversized array result must still be marked truncated',
    );
    assert.ok(
      inputValue.length <= MAX_TOOL_IO_JSON_CHARS + TRUNCATION_MARKER.length,
      'the exported input.value must stay within the MAX + marker bound',
    );
  } finally {
    JSON.stringify = originalStringify;
  }
});
