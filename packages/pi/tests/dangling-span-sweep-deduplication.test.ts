/**
 * Regression guard for a P2 code-review finding on
 * packages/pi/src/instrumentation.ts: the "force-close every open tool
 * span, then the LLM span, then (sometimes) the root span" dangling-span
 * sweep was copy-pasted independently at 4 call sites — agent_start,
 * turn_end, agent_end, and dispose() — each with its own slightly
 * inconsistent `if (span)` / `if (size > 0)` guard, even though
 * closeDanglingSpan() and an empty Map's for/.clear() are already no-ops.
 *
 * This is a structural duplication finding, not a behavioral one: every
 * call site already produces identical spans today no matter how many times
 * the sweep is copy-pasted, so no black-box test driven through
 * instrumentPiCodingAgent() can fail against the current code. The actual
 * risk is a *future* edit (e.g. a new span type added to SessionSpanState)
 * landing at 3 of the 4 near-identical blocks and silently missing the
 * 4th. The only test that can catch that ahead of time is a structural one:
 * assert the sweep is implemented exactly once and reused everywhere, so
 * any future change to sweep order/span types can only be made in one
 * place — matching every other tests/*.test.ts file's black-box convention
 * would not exercise this finding at all, since it has no observable
 * runtime effect.
 */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { test } from 'node:test';

const SOURCE = readFileSync(join(__dirname, '..', 'src', 'instrumentation.ts'), 'utf8');

test('the tool+LLM dangling-span sweep is defined exactly once, not copy-pasted per call site', () => {
  const sweepDefinitions = SOURCE.match(/function sweepDanglingSpans\(/g) ?? [];
  assert.equal(
    sweepDefinitions.length,
    1,
    'expected exactly one sweepDanglingSpans() helper definition — found ' +
      `${sweepDefinitions.length}. The tool-span-sweep-then-llm-span-close block must live in ` +
      'a single shared function, not be copy-pasted at each call site.',
  );
});

test('the shared sweep helper is called from all 4 dangling-span call sites (agent_start, turn_end, agent_end, dispose())', () => {
  const sweepCalls = SOURCE.match(/sweepDanglingSpans\(state/g) ?? [];
  assert.equal(
    sweepCalls.length,
    4,
    'expected sweepDanglingSpans(state, ...) to be called from all 4 sweep sites (agent_start, ' +
      `turn_end, agent_end, dispose()) — found ${sweepCalls.length} call(s). A future change to ` +
      'sweep order or a new span type only needs to touch the shared helper, not be hand-edited ' +
      'at every call site.',
  );
});

test('the raw tool-span-sweep loop no longer appears inline at each call site', () => {
  const inlineToolSweeps =
    SOURCE.match(
      /for \(const span of state\.toolSpans\.values\(\)\) closeDanglingSpan\(span\);/g,
    ) ?? [];
  assert.equal(
    inlineToolSweeps.length,
    1,
    'the `for (const span of state.toolSpans.values()) closeDanglingSpan(span);` loop must ' +
      `appear exactly once (inside the shared helper) — found ${inlineToolSweeps.length} ` +
      'copy-pasted occurrence(s) across the 4 call sites.',
  );
});
