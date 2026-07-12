/**
 * Lens: the shared surrogate-pair-safe slicing primitive that spans.ts's
 * truncateJsonSafe/capOversizedStringValue and span-name.ts's
 * describeToolCallSpan both delegate to. Before this file's fix landed, the
 * boundary-detection algorithm (length check, charCodeAt high-surrogate
 * range test, backoff, slice) was copy-pasted independently into spans.ts's
 * cutSurrogateSafe and span-name.ts's truncateSurrogateSafe — this test
 * targets the single shared implementation directly so a future fix to the
 * boundary math only has one place to land, and a regression here fails
 * regardless of which caller exercises it.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { sliceSurrogateSafe } from '../src/surrogate-safe';

test('sliceSurrogateSafe returns text unchanged when at or under maxLen (<=, not <)', () => {
  assert.equal(sliceSurrogateSafe('hello', 5), 'hello');
  assert.equal(sliceSurrogateSafe('hello', 10), 'hello');
  assert.equal(sliceSurrogateSafe('', 0), '');
});

test('sliceSurrogateSafe slices plainly when no surrogate sits at the cut boundary', () => {
  assert.equal(sliceSurrogateSafe('abcdef', 3), 'abc');
});

test('sliceSurrogateSafe backs off one code unit when a high surrogate sits at the cut boundary', () => {
  // 3 ASCII chars + an emoji (a surrogate pair): cutting at maxLen=4 would
  // otherwise land the boundary exactly on the emoji's high surrogate.
  const emoji = '\u{1F600}';
  const text = `abc${emoji}tail`;
  const sliced = sliceSurrogateSafe(text, 4);
  assert.equal(sliced, 'abc');
  const lastCode = sliced.charCodeAt(sliced.length - 1);
  assert.ok(lastCode < 0xd800 || lastCode > 0xdbff, 'must not end on an unpaired high surrogate');
});

test('sliceSurrogateSafe appends no suffix — callers own their own marker', () => {
  const sliced = sliceSurrogateSafe('abcdefgh', 3);
  assert.equal(sliced, 'abc');
  assert.ok(!sliced.includes('…'), 'sliceSurrogateSafe itself must not add an ellipsis or marker');
});
