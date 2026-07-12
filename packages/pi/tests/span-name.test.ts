import assert from 'node:assert/strict';
import { test } from 'node:test';
import { describeToolCallSpan } from '../src/span-name';

test('describeToolCallSpan uses the file basename for path-like args', () => {
  assert.equal(describeToolCallSpan('read', { path: '/a/b/app.py' }), 'read: app.py');
  assert.equal(describeToolCallSpan('write', { file: 'notes.md' }), 'write: notes.md');
  assert.equal(describeToolCallSpan('edit', { filePath: 'src/x/y.ts' }), 'edit: y.ts');
  assert.equal(describeToolCallSpan('grep', { target: '/etc/hosts' }), 'grep: hosts');
  assert.equal(describeToolCallSpan('read', { file_path: '/a/b/app.py' }), 'read: app.py');
  assert.equal(describeToolCallSpan('read', { filename: 'notes.md' }), 'read: notes.md');
});

test('describeToolCallSpan reduces a Windows-style path to its filename (no path leak)', () => {
  assert.equal(
    describeToolCallSpan('read', { path: 'C:\\Users\\alice\\secret-project\\app.py' }),
    'read: app.py',
  );
  assert.equal(describeToolCallSpan('edit', { filePath: 'D:\\work\\notes.md' }), 'edit: notes.md');
});

test('describeToolCallSpan summarizes bash by its (whitespace-collapsed) command', () => {
  assert.equal(describeToolCallSpan('bash', { command: '  npm   test ' }), 'bash: npm test');
});

test('describeToolCallSpan truncates a long bash command', () => {
  const name = describeToolCallSpan('bash', { command: `echo ${'x'.repeat(200)}` });
  assert.ok(name.startsWith('bash: '));
  assert.ok(name.endsWith('…'));
  assert.ok(name.length <= 'bash: '.length + 60 + 1);
});

test('describeToolCallSpan falls back to the bare tool name', () => {
  assert.equal(describeToolCallSpan('think', {}), 'think');
  assert.equal(describeToolCallSpan('think', undefined), 'think');
  assert.equal(describeToolCallSpan('bash', { command: '' }), 'bash');
});

test('describeToolCallSpan does not split a surrogate pair at the truncation boundary', () => {
  // 59 ASCII chars + an emoji (a surrogate pair) puts the pair astride the 60-char cut.
  const name = describeToolCallSpan('bash', { command: 'x'.repeat(59) + '\u{1F600}tail' });
  assert.ok(name.startsWith('bash: '));
  assert.ok(name.endsWith('…'));
  const body = name.slice('bash: '.length, -1);
  // A lone high surrogate at the end would be invalid UTF-16; confirm the
  // last code unit is not an unpaired high surrogate.
  const lastCode = body.charCodeAt(body.length - 1);
  assert.ok(lastCode < 0xd800 || lastCode > 0xdbff);
});

test('describeToolCallSpan prefers a path arg over a bash command for non-bash tools', () => {
  assert.equal(describeToolCallSpan('read', { path: '/a.py', command: 'ignored' }), 'read: a.py');
});

test('describeToolCallSpan ignores non-string path-like args', () => {
  assert.equal(describeToolCallSpan('read', { path: 42 }), 'read');
  assert.equal(describeToolCallSpan('read', { path: null }), 'read');
});

test('describeToolCallSpan bounds a path-like arg with no separators the same way it bounds a bash command', () => {
  // basename() only strips separators — a string with none at all (or whose
  // final segment is huge) passes through completely unchanged. Unlike the
  // bash branch (explicitly capped at MAX_BASH_NAME via truncateSurrogateSafe
  // — see the file header's stated privacy/size-bound rationale), the path
  // branch had no equivalent cap: an untrusted/hallucinated "path" argument
  // with no "/" or "\\" could inflate the span NAME itself without bound.
  const hugeNoSeparators = 'x'.repeat(5000);
  const name = describeToolCallSpan('read', { path: hugeNoSeparators });
  assert.ok(
    name.length <= 'read: '.length + 60 + 1,
    `expected the span name to be bounded like the bash branch is, got length ${name.length}`,
  );
  assert.ok(name.startsWith('read: '));
  assert.ok(
    name.endsWith('…'),
    'a truncated path-derived name must carry the same ellipsis marker',
  );
});

test('describeToolCallSpan truncates an over-long basename without splitting a surrogate pair at the boundary', () => {
  // Mirrors the equivalent bash surrogate-pair test above, but for the path
  // branch: a filename component with no separators, long enough to force
  // truncation, with a surrogate pair astride the cut boundary.
  const name = describeToolCallSpan('read', { path: 'x'.repeat(59) + '\u{1F600}tail' });
  assert.ok(name.startsWith('read: '));
  assert.ok(name.endsWith('…'));
  const body = name.slice('read: '.length, -1);
  const lastCode = body.charCodeAt(body.length - 1);
  assert.ok(lastCode < 0xd800 || lastCode > 0xdbff);
});

test('describeToolCallSpan falls back to the bare tool name when a non-empty path-like arg has NO basename component at all (win32.basename reduces a root/drive-only reference to "")', () => {
  // Unlike the args.path === '' case (already covered above, where
  // firstPathArgument itself skips the falsy candidate), these path values
  // are genuinely non-empty and truthy — firstPathArgument happily returns
  // them — but win32.basename() strips them down to nothing because they are
  // ENTIRELY separators (or a bare drive letter) with no filename component
  // to keep. A perfectly ordinary, non-adversarial tool call reading or
  // listing a root directory (`{ path: '/' }`) must not produce a dangling
  // "toolName: " with nothing after the colon — the exact pattern the
  // args.path === '' test above already asserts must never happen.
  assert.equal(describeToolCallSpan('list_dir', { path: '/' }), 'list_dir');
  assert.equal(describeToolCallSpan('read', { path: '\\' }), 'read');
  assert.equal(describeToolCallSpan('read', { path: '///' }), 'read');
  assert.equal(describeToolCallSpan('read', { path: 'C:\\' }), 'read');
  assert.equal(describeToolCallSpan('read', { path: 'C:/' }), 'read');
  // A bash tool call whose ONLY path-like arg reduces to nothing must still
  // fall through to the bash-command branch rather than emitting a dangling
  // "bash: " and ignoring a perfectly good command right next to it.
  assert.equal(
    describeToolCallSpan('bash', { path: '/', command: 'ls -la' }),
    'bash: ls -la',
    'an empty-basename path must not shadow a real bash command in the same args object',
  );
});

test('describeToolCallSpan keeps the bash command when a non-empty-basename path-like arg rides along in the same args object', () => {
  // A bash tool call can carry an incidental path/file/target argument
  // alongside `command` (plausible for tool schemas that add a
  // target/cwd-like field). Unlike the empty-basename case above, this path
  // arg resolves to a real, non-empty basename ("data") — but the bash
  // command must still win, matching the file header's claimed
  // leak-prevention/informativeness tradeoff for bash commands.
  assert.equal(
    describeToolCallSpan('bash', { command: 'rm -rf /data', target: '/data' }),
    'bash: rm -rf /data',
    'a non-empty path-like arg (e.g. target) must not shadow a real bash command',
  );
});
