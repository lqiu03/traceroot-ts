import assert from 'node:assert/strict';
import { test } from 'node:test';
import { describeToolCallSpan } from '../src/pi/spans';

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

test('describeToolCallSpan resolves multiple simultaneous path-like keys by TOOL_PATH_ARGUMENT_KEYS order, not object insertion order, and skips an empty-string candidate', () => {
  // 'target' is inserted first in the object literal but 'path' outranks it
  // in TOOL_PATH_ARGUMENT_KEYS — the winner must be decided by the fixed key
  // list, not by whichever property happens to appear first in the object.
  assert.equal(
    describeToolCallSpan('grep', { target: '/z.txt', path: '/x.txt', filename: 'y.txt' }),
    'grep: x.txt',
    'path outranks target and filename regardless of property insertion order',
  );
  // With 'path' absent, 'file' (earlier in the key list) must beat 'filePath'.
  assert.equal(
    describeToolCallSpan('grep', { filePath: '/should-lose.txt', file: '/x.txt' }),
    'grep: x.txt',
  );
  // An empty string at the highest-priority key ('path') is falsy and must be
  // skipped in favor of the next candidate, not treated as "no path found"
  // for the whole args object.
  assert.equal(
    describeToolCallSpan('read', { path: '', file: 'notes.md' }),
    'read: notes.md',
    'an empty-string path must fall through to the next path-like key, not win by being present',
  );
  // If every path-like key is empty/absent, fall back to the bare tool name
  // (no dangling "toolName: " with nothing after the colon).
  assert.equal(describeToolCallSpan('read', { path: '' }), 'read');
});

test('describeToolCallSpan never leaks positional array elements as a path and safely falls back to the bare tool name', () => {
  // Arrays are typeof 'object' in JS, so they pass describeToolCallSpan's
  // `args && typeof args === 'object'` guard. TOOL_PATH_ARGUMENT_KEYS are all
  // named string keys ('path', 'file', ...), which a plain array never has as
  // own properties, so this must degrade to the bare tool name rather than
  // reading array indices or throwing.
  assert.equal(
    describeToolCallSpan('read', ['/etc/passwd', 'ignored']),
    'read',
    'an array must never be scanned for a path-like value by index',
  );
  // The bash-specific branch reads args.command; an array has no .command
  // own property either, so this must also degrade safely instead of
  // crashing or stringifying the array into the span name.
  assert.equal(describeToolCallSpan('bash', ['ls', '-la']), 'bash');
  assert.equal(describeToolCallSpan('bash', []), 'bash');
});

test('describeToolCallSpan reduces a directory-like path arg (trailing slash) to the directory name, not the full path or an empty string', () => {
  assert.equal(
    describeToolCallSpan('read', { path: '/a/b/secret-dir/' }),
    'read: secret-dir',
    'trailing slash must not defeat basename reduction (POSIX-style)',
  );
  assert.equal(
    describeToolCallSpan('edit', { filePath: 'C:\\Users\\alice\\project\\' }),
    'edit: project',
    'trailing backslash must not defeat basename reduction (Windows-style)',
  );
  // Nested directory path with no filename component at all.
  assert.equal(describeToolCallSpan('list', { target: '/var/log/app/' }), 'list: app');
});

test("describeToolCallSpan passes unusual characters in toolName itself through unchanged (toolName is Pi's own identifier, not user-controlled args)", () => {
  // MCP-style namespaced tool identifiers use double underscores and must
  // survive untouched when no path/command arg triggers reduction.
  assert.equal(
    describeToolCallSpan('mcp__filesystem__read_file', {}),
    'mcp__filesystem__read_file',
  );
  // A toolName containing a newline is not sanitized by describeToolCallSpan
  // (only args are privacy-reduced) — confirm this doesn't crash and the
  // path reduction still applies on top of it.
  assert.equal(describeToolCallSpan('weird\ntool', { path: '/a/b/c.txt' }), 'weird\ntool: c.txt');
  assert.equal(describeToolCallSpan('', { path: '/a/b/c.txt' }), ': c.txt');
});

test('describeToolCallSpan passes non-ASCII BMP unicode in a bash command well under the 60-char limit through unmangled', () => {
  // Chinese, Cyrillic, and accented Latin characters are all single UTF-16
  // code units (unlike the emoji surrogate-pair case covered above), and the
  // whole collapsed command is far below MAX_BASH_NAME, so no truncation
  // should occur at all — the assertion is that whitespace-collapse does not
  // corrupt or drop any multi-byte chars.
  const command = 'echo   café   北京   Москва  ';
  const collapsed = 'echo café 北京 Москва';
  assert.ok(collapsed.length < 60, 'sanity check: well under the truncation limit');
  const name = describeToolCallSpan('bash', { command });
  assert.equal(name, `bash: ${collapsed}`);
  assert.ok(!name.endsWith('…'), 'must not be truncated when comfortably under the limit');
});

test('describeToolCallSpan treats a whitespace-only bash command as absent — falls back to a path arg or the bare tool name, never emitting a dangling "bash: "', () => {
  // A truthy-but-blank command ('   ') survives the `typeof === "string" &&
  // a.command` truthiness check, but collapses to '' after the whitespace
  // normalization — the inner `if (cmd)` guard must then fall through rather
  // than returning 'bash: ' with nothing after the colon.
  assert.equal(describeToolCallSpan('bash', { command: '   ' }), 'bash');
  assert.equal(describeToolCallSpan('bash', { command: ' \t\n ' }), 'bash');
  // With a path-like arg riding along, the fall-through continues into the
  // path branch instead of stopping at the bare name.
  assert.equal(describeToolCallSpan('bash', { command: '   ', path: '/a/b/c.txt' }), 'bash: c.txt');
});
