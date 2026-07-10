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
