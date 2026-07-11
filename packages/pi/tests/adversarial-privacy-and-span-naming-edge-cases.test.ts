/**
 * Lens: privacy-and-span-naming-edge-cases.
 *
 * Probes describeToolCallSpan's TOOL_PATH_ARGUMENT_KEYS resolution order,
 * its handling of non-object-shaped args, directory-like paths, unusual
 * toolName values, non-ASCII bash commands under the truncation limit, and
 * the captureContent:false gate on the ROOT span for a falsy-but-present
 * ('') prompt vs an absent (undefined) one. None of these are covered by
 * span-name.test.ts or instrumentation.test.ts today (verified by reading
 * both fully before writing this file).
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import type { ExportResult } from '@opentelemetry/core';
import { ExportResultCode } from '@opentelemetry/core';
import type { ReadableSpan, SpanExporter } from '@opentelemetry/sdk-trace-base';
import { instrumentPiCodingAgent } from '../src/instrumentation';
import { describeToolCallSpan } from '../src/span-name';
import type { AgentEvent } from '../src/types';

// Copied locally — no shared state across test files, matching every other
// adversarial-*.test.ts file's explicit convention in this package.
class CapturingExporter implements SpanExporter {
  readonly spans: ReadableSpan[] = [];
  export(spans: ReadableSpan[], resultCallback: (result: ExportResult) => void): void {
    this.spans.push(...spans);
    resultCallback({ code: ExportResultCode.SUCCESS });
  }
  async shutdown(): Promise<void> {}
}

// Fresh class per test, not a shared module-level class — instrumentPiCodingAgent
// patches AgentSession.prototype directly, so reusing one class across tests
// would stack multiple wrap layers onto the same prototype method.
function makeFakeSessionClass() {
  return class FakeAgentSession {
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
  };
}

function attrs(span: ReadableSpan): Record<string, unknown> {
  return span.attributes as Record<string, unknown>;
}

test('multiple simultaneous path-like keys resolve by TOOL_PATH_ARGUMENT_KEYS order, not object insertion order, and an empty-string candidate is skipped', () => {
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

test('array args never leak positional elements as a path and safely fall back to the bare tool name', () => {
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

test('a directory-like path arg (trailing slash) reduces to the directory name, not the full path or an empty string', () => {
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

test("unusual characters in toolName itself pass through unchanged (toolName is Pi's own identifier, not user-controlled args)", () => {
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

test('non-ASCII BMP unicode in a bash command well under the 60-char limit passes through unmangled', () => {
  // Chinese, Cyrillic, and accented Latin characters are all single UTF-16
  // code units (unlike the emoji surrogate-pair case already covered in
  // span-name.test.ts), and the whole collapsed command is far below
  // MAX_BASH_NAME, so no truncation should occur at all — the assertion is
  // that whitespace-collapse does not corrupt or drop any multi-byte chars.
  const command = 'echo   café   北京   Москва  ';
  const collapsed = 'echo café 北京 Москва';
  assert.ok(collapsed.length < 60, 'sanity check: well under the truncation limit');
  const name = describeToolCallSpan('bash', { command });
  assert.equal(name, `bash: ${collapsed}`);
  assert.ok(!name.endsWith('…'), 'must not be truncated when comfortably under the limit');
});

test('captureContent:false suppresses input.value on the ROOT span for an empty-string prompt exactly as for an absent one, while captureContent:true legitimately records the empty string', async () => {
  async function rootInputValue(
    captureContent: boolean,
    promptText: string | undefined,
  ): Promise<{ hasKey: boolean; value: unknown }> {
    const capture = new CapturingExporter();
    const Session = makeFakeSessionClass();
    const sdk = { AgentSession: Session };
    instrumentPiCodingAgent(sdk, { apiKey: 'k', _spanExporter: capture, captureContent });
    const session = new Session();
    // Calling prompt() with a non-string bypasses the pendingInput.set() typeof
    // guard in instrumentation.ts, simulating a caller whose prompt text is
    // genuinely absent (as opposed to the empty-string case below) while still
    // registering the subscribe() listener that instrumentPiCodingAgent wires
    // up inside the wrapped prompt() call itself.
    await session.prompt(promptText as unknown as string);
    session.emit({ type: 'agent_start' });
    session.emit({ type: 'agent_end', messages: [], willRetry: false });
    const root = capture.spans[0]!;
    return {
      hasKey: Object.prototype.hasOwnProperty.call(attrs(root), 'input.value'),
      value: attrs(root)['input.value'],
    };
  }

  const falseEmpty = await rootInputValue(false, '');
  assert.equal(falseEmpty.hasKey, false, 'captureContent:false must omit input.value for ""');
  assert.equal(falseEmpty.value, undefined);

  const falseUndefined = await rootInputValue(false, undefined);
  assert.equal(
    falseUndefined.hasKey,
    false,
    'captureContent:false must omit input.value for undefined',
  );
  assert.equal(falseUndefined.value, undefined);

  // Contrast: with captureContent on, "" is a real (if uninformative) value
  // and must be distinguishable from "no prompt text at all".
  const trueEmpty = await rootInputValue(true, '');
  assert.equal(trueEmpty.hasKey, true, 'captureContent:true legitimately sets input.value to ""');
  assert.equal(trueEmpty.value, '');

  const trueUndefined = await rootInputValue(true, undefined);
  assert.equal(
    trueUndefined.hasKey,
    false,
    'no prompt text at all must leave input.value unset even when captureContent is on',
  );
  assert.equal(trueUndefined.value, undefined);
});
