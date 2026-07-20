import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { ROOT_CONTEXT, SpanStatusCode, trace } from '@opentelemetry/api';
import type { ReadableSpan } from '@opentelemetry/sdk-trace-base';
import { SimpleSpanProcessor } from '@opentelemetry/sdk-trace-base';
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import {
  resolveConfig,
  instrumentPiCodingAgent,
  describeToolCallSpan,
  openRootSpan,
  stampRootOutput,
  finalizeRootSpan,
  openLlmSpan,
  closeLlmSpan,
  openToolSpan,
  closeToolSpan,
  closeDanglingSpan,
  sliceSurrogateSafe,
  type AgentEvent,
  type AgentMessage,
  type AssistantMessage,
} from '../src/pi';
import { CapturingExporter } from './pi-test-helpers';

describe('span name', () => {
  it('describeToolCallSpan uses the file basename for path-like args', () => {
    assert.equal(describeToolCallSpan('read', { path: '/a/b/app.py' }), 'read: app.py');
    assert.equal(describeToolCallSpan('write', { file: 'notes.md' }), 'write: notes.md');
    assert.equal(describeToolCallSpan('edit', { filePath: 'src/x/y.ts' }), 'edit: y.ts');
    assert.equal(describeToolCallSpan('grep', { target: '/etc/hosts' }), 'grep: hosts');
    assert.equal(describeToolCallSpan('read', { file_path: '/a/b/app.py' }), 'read: app.py');
    assert.equal(describeToolCallSpan('read', { filename: 'notes.md' }), 'read: notes.md');
  });

  it('describeToolCallSpan reduces a Windows-style path to its filename (no path leak)', () => {
    assert.equal(
      describeToolCallSpan('read', { path: 'C:\\Users\\alice\\secret-project\\app.py' }),
      'read: app.py',
    );
    assert.equal(
      describeToolCallSpan('edit', { filePath: 'D:\\work\\notes.md' }),
      'edit: notes.md',
    );
  });

  it('describeToolCallSpan summarizes bash by its (whitespace-collapsed) command', () => {
    assert.equal(describeToolCallSpan('bash', { command: '  npm   test ' }), 'bash: npm test');
  });

  it('describeToolCallSpan truncates a long bash command', () => {
    const name = describeToolCallSpan('bash', { command: `echo ${'x'.repeat(200)}` });
    assert.ok(name.startsWith('bash: '));
    assert.ok(name.endsWith('…'));
    assert.ok(name.length <= 'bash: '.length + 60 + 1);
  });

  it('describeToolCallSpan falls back to the bare tool name', () => {
    assert.equal(describeToolCallSpan('think', {}), 'think');
    assert.equal(describeToolCallSpan('think', undefined), 'think');
    assert.equal(describeToolCallSpan('bash', { command: '' }), 'bash');
  });

  it('describeToolCallSpan does not split a surrogate pair at the truncation boundary', () => {
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

  it('describeToolCallSpan ignores non-string path-like args', () => {
    assert.equal(describeToolCallSpan('read', { path: 42 }), 'read');
    assert.equal(describeToolCallSpan('read', { path: null }), 'read');
  });

  it('describeToolCallSpan bounds a path-like arg with no separators the same way it bounds a bash command', () => {
    // basename() only strips separators — a string with none at all (or whose
    // final segment is huge) passes through completely unchanged. Unlike the
    // bash branch (explicitly capped at MAX_NAME_SEGMENT_CHARS via truncateWithEllipsis
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

  it('describeToolCallSpan truncates an over-long basename without splitting a surrogate pair at the boundary', () => {
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

  it('describeToolCallSpan falls back to the bare tool name when a non-empty path-like arg has NO basename component at all (win32.basename reduces a root/drive-only reference to "")', () => {
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

  it('describeToolCallSpan resolves multiple simultaneous path-like keys by TOOL_PATH_ARGUMENT_KEYS order, not object insertion order, and skips an empty-string candidate', () => {
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

  it('describeToolCallSpan never leaks positional array elements as a path and safely falls back to the bare tool name', () => {
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

  it('describeToolCallSpan reduces a directory-like path arg (trailing slash) to the directory name, not the full path or an empty string', () => {
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

  it("describeToolCallSpan passes unusual characters in toolName itself through unchanged (toolName is Pi's own identifier, not user-controlled args)", () => {
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

  it('describeToolCallSpan passes non-ASCII BMP unicode in a bash command well under the 60-char limit through unmangled', () => {
    // Chinese, Cyrillic, and accented Latin characters are all single UTF-16
    // code units (unlike the emoji surrogate-pair case covered above), and the
    // whole collapsed command is far below MAX_NAME_SEGMENT_CHARS, so no truncation
    // should occur at all — the assertion is that whitespace-collapse does not
    // corrupt or drop any multi-byte chars.
    const command = 'echo   café   北京   Москва  ';
    const collapsed = 'echo café 北京 Москва';
    assert.ok(collapsed.length < 60, 'sanity check: well under the truncation limit');
    const name = describeToolCallSpan('bash', { command });
    assert.equal(name, `bash: ${collapsed}`);
    assert.ok(!name.endsWith('…'), 'must not be truncated when comfortably under the limit');
  });

  it('describeToolCallSpan treats a whitespace-only bash command as absent — falls back to a path arg or the bare tool name, never emitting a dangling "bash: "', () => {
    // A truthy-but-blank command ('   ') survives the `typeof === "string" &&
    // a.command` truthiness check, but collapses to '' after the whitespace
    // normalization — the inner `if (cmd)` guard must then fall through rather
    // than returning 'bash: ' with nothing after the colon.
    assert.equal(describeToolCallSpan('bash', { command: '   ' }), 'bash');
    assert.equal(describeToolCallSpan('bash', { command: ' \t\n ' }), 'bash');
    // With a path-like arg riding along, the fall-through continues into the
    // path branch instead of stopping at the bare name.
    assert.equal(
      describeToolCallSpan('bash', { command: '   ', path: '/a/b/c.txt' }),
      'bash: c.txt',
    );
  });
});

describe('spans and config boundary coverage', () => {
  /**
   * Direct unit coverage of spans.ts's LLM/tool/dangling span helpers
   * (usage-token mapping incl. zero values, error/aborted status, updateName,
   * captureContent gating), circular-ref handling through openToolSpan —
   * surfaces the rest of the suite leaves thin — and the canonical
   * sliceSurrogateSafe unit tests (boundary edges, the maxLen<=0 and
   * negative-maxLen guards), which live in this file since sliceSurrogateSafe
   * itself now lives in spans.ts.
   *
   * Two subjects from the original packages/pi version of this file are
   * deliberately NOT carried over: baseUrl whitespace normalization (config.ts
   * no longer has a baseUrl field — see packages/traceroot/src/pi.ts)
   * and provider forceFlush idempotency (this in-tree integration never builds
   * or owns its own TracerProvider — see instrumentation.ts's
   * createReresolvingTracer — so there is no createTracing()/forceFlush() of
   * its own left to test).
   */

  // Direct-span rig, mirroring close-root-span-backward-scan.test.ts: a real
  // NodeTracerProvider + SimpleSpanProcessor is the most direct way to unit-test
  // the spans.ts helpers in isolation from the instrumentation event pipeline.
  function makeTracer() {
    const spans: ReadableSpan[] = [];
    const provider = new NodeTracerProvider();
    provider.addSpanProcessor(
      new SimpleSpanProcessor({
        export(batch, cb) {
          spans.push(...batch);
          cb({ code: 0 });
        },
        async shutdown() {},
      }),
    );
    const tracer = provider.getTracer('pi-spans-unit');
    return { tracer, spans };
  }

  function attrs(span: ReadableSpan): Record<string, unknown> {
    return span.attributes as Record<string, unknown>;
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
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
      },
      stopReason: 'stop',
      timestamp: 0,
      ...overrides,
    } as AssistantMessage;
  }

  // --- spans.ts: LLM span attribute mapping -------------------------------

  it('closeLlmSpan emits ZERO-valued usage tokens as attributes (setAttr must not treat 0 as absent)', () => {
    const { tracer, spans } = makeTracer();
    const message = assistantMessage({
      usage: {
        input: 0,
        output: 0,
        cacheRead: 0,
        cacheWrite: 0,
        totalTokens: 0,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
      },
    });
    const span = openLlmSpan(tracer, ROOT_CONTEXT, message);
    closeLlmSpan(span, message, true);

    const a = attrs(spans[0]!);
    // A genuine "0 output tokens" turn is real trace data — a `if (!value) skip`
    // guard would drop it. setAttr only skips null/undefined, so 0 must land.
    assert.equal(a['gen_ai.usage.input_tokens'], 0);
    assert.equal(a['gen_ai.usage.output_tokens'], 0);
    assert.equal(a['gen_ai.usage.cache_read_input_tokens'], 0);
    assert.equal(a['gen_ai.usage.cache_creation_input_tokens'], 0);
  });

  it('closeLlmSpan maps cache tokens to the correct gen_ai keys (read<->cacheRead, creation<->cacheWrite)', () => {
    const { tracer, spans } = makeTracer();
    const message = assistantMessage({
      usage: {
        input: 5,
        output: 7,
        cacheRead: 11,
        cacheWrite: 13,
        totalTokens: 36,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
      },
    });
    const span = openLlmSpan(tracer, ROOT_CONTEXT, message);
    closeLlmSpan(span, message, false);

    const a = attrs(spans[0]!);
    assert.equal(a['gen_ai.usage.input_tokens'], 5);
    assert.equal(a['gen_ai.usage.output_tokens'], 7);
    assert.equal(a['gen_ai.usage.cache_read_input_tokens'], 11);
    assert.equal(a['gen_ai.usage.cache_creation_input_tokens'], 13);
  });

  it('openLlmSpan names the span by request model and falls back to "pi.llm" when model is empty', () => {
    const { tracer, spans } = makeTracer();
    const named = openLlmSpan(tracer, ROOT_CONTEXT, assistantMessage({ model: 'gpt-5' }));
    closeLlmSpan(named, assistantMessage({ model: 'gpt-5', responseModel: undefined }), false);
    assert.equal(spans[0]!.name, 'gpt-5');
    assert.equal(attrs(spans[0]!)['gen_ai.request.model'], 'gpt-5');
    assert.equal(attrs(spans[0]!)['gen_ai.system'], 'anthropic');
    assert.equal(attrs(spans[0]!)['openinference.span.kind'], 'LLM');

    const { tracer: t2, spans: s2 } = makeTracer();
    const blank = openLlmSpan(t2, ROOT_CONTEXT, assistantMessage({ model: '' }));
    // updateName in close would also fall back; force both model+responseModel empty.
    closeLlmSpan(blank, assistantMessage({ model: '', responseModel: '' }), false);
    assert.equal(s2[0]!.name, 'pi.llm');
  });

  it('closeLlmSpan updates the span name to responseModel when it differs from the request model', () => {
    const { tracer, spans } = makeTracer();
    const message = assistantMessage({ model: 'claude-req', responseModel: 'claude-resp-dated' });
    const span = openLlmSpan(tracer, ROOT_CONTEXT, message);
    closeLlmSpan(span, message, false);
    assert.equal(spans[0]!.name, 'claude-resp-dated');
    assert.equal(attrs(spans[0]!)['gen_ai.response.model'], 'claude-resp-dated');
  });

  it('closeLlmSpan sets ERROR status with errorMessage for stopReason "error", falling back to the reason when no message', () => {
    const { tracer, spans } = makeTracer();
    const withMsg = openLlmSpan(tracer, ROOT_CONTEXT, assistantMessage());
    closeLlmSpan(
      withMsg,
      assistantMessage({ stopReason: 'error', errorMessage: 'rate limited' }),
      false,
    );
    assert.equal(spans[0]!.status.code, SpanStatusCode.ERROR);
    assert.equal(spans[0]!.status.message, 'rate limited');

    const { tracer: t2, spans: s2 } = makeTracer();
    const noMsg = openLlmSpan(t2, ROOT_CONTEXT, assistantMessage());
    closeLlmSpan(noMsg, assistantMessage({ stopReason: 'error', errorMessage: undefined }), false);
    assert.equal(s2[0]!.status.code, SpanStatusCode.ERROR);
    assert.equal(s2[0]!.status.message, 'error', 'must fall back to the stopReason string');
  });

  it('closeLlmSpan sets ERROR status for stopReason "aborted"', () => {
    const { tracer, spans } = makeTracer();
    const span = openLlmSpan(tracer, ROOT_CONTEXT, assistantMessage());
    closeLlmSpan(span, assistantMessage({ stopReason: 'aborted' }), false);
    assert.equal(spans[0]!.status.code, SpanStatusCode.ERROR);
    assert.equal(spans[0]!.status.message, 'aborted');
  });

  it('closeLlmSpan leaves status UNSET for a normal stop', () => {
    const { tracer, spans } = makeTracer();
    const span = openLlmSpan(tracer, ROOT_CONTEXT, assistantMessage());
    closeLlmSpan(span, assistantMessage({ stopReason: 'stop' }), false);
    assert.equal(spans[0]!.status.code, SpanStatusCode.UNSET);
  });

  it('closeLlmSpan joins multiple assistant text parts and skips non-text parts when capturing content', () => {
    const { tracer, spans } = makeTracer();
    const message = assistantMessage({
      content: [
        { type: 'text', text: 'Hello ' },
        { type: 'tool_use', id: 'x', name: 'bash', input: {} },
        { type: 'text', text: 'world' },
      ],
    });
    const span = openLlmSpan(tracer, ROOT_CONTEXT, message);
    closeLlmSpan(span, message, true);
    assert.equal(attrs(spans[0]!)['output.value'], 'Hello world');
  });

  it('closeLlmSpan omits output.value entirely when captureContent is false', () => {
    const { tracer, spans } = makeTracer();
    const message = assistantMessage({ content: [{ type: 'text', text: 'secret reply' }] });
    const span = openLlmSpan(tracer, ROOT_CONTEXT, message);
    closeLlmSpan(span, message, false);
    assert.equal(attrs(spans[0]!)['output.value'], undefined);
  });

  it('closeLlmSpan omits output.value when the assistant message has no text parts (tool-only turn)', () => {
    const { tracer, spans } = makeTracer();
    const message = assistantMessage({
      content: [{ type: 'tool_use', id: 'x', name: 'bash', input: {} }],
    });
    const span = openLlmSpan(tracer, ROOT_CONTEXT, message);
    closeLlmSpan(span, message, true);
    assert.equal(
      attrs(spans[0]!)['output.value'],
      undefined,
      'a tool-only assistant turn has no text — output.value must be absent, not empty string',
    );
  });

  // --- spans.ts: root span ------------------------------------------------

  // Rephrased from a pre-fix test of the now-removed TR_ATTRIBUTES.WILL_RETRY
  // flag (closeRootSpan used to stamp will_retry unconditionally, coercing
  // undefined to false). Under the new prompt()-anchored root model, the root
  // closes once via finalizeRootSpan — called by instrumentation.ts's
  // proto.prompt when the enclosing prompt() call's own promise settles — and
  // stamps the observable per-window retry ATTEMPT COUNT instead of a
  // per-attempt boolean flag (see spans.ts's own header comment on the split).
  it('finalizeRootSpan stamps retry_count and sets the given status', () => {
    const { tracer, spans } = makeTracer();
    const span = openRootSpan(tracer, ROOT_CONTEXT, {
      text: 'hi',
      sessionId: 's1',
      captureContent: true,
    });
    stampRootOutput(span, [assistantMessage()], true);
    finalizeRootSpan(span, 2, { code: SpanStatusCode.OK });
    assert.equal(attrs(spans[0]!)['traceroot.pi.retry_count'], 2);
    assert.equal(attrs(spans[0]!)['traceroot.pi.will_retry'], undefined, 'the old flag is gone');
    assert.equal(spans[0]!.status.code, SpanStatusCode.OK);
  });

  it('finalizeRootSpan records an exception and sets ERROR status when given an error', () => {
    const { tracer, spans } = makeTracer();
    const span = openRootSpan(tracer, ROOT_CONTEXT, {
      text: 'hi',
      sessionId: 's1',
      captureContent: true,
    });
    finalizeRootSpan(span, 0, { code: SpanStatusCode.ERROR, message: 'boom' }, new Error('boom'));
    assert.equal(spans[0]!.status.code, SpanStatusCode.ERROR);
    assert.equal(spans[0]!.status.message, 'boom');
    assert.ok(
      spans[0]!.events.some((e) => e.name === 'exception'),
      'a recorded exception must appear as a span event',
    );
  });

  // F9: instrumentation.ts's proto.prompt calls finalizeRootSpan from inside a
  // detached `.then(onResolve, onReject)` chain nobody awaits — a throw there
  // becomes an unhandledRejection capable of crashing the host. setAttribute/
  // setStatus/recordException are not otherwise guarded the way endSpanSafe
  // already guards span.end(), so a single misbehaving Span implementation
  // could throw out of finalizeRootSpan before ever reaching endSpanSafe. This
  // drives exactly that: a span whose setStatus() throws must not propagate
  // out of finalizeRootSpan, and the span must still be ended (so it still
  // exports) despite the mid-call failure.
  it('finalizeRootSpan never throws when the span misbehaves (setStatus throws), and still ends the span', () => {
    const { tracer } = makeTracer();
    const realSpan = tracer.startSpan('AgentSession.prompt');
    let endCalled = false;
    const misbehavingSpan = new Proxy(realSpan, {
      get(target, prop, receiver) {
        if (prop === 'setStatus') {
          return () => {
            throw new Error('injected setStatus failure');
          };
        }
        if (prop === 'end') {
          return (...args: Parameters<typeof realSpan.end>) => {
            endCalled = true;
            return target.end(...args);
          };
        }
        return Reflect.get(target, prop, receiver);
      },
    });

    assert.doesNotThrow(() => {
      finalizeRootSpan(misbehavingSpan, 0, { code: SpanStatusCode.OK });
    });
    assert.equal(
      endCalled,
      true,
      'the span must still be ended (so it still exports) despite setStatus throwing mid-call',
    );
  });

  it('openRootSpan sets session id, does not self-stamp sdk identity, and gates input.value on captureContent', () => {
    const { tracer, spans } = makeTracer();
    const withContent = openRootSpan(tracer, ROOT_CONTEXT, {
      text: 'my prompt',
      sessionId: 'sess-9',
      captureContent: true,
    });
    finalizeRootSpan(withContent, 0, { code: SpanStatusCode.OK });
    const a = attrs(spans[0]!);
    assert.equal(a['openinference.span.kind'], 'AGENT');
    assert.equal(a['session.id'], 'sess-9');
    assert.equal(a['input.value'], 'my prompt');
    // pi no longer self-stamps traceroot.sdk.name; core's TraceRootSpanProcessor
    // owns it uniformly (matching the Claude Agent SDK integration). This unit
    // harness has no such processor, so the attribute is absent here.
    assert.equal(a['traceroot.sdk.name'], undefined);

    const { tracer: t2, spans: s2 } = makeTracer();
    const noContent = openRootSpan(t2, ROOT_CONTEXT, {
      text: 'my prompt',
      sessionId: undefined,
      captureContent: false,
    });
    finalizeRootSpan(noContent, 0, { code: SpanStatusCode.OK });
    assert.equal(attrs(s2[0]!)['input.value'], undefined);
    assert.equal(attrs(s2[0]!)['session.id'], undefined);
  });

  it('stampRootOutput omits output.value when the final history has no assistant message', () => {
    const { tracer, spans } = makeTracer();
    const span = openRootSpan(tracer, ROOT_CONTEXT, {
      text: 'hi',
      sessionId: 's',
      captureContent: true,
    });
    const history: AgentMessage[] = [
      { role: 'user', content: 'q', timestamp: 0 },
      {
        role: 'toolResult',
        toolCallId: 't',
        toolName: 'bash',
        content: [],
        isError: false,
        timestamp: 0,
      },
    ];
    stampRootOutput(span, history, true);
    finalizeRootSpan(span, 0, { code: SpanStatusCode.OK });
    assert.equal(attrs(spans[0]!)['output.value'], undefined);
  });

  // --- spans.ts: dangling + tool spans ------------------------------------

  it('closeDanglingSpan marks force_closed and ends; is a no-op on undefined', () => {
    const { tracer, spans } = makeTracer();
    const span = tracer.startSpan('AgentSession.prompt');
    closeDanglingSpan(span);
    assert.equal(attrs(spans[0]!)['traceroot.pi.force_closed'], true);
    // Must not throw on undefined (the abandoned-span sweep can pass undefined slots).
    assert.doesNotThrow(() => closeDanglingSpan(undefined));
  });

  it('openToolSpan swallows a circular-reference arg without setting input.value or throwing', () => {
    const { tracer, spans } = makeTracer();
    const circular: Record<string, unknown> = { a: 1 };
    circular.self = circular;
    assert.doesNotThrow(() => {
      const span = openToolSpan(tracer, ROOT_CONTEXT, 'call-1', 'bash', circular, true);
      closeToolSpan(span, { ok: true }, false, true);
    });
    const toolSpan = spans.find((s) => attrs(s)['gen_ai.tool.call.id'] === 'call-1');
    assert.ok(toolSpan);
    assert.equal(
      attrs(toolSpan!)['input.value'],
      undefined,
      'a circular arg must be skipped (caught), not partially serialized',
    );
    // The result was serializable and must still be captured.
    assert.equal(attrs(toolSpan!)['output.value'], JSON.stringify({ ok: true }));
  });

  it('closeToolSpan sets ERROR status (no message) when isError is true', () => {
    const { tracer, spans } = makeTracer();
    const span = openToolSpan(tracer, ROOT_CONTEXT, 'call-err', 'bash', { command: 'x' }, false);
    closeToolSpan(span, 'boom', true, false);
    const toolSpan = spans.find((s) => attrs(s)['gen_ai.tool.call.id'] === 'call-err');
    assert.equal(toolSpan!.status.code, SpanStatusCode.ERROR);
  });

  // --- sliceSurrogateSafe: boundary tests ---------------------------------

  it('sliceSurrogateSafe on a string of only lone high surrogates never emits a trailing lone high surrogate', () => {
    const loneHighs = '\uD800\uD800\uD800\uD800\uD800';
    const sliced = sliceSurrogateSafe(loneHighs, 3);
    // The boundary char is a high surrogate, so it must back off to length 2.
    assert.equal(sliced.length, 2);
    const last = sliced.charCodeAt(sliced.length - 1);
    assert.ok(
      last >= 0xd800 && last <= 0xdbff,
      'pre-existing lone highs remain (not this fn to fix)',
    );
  });

  it('sliceSurrogateSafe keeps a complete surrogate pair when the LOW surrogate sits at the boundary', () => {
    // 'ab' + emoji: indices 0,1 = a,b; 2 = high; 3 = low. maxLen 4 lands the cut
    // right after the low surrogate — the pair is complete and must be kept whole.
    const text = 'ab\u{1F600}cd';
    const sliced = sliceSurrogateSafe(text, 4);
    assert.equal(sliced, 'ab\u{1F600}');
    assert.equal([...sliced].length, 3, 'three code points: a, b, emoji');
  });

  it('sliceSurrogateSafe returns text unchanged when a surrogate pair sits entirely under maxLen', () => {
    const text = 'x\u{1F600}y';
    assert.equal(
      sliceSurrogateSafe(text, 10),
      text,
      'a fully-contained pair must never be touched',
    );
  });

  it('sliceSurrogateSafe returns text unchanged when at or under maxLen (<=, not <)', () => {
    assert.equal(sliceSurrogateSafe('hello', 5), 'hello');
    assert.equal(sliceSurrogateSafe('hello', 10), 'hello');
    assert.equal(sliceSurrogateSafe('', 0), '');
  });

  it('sliceSurrogateSafe slices plainly when no surrogate sits at the cut boundary', () => {
    assert.equal(sliceSurrogateSafe('abcdef', 3), 'abc');
  });

  it('sliceSurrogateSafe backs off one code unit when a high surrogate sits at the cut boundary', () => {
    // 3 ASCII chars + an emoji (a surrogate pair): cutting at maxLen=4 would
    // otherwise land the boundary exactly on the emoji's high surrogate.
    const emoji = '\u{1F600}';
    const text = `abc${emoji}tail`;
    const sliced = sliceSurrogateSafe(text, 4);
    assert.equal(sliced, 'abc');
    const lastCode = sliced.charCodeAt(sliced.length - 1);
    assert.ok(lastCode < 0xd800 || lastCode > 0xdbff, 'must not end on an unpaired high surrogate');
  });

  it('sliceSurrogateSafe appends no suffix — callers own their own marker', () => {
    const sliced = sliceSurrogateSafe('abcdefgh', 3);
    assert.equal(sliced, 'abc');
    assert.ok(
      !sliced.includes('…'),
      'sliceSurrogateSafe itself must not add an ellipsis or marker',
    );
  });

  it('sliceSurrogateSafe returns empty string for maxLen === 0 rather than slicing', () => {
    assert.equal(sliceSurrogateSafe('abcdefghij', 0), '');
  });

  it('sliceSurrogateSafe returns empty string for a negative maxLen instead of slicing from the end', () => {
    // Regression guard: text.slice(0, cut) with a negative cut slices from the
    // END of the string (e.g. 'abcdefghij'.slice(0, -3) === 'abcdefg'), which
    // is the opposite of capping. A negative maxLen must be treated as "cap to
    // nothing" and return ''.
    assert.equal(sliceSurrogateSafe('abcdefghij', -3), '');
  });
});

describe('spans truncation', () => {
  /**
   * Lens: src/spans.ts's stringifyToolIo (capFieldReplacer) / capJsonWithMarker
   * boundary logic, exercised through the real openToolSpan/closeToolSpan call
   * path rather than as a standalone unit test of the helpers — a future
   * refactor that stops calling capJsonWithMarker from either function must fail
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
   * full before capJsonWithMarker's post-hoc backstop slices the final string —
   * an accepted O(N) trade on already-materialized data. The tests below for
   * that shape assert the surviving final char-cap/marker invariant rather than
   * the removed mid-walk behavior; see each test's own comment.
   */

  // Mirrors src/spans.ts's MAX_TOOL_IO_JSON_CHARS (not exported — hardcoded
  // here the same way span-name.test.ts hardcodes MAX_NAME_SEGMENT_CHARS as 60).
  const MAX_TOOL_IO_JSON_CHARS = 32 * 1024;

  // Mirrors src/spans.ts's capJsonWithMarker marker text exactly (single U+2026
  // ellipsis, not three ASCII dots).
  const TRUNCATION_MARKER = '…[truncated]';

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
    // hatch (see packages/traceroot/src/pi.ts); the in-tree
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
  // emoji) astride the exact capJsonWithMarker cut boundary: the high surrogate
  // lands at character index MAX_TOOL_IO_JSON_CHARS - 1 (0-indexed), i.e.
  // exactly the last character capJsonWithMarker would otherwise keep.
  function payloadWithSurrogateAtBoundary(): { data: string } {
    const emoji = '\u{1F600}'; // 2 UTF-16 code units (a surrogate pair).
    const prefixLen = JSON_WRAPPER_OVERHEAD - '"}'.length; // chars before the string value starts: `{"data":"`.
    const fillerLength = MAX_TOOL_IO_JSON_CHARS - 1 - prefixLen;
    return { data: 'x'.repeat(fillerLength) + emoji + 'tail' };
  }

  it('a tool arg/result under the cap passes through openToolSpan/closeToolSpan unchanged', async () => {
    const args = { command: 'echo hello', note: 'small arg' };
    const result = { content: [{ type: 'text', text: 'hello' }] };
    const toolSpan = await runToolCall(args, result);

    assert.equal(attrs(toolSpan)['input.value'], JSON.stringify(args));
    assert.equal(attrs(toolSpan)['output.value'], JSON.stringify(result));
    assert.ok(!String(attrs(toolSpan)['input.value']).includes(TRUNCATION_MARKER));
    assert.ok(!String(attrs(toolSpan)['output.value']).includes(TRUNCATION_MARKER));
  });

  it('a tool arg/result exactly at the cap boundary passes through unchanged (<=, not <)', async () => {
    const args = payloadOfExactJsonLength(MAX_TOOL_IO_JSON_CHARS);
    const result = payloadOfExactJsonLength(MAX_TOOL_IO_JSON_CHARS, 'y');
    assert.equal(
      JSON.stringify(args).length,
      MAX_TOOL_IO_JSON_CHARS,
      'sanity check on args length',
    );
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

  it('a tool arg/result over the cap is truncated with the marker appended, total length within bound', async () => {
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

  it('a UTF-16 surrogate pair astride the exact truncation boundary is not split into a lone surrogate', async () => {
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

  it('captureToolIo: false still bypasses capJsonWithMarker entirely (no marker, no attribute)', async () => {
    const args = payloadOfExactJsonLength(MAX_TOOL_IO_JSON_CHARS + 1000);
    const toolSpan = await runToolCall(args, {}, { captureToolIo: false });

    assert.equal(attrs(toolSpan)['input.value'], undefined);
    assert.equal(attrs(toolSpan)['output.value'], undefined);
  });

  // A single huge string field (e.g. a big file read, or long command stdout —
  // the dominant real-world shape of oversized tool I/O) must never be fully
  // materialized by JSON.stringify before capJsonWithMarker's post-hoc cap runs.
  // capJsonWithMarker alone already bounds the *final* attribute either way — a
  // naive JSON.stringify(args) then slice(0, MAX) produces byte-identical
  // output to a properly-capped serialization for this single-field shape, so
  // asserting only on the final attribute's length can't tell a fixed
  // implementation apart from a broken one here. Instead this spies on the
  // global JSON.stringify to observe *how* serialization happened: a bare
  // JSON.stringify(args) call (no replacer) covering the huge value is exactly
  // the bug — it means the full string was embedded in the output before any
  // cap applied. A fixed implementation must instead pass a replacer that caps
  // the oversized string as JSON.stringify visits it, before it's embedded.
  it('a single huge string tool argument is capped during serialization, not only after the fact', async () => {
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
        // embedded in the serialized output before capJsonWithMarker ever runs.
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
  // capJsonWithMarker(...) throws a TypeError on `.length`, silently swallowed
  // by openToolSpan/closeToolSpan's catch — whose comment claims it exists only
  // for "circular refs or BigInt", which this isn't. This spies on the global
  // JSON.stringify the same way the huge-string test above does, but asserts
  // the opposite: a correct implementation must recognize a literal `undefined`
  // itself and never hand it to JSON.stringify in the first place, exactly like
  // packages/traceroot/src/claude-agent-sdk.ts's tryStringify already does.
  it('undefined args/result (parameterless tool call / void-returning tool) never reach a bare JSON.stringify(undefined) call, and produce no input.value/output.value attribute', async () => {
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
        'stringifyToolIo must recognize a literal undefined value itself and short-circuit before ever calling JSON.stringify(undefined, ...) — JSON.stringify(undefined, replacer) returns the value undefined (not a string), and handing that to capJsonWithMarker throws a TypeError that gets silently swallowed under a misleading "circular refs or BigInt" comment',
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
  // before capJsonWithMarker's post-hoc backstop slices the final string (the
  // accepted O(N) trade documented in this file's header). What must still hold
  // is the final char-cap/marker invariant, and that the truncated body is a
  // verbatim prefix of the real serialization (not corrupted or re-derived).
  it('a large array of many small strings still exports a bounded, marked-truncated input.value', async () => {
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

  it('a flat object with a huge number of SHORT-valued keys still exports a bounded input.value (the post-hoc backstop must hold the line)', async () => {
    // capFieldReplacer only caps individually-oversized STRING values — none of
    // this object's values are oversized, and its keys are never inspected at
    // all, so a many-keyed object (a word-count map, a file->stat dictionary)
    // is fully materialized by JSON.stringify. The exported attribute must
    // nevertheless respect the hard MAX + marker bound via capJsonWithMarker's
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
});

describe('config resolution', () => {
  /**
   * Lens: config resolution (packages/traceroot/src/pi.ts).
   *
   * Split out of packages/pi/tests/config-resolution.test.ts: this in-tree
   * integration's PiInstrumentationConfig no longer has apiKey/baseUrl fields
   * (see config.ts's own header — there is no export-pipeline configuration
   * here at all, since this integration always gets its tracer from the
   * globally-registered OTel provider core sets up, never builds its own). So
   * every apiKey/baseUrl/env-var-precedence/empty-apiKey-disables-instrumentation
   * test from the original file is deleted along with those fields — there is
   * nothing left for them to probe. What remains: resolveConfig()'s
   * captureContent/captureToolIo defaults, and instrumentPiCodingAgent()'s
   * config-snapshot immutability (adapted to the surviving fields).
   */

  // Fresh class per rig, not a shared module-level class — instrumentPiCodingAgent
  // patches AgentSession.prototype directly, so reusing one class across tests
  // would stack multiple wrap layers onto the same prototype method.
  function makeFakeSessionClass() {
    return class FakeAgentSession {
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
    };
  }

  function assistantMessage(overrides: Partial<AssistantMessage> = {}): AssistantMessage {
    return {
      role: 'assistant',
      content: [{ type: 'text', text: 'done' }],
      api: 'anthropic-messages',
      provider: 'anthropic',
      model: 'claude-sonnet-5',
      usage: {
        input: 1,
        output: 1,
        cacheRead: 0,
        cacheWrite: 0,
        totalTokens: 2,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
      },
      stopReason: 'stop',
      timestamp: 0,
      ...overrides,
    } as AssistantMessage;
  }

  function attrs(span: ReadableSpan): Record<string, unknown> {
    return span.attributes as Record<string, unknown>;
  }

  function registerCapturingProvider(capture: CapturingExporter): void {
    trace.disable();
    const provider = new NodeTracerProvider();
    provider.addSpanProcessor(new SimpleSpanProcessor(capture));
    provider.register();
  }

  it('resolveConfig() defaults captureContent and captureToolIo to true when omitted', () => {
    const resolved = resolveConfig();
    assert.equal(resolved.captureContent, true);
    assert.equal(resolved.captureToolIo, true);

    const resolvedFromEmptyObject = resolveConfig({});
    assert.equal(resolvedFromEmptyObject.captureContent, true);
    assert.equal(resolvedFromEmptyObject.captureToolIo, true);
  });

  it('resolveConfig() respects explicit false overrides for captureContent and captureToolIo independently', () => {
    const contentOff = resolveConfig({ captureContent: false });
    assert.equal(contentOff.captureContent, false);
    assert.equal(contentOff.captureToolIo, true, 'captureToolIo must keep its own default');

    const toolIoOff = resolveConfig({ captureToolIo: false });
    assert.equal(toolIoOff.captureContent, true, 'captureContent must keep its own default');
    assert.equal(toolIoOff.captureToolIo, false);

    const bothOff = resolveConfig({ captureContent: false, captureToolIo: false });
    assert.equal(bothOff.captureContent, false);
    assert.equal(bothOff.captureToolIo, false);
  });

  it('instrumentPiCodingAgent() snapshots the config object at call time — mutating captureContent on the caller-owned object after the call returns has no effect on already-instrumented behavior', async () => {
    const capture = new CapturingExporter();
    const Session = makeFakeSessionClass();
    const sdk = { AgentSession: Session };

    // A plain mutable object, exactly as a caller might build and later reuse
    // or mutate it (e.g. a shared config object edited elsewhere in the host
    // app).
    const config = { captureContent: true };

    registerCapturingProvider(capture);
    instrumentPiCodingAgent(sdk, config);

    // Mutate the field AFTER instrumentPiCodingAgent() has already returned.
    // resolveConfig() must have copied the primitive value out (not retained a
    // live reference to `config`).
    config.captureContent = false;

    const session = new Session();
    const done = session.prompt('sensitive prompt text');
    session.emit({ type: 'agent_start' });
    session.emit({
      type: 'agent_end',
      messages: [assistantMessage({ content: [{ type: 'text', text: 'sensitive reply' }] })],
      willRetry: false,
    });
    await done;

    assert.equal(capture.spans.length, 1);
    const [rootSpan] = capture.spans;
    assert.equal(
      attrs(rootSpan!)['input.value'],
      'sensitive prompt text',
      'captureContent must still resolve to its call-time value (true), not the post-call ' +
        'mutation to false — resolveConfig() must copy primitives by value, not hold a live ' +
        'reference to the caller-owned config object',
    );
    assert.equal(attrs(rootSpan!)['output.value'], 'sensitive reply');
  });
});
