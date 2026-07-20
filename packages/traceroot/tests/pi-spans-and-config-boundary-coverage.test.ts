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
 * no longer has a baseUrl field — see packages/traceroot/src/pi/config.ts)
 * and provider forceFlush idempotency (this in-tree integration never builds
 * or owns its own TracerProvider — see instrumentation.ts's
 * createReresolvingTracer — so there is no createTracing()/forceFlush() of
 * its own left to test).
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { SpanStatusCode } from '@opentelemetry/api';
import type { ReadableSpan } from '@opentelemetry/sdk-trace-base';
import { SimpleSpanProcessor } from '@opentelemetry/sdk-trace-base';
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import { ROOT_CONTEXT } from '@opentelemetry/api';
import {
  openRootSpan,
  stampRootOutput,
  finalizeRootSpan,
  openLlmSpan,
  closeLlmSpan,
  openToolSpan,
  closeToolSpan,
  closeDanglingSpan,
} from '../src/pi/spans';
import { sliceSurrogateSafe } from '../src/pi/spans';
import type { AgentMessage, AssistantMessage } from '../src/pi/types';

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
  const tracer = provider.getTracer('opus-review');
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

test('closeLlmSpan emits ZERO-valued usage tokens as attributes (setAttr must not treat 0 as absent)', () => {
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

test('closeLlmSpan maps cache tokens to the correct gen_ai keys (read<->cacheRead, creation<->cacheWrite)', () => {
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

test('openLlmSpan names the span by request model and falls back to "pi.llm" when model is empty', () => {
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

test('closeLlmSpan updates the span name to responseModel when it differs from the request model', () => {
  const { tracer, spans } = makeTracer();
  const message = assistantMessage({ model: 'claude-req', responseModel: 'claude-resp-dated' });
  const span = openLlmSpan(tracer, ROOT_CONTEXT, message);
  closeLlmSpan(span, message, false);
  assert.equal(spans[0]!.name, 'claude-resp-dated');
  assert.equal(attrs(spans[0]!)['gen_ai.response.model'], 'claude-resp-dated');
});

test('closeLlmSpan sets ERROR status with errorMessage for stopReason "error", falling back to the reason when no message', () => {
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

test('closeLlmSpan sets ERROR status for stopReason "aborted"', () => {
  const { tracer, spans } = makeTracer();
  const span = openLlmSpan(tracer, ROOT_CONTEXT, assistantMessage());
  closeLlmSpan(span, assistantMessage({ stopReason: 'aborted' }), false);
  assert.equal(spans[0]!.status.code, SpanStatusCode.ERROR);
  assert.equal(spans[0]!.status.message, 'aborted');
});

test('closeLlmSpan leaves status UNSET for a normal stop', () => {
  const { tracer, spans } = makeTracer();
  const span = openLlmSpan(tracer, ROOT_CONTEXT, assistantMessage());
  closeLlmSpan(span, assistantMessage({ stopReason: 'stop' }), false);
  assert.equal(spans[0]!.status.code, SpanStatusCode.UNSET);
});

test('closeLlmSpan joins multiple assistant text parts and skips non-text parts when capturing content', () => {
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

test('closeLlmSpan omits output.value entirely when captureContent is false', () => {
  const { tracer, spans } = makeTracer();
  const message = assistantMessage({ content: [{ type: 'text', text: 'secret reply' }] });
  const span = openLlmSpan(tracer, ROOT_CONTEXT, message);
  closeLlmSpan(span, message, false);
  assert.equal(attrs(spans[0]!)['output.value'], undefined);
});

test('closeLlmSpan omits output.value when the assistant message has no text parts (tool-only turn)', () => {
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
test('finalizeRootSpan stamps retry_count and sets the given status', () => {
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

test('finalizeRootSpan records an exception and sets ERROR status when given an error', () => {
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
test('finalizeRootSpan never throws when the span misbehaves (setStatus throws), and still ends the span', () => {
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

test('openRootSpan sets session id, does not self-stamp sdk identity, and gates input.value on captureContent', () => {
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

test('stampRootOutput omits output.value when the final history has no assistant message', () => {
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

test('closeDanglingSpan marks force_closed and ends; is a no-op on undefined', () => {
  const { tracer, spans } = makeTracer();
  const span = tracer.startSpan('AgentSession.prompt');
  closeDanglingSpan(span);
  assert.equal(attrs(spans[0]!)['traceroot.pi.force_closed'], true);
  // Must not throw on undefined (the abandoned-span sweep can pass undefined slots).
  assert.doesNotThrow(() => closeDanglingSpan(undefined));
});

test('openToolSpan swallows a circular-reference arg without setting input.value or throwing', () => {
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

test('closeToolSpan sets ERROR status (no message) when isError is true', () => {
  const { tracer, spans } = makeTracer();
  const span = openToolSpan(tracer, ROOT_CONTEXT, 'call-err', 'bash', { command: 'x' }, false);
  closeToolSpan(span, 'boom', true, false);
  const toolSpan = spans.find((s) => attrs(s)['gen_ai.tool.call.id'] === 'call-err');
  assert.equal(toolSpan!.status.code, SpanStatusCode.ERROR);
});

// --- sliceSurrogateSafe: boundary tests ---------------------------------

test('sliceSurrogateSafe on a string of only lone high surrogates never emits a trailing lone high surrogate', () => {
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

test('sliceSurrogateSafe keeps a complete surrogate pair when the LOW surrogate sits at the boundary', () => {
  // 'ab' + emoji: indices 0,1 = a,b; 2 = high; 3 = low. maxLen 4 lands the cut
  // right after the low surrogate — the pair is complete and must be kept whole.
  const text = 'ab\u{1F600}cd';
  const sliced = sliceSurrogateSafe(text, 4);
  assert.equal(sliced, 'ab\u{1F600}');
  assert.equal([...sliced].length, 3, 'three code points: a, b, emoji');
});

test('sliceSurrogateSafe returns text unchanged when a surrogate pair sits entirely under maxLen', () => {
  const text = 'x\u{1F600}y';
  assert.equal(sliceSurrogateSafe(text, 10), text, 'a fully-contained pair must never be touched');
});

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

test('sliceSurrogateSafe returns empty string for maxLen === 0 rather than slicing', () => {
  assert.equal(sliceSurrogateSafe('abcdefghij', 0), '');
});

test('sliceSurrogateSafe returns empty string for a negative maxLen instead of slicing from the end', () => {
  // Regression guard: text.slice(0, cut) with a negative cut slices from the
  // END of the string (e.g. 'abcdefghij'.slice(0, -3) === 'abcdefg'), which
  // is the opposite of capping. A negative maxLen must be treated as "cap to
  // nothing" and return ''.
  assert.equal(sliceSurrogateSafe('abcdefghij', -3), '');
});
