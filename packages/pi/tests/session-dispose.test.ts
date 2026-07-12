/**
 * Verifies AgentSession.dispose()'s real, confirmed contract (see
 * instrumentation.ts's module header and types.ts's AgentSessionInstance
 * doc comment for the full verified call chain, read directly out of the
 * real, installed @earendil-works/pi-coding-agent@0.80.6 dist/core/
 * agent-session.js): dispose() clears every listener registered via
 * subscribe() — including instrumentPiCodingAgent()'s own — by reassigning
 * the session's internal listener array to a fresh empty one, with no
 * per-listener unsubscribe() call required from either side.
 *
 * FakeAgentSession here intentionally mirrors that exact mechanism (push on
 * subscribe, splice via the returned closure, reassign to [] on dispose) so
 * this test exercises the same shape the real SDK does, not a simplified
 * stand-in that would pass for the wrong reason.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { Span } from '@opentelemetry/sdk-trace-base';
import type { ReadableSpan } from '@opentelemetry/sdk-trace-base';
import { instrumentPiCodingAgent } from '../src/instrumentation';
import { assistantMessage, CapturingExporter, makeFakeSessionClass } from './test-helpers';

test('session.dispose() does not throw and requires no extra cleanup call from instrumentPiCodingAgent()', async () => {
  const capture = new CapturingExporter();
  const Session = makeFakeSessionClass();
  const sdk = { AgentSession: Session };
  instrumentPiCodingAgent(sdk, { apiKey: 'k', _spanExporter: capture });
  const session = new Session();

  // Drive one full, cleanly-closed turn so instrumentPiCodingAgent()'s
  // subscribe() listener is actually registered and has produced a span,
  // matching real usage instead of disposing an untouched session.
  await session.prompt('do something');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
  assert.equal(capture.spans.length, 1, 'the normal run must still produce its span');

  assert.doesNotThrow(() => {
    session.dispose();
  }, 'dispose() must be safe to call even though instrumentPiCodingAgent() never captured or called the subscribe() unsubscribe function itself');
  assert.equal(session.disposed, true);

  // instrumentPiCodingAgent() never stored or invoked the unsubscribe
  // function subscribe() returned — it relies entirely on dispose() clearing
  // the SDK's own listener array. Firing more events post-dispose (as a
  // buggy or unusual host might) must produce no further spans, proving
  // instrumentation.ts needs no dispose-time hook of its own: the host
  // session's own dispose() is sufficient to stop delivery on its own.
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
  assert.equal(
    capture.spans.length,
    1,
    'no new spans may appear after dispose() — the listener must no longer be reachable',
  );
});

test('dispose() on a session that never had prompt() called (no traceroot-pi subscription registered yet) is still safe', async () => {
  const capture = new CapturingExporter();
  const Session = makeFakeSessionClass();
  const sdk = { AgentSession: Session };
  instrumentPiCodingAgent(sdk, { apiKey: 'k', _spanExporter: capture });
  const session = new Session();

  assert.doesNotThrow(() => {
    session.dispose();
  });
  assert.equal(session.disposed, true);
  assert.equal(capture.spans.length, 0);
});

function attrs(span: ReadableSpan): Record<string, unknown> {
  return span.attributes as Record<string, unknown>;
}

test('dispose() mid-run (before agent_end) force-closes and exports any still-open AGENT/LLM/TOOL spans instead of leaking them', async () => {
  const capture = new CapturingExporter();
  const Session = makeFakeSessionClass();
  const sdk = { AgentSession: Session };
  instrumentPiCodingAgent(sdk, { apiKey: 'k', _spanExporter: capture });
  const session = new Session();

  // Open a run and leave it mid-flight: agent_start opens the AGENT
  // (root) span, message_start opens an LLM span, tool_execution_start
  // opens a TOOL span — and crucially agent_end never fires, exactly the
  // "host disposes while a run is in progress" scenario.
  await session.prompt('long-running task');
  session.emit({ type: 'agent_start' });
  session.emit({
    type: 'message_start',
    message: assistantMessage({ model: 'mid-run-model' }),
  });
  session.emit({
    type: 'tool_execution_start',
    toolCallId: 'call-1',
    toolName: 'read_file',
    args: { path: '/tmp/x' },
  });

  // Nothing has exported yet — all three spans are still open.
  assert.equal(capture.spans.length, 0, 'no span should export before dispose() while still open');

  assert.doesNotThrow(() => {
    session.dispose();
  }, 'dispose() must never throw even though it now force-closes in-flight spans');
  assert.equal(
    session.disposed,
    true,
    'the real dispose() must still run and mark the session disposed',
  );

  assert.equal(
    capture.spans.length,
    3,
    'the open AGENT root span, LLM span, and TOOL span must all be force-closed and exported by dispose()',
  );

  const rootSpan = capture.spans.find((s) => s.name === 'AgentSession.prompt');
  const llmSpan = capture.spans.find((s) => attrs(s)['gen_ai.request.model'] === 'mid-run-model');
  const toolSpan = capture.spans.find((s) => attrs(s)['gen_ai.tool.call.id'] === 'call-1');

  assert.ok(rootSpan, 'the AGENT root span must be exported');
  assert.ok(llmSpan, 'the LLM span must be exported');
  assert.ok(toolSpan, 'the TOOL span must be exported');

  assert.equal(
    attrs(rootSpan!)['traceroot.pi.force_closed'],
    true,
    'the root span must be marked force_closed, distinguishing it from a normal agent_end close',
  );
  assert.equal(
    attrs(llmSpan!)['traceroot.pi.force_closed'],
    true,
    'the LLM span must be marked force_closed',
  );
  assert.equal(
    attrs(toolSpan!)['traceroot.pi.force_closed'],
    true,
    'the TOOL span must be marked force_closed',
  );

  // Firing more events post-dispose must produce no further spans — the
  // real dispose() already cleared _eventListeners, and our own patch must
  // not have reintroduced a way to reach the (now torn-down) state.
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
  assert.equal(
    capture.spans.length,
    3,
    'no further spans may appear after dispose() already force-closed everything',
  );
});

test('a session reused after dispose() re-subscribes and resumes tracing on its next prompt()', async () => {
  // Nothing in the real SDK prevents calling prompt()/steer()/followUp()
  // again on a session instance after dispose() — dispose() only clears the
  // SDK's own _eventListeners array (see instrumentation.ts's module
  // header), it does not make the session instance itself unusable. Before
  // the fix, instrumentPiCodingAgent()'s own `subscribedSessions` WeakSet
  // permanently remembered this session as "already subscribed" and never
  // re-attached its span listener on the next prompt() call, so every run
  // after the first dispose() silently produced zero spans.
  const capture = new CapturingExporter();
  const Session = makeFakeSessionClass();
  const sdk = { AgentSession: Session };
  instrumentPiCodingAgent(sdk, { apiKey: 'k', _spanExporter: capture });
  const session = new Session();

  // First run: clean prompt/agent_start/agent_end cycle.
  await session.prompt('first run');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
  assert.equal(capture.spans.length, 1, 'the first run must export its span');

  session.dispose();
  assert.equal(session.disposed, true);

  // Second run on the SAME session instance, after dispose(). A real host
  // reusing a session (or a test harness that calls dispose() defensively
  // between runs) must still get tracing on this next run.
  await session.prompt('second run, after dispose()');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });

  assert.equal(
    capture.spans.length,
    2,
    'the second run (after dispose() and reuse) must ALSO export its span, not be silently ' +
      'dropped because instrumentPiCodingAgent() thinks this session is still subscribed',
  );
});

test('dispose() force-closes the OTHER open spans even when one span throws while being force-closed', async () => {
  // spans.ts's closeDanglingSpan() calls setAttr() (span.setAttribute(),
  // NOT wrapped in try/catch, unlike endSpanSafe()) before span.end(). If
  // one open span's setAttribute() throws — a misbehaving Span
  // implementation, or a bug triggered by that span's own attribute values —
  // that must not abort the rest of dispose()'s sweep: the other still-open
  // spans must still be force-closed and exported, not silently discarded
  // just because a span with no .end() call is never exported.
  const capture = new CapturingExporter();
  const Session = makeFakeSessionClass();
  const sdk = { AgentSession: Session };
  instrumentPiCodingAgent(sdk, { apiKey: 'k', _spanExporter: capture });
  const session = new Session();

  await session.prompt('multi-tool run');
  session.emit({ type: 'agent_start' });
  session.emit({
    type: 'message_start',
    message: assistantMessage({ model: 'sweep-test-model' }),
  });
  // Three open tool spans left dangling (call-1, call-2, call-3), plus the
  // still-open LLM span and root span — 5 spans total, none of which have
  // had agent_end/message_end/tool_execution_end fire for them.
  session.emit({
    type: 'tool_execution_start',
    toolCallId: 'call-1',
    toolName: 'read_file',
    args: { path: '/tmp/a' },
  });
  session.emit({
    type: 'tool_execution_start',
    toolCallId: 'call-2',
    toolName: 'read_file',
    args: { path: '/tmp/b' },
  });
  session.emit({
    type: 'tool_execution_start',
    toolCallId: 'call-3',
    toolName: 'read_file',
    args: { path: '/tmp/c' },
  });
  assert.equal(capture.spans.length, 0, 'no span should export before dispose() while still open');

  // Poison exactly the call-2 tool span: throw the first time dispose()'s
  // sweep tries to mark it force_closed, simulating a single misbehaving
  // span mid-sweep. Identified by its own gen_ai.tool.call.id attribute
  // (already set at tool_execution_start time), not by call order, so this
  // stays correct regardless of Map iteration order.
  type SetAttributeFn = typeof Span.prototype.setAttribute;
  const originalSetAttribute: SetAttributeFn = Span.prototype.setAttribute;
  Span.prototype.setAttribute = function (
    this: Span,
    key: string,
    value?: Parameters<SetAttributeFn>[1],
  ): Span {
    if (
      key === 'traceroot.pi.force_closed' &&
      this.attributes['gen_ai.tool.call.id'] === 'call-2'
    ) {
      throw new Error('injected span failure for call-2');
    }
    return originalSetAttribute.call(this, key, value);
  } as SetAttributeFn;

  try {
    assert.doesNotThrow(() => {
      session.dispose();
    }, 'dispose() must not throw even though force-closing one span (call-2) failed internally');
  } finally {
    Span.prototype.setAttribute = originalSetAttribute;
  }
  assert.equal(session.disposed, true);

  // call-2's own force-close threw partway through (setAttribute, before
  // .end() is ever reached for it) — it is legitimately never exported. But
  // that one failure must not have aborted the rest of the sweep: the root
  // span, the LLM span, call-1, and call-3 must all still be force-closed
  // and exported despite it.
  assert.equal(
    capture.spans.length,
    4,
    'the 4 non-poisoned spans (root, LLM, call-1, call-3) must still export even though ' +
      'force-closing call-2 threw partway through the sweep',
  );
  const exportedToolCallIds = capture.spans
    .map((s) => (s.attributes as Record<string, unknown>)['gen_ai.tool.call.id'])
    .filter((id): id is string => typeof id === 'string')
    .sort();
  assert.deepEqual(
    exportedToolCallIds,
    ['call-1', 'call-3'],
    'call-1 and call-3 must still be exported; call-2 (poisoned) is legitimately dropped, ' +
      'but must not have taken the others down with it',
  );
  const rootSpan = capture.spans.find((s) => s.name === 'AgentSession.prompt');
  const llmSpan = capture.spans.find(
    (s) => (s.attributes as Record<string, unknown>)['gen_ai.request.model'] === 'sweep-test-model',
  );
  assert.ok(rootSpan, 'the root span must still be exported despite call-2 throwing mid-sweep');
  assert.ok(llmSpan, 'the LLM span must still be exported despite call-2 throwing mid-sweep');
});
