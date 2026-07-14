/**
 * Lens: a host listener that disposes the session synchronously while
 * handling agent_end races pi's own agent_end handler.
 *
 * If a host registers its own session.subscribe() listener BEFORE pi does
 * (i.e. before its first prompt()) and that listener calls session.dispose()
 * synchronously on agent_end, dispose()'s sweep force-closes this run's root
 * span before pi's own agent_end handler runs for the SAME event. Because
 * dispose() reassigns the listener array rather than mutating it mid-dispatch,
 * pi's handler still fires afterward — but finds state.rootSpan already gone.
 * Pi must DETECT that its root span was force-closed out from under it and
 * surface it, rather than silently skipping the real close and quietly
 * producing an incomplete trace.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { assistantMessage, attrs, CapturingExporter, makeFakeSessionClass } from './test-helpers';
import { instrumentPiCodingAgent } from '../src/instrumentation';
import type { AgentEvent } from '../src/types';

test('agent_end after a reentrant dispose() force-closed the root span is surfaced, not silently skipped', async () => {
  const capture = new CapturingExporter();
  const Session = makeFakeSessionClass();
  const sdk = { AgentSession: Session };
  instrumentPiCodingAgent(sdk, { apiKey: 'k', _spanExporter: capture });
  const session = new Session();

  // Host listener registered BEFORE pi's own subscribe() (which happens on the
  // first prompt() below), so it sits ahead of pi's handler in the dispatch
  // order. It disposes the session the instant it sees agent_end — the exact
  // reentrant race. dispose() reassigns the listener array, so pi's own
  // agent_end handler still runs in this same emit loop, just after dispose()
  // already tore the run's root span down.
  session.subscribe((event: AgentEvent) => {
    if (event.type === 'agent_end') session.dispose();
  });

  const warnings: string[] = [];
  const originalWarn = console.warn;
  console.warn = (...args: unknown[]): void => {
    warnings.push(args.map((a) => String(a)).join(' '));
  };
  try {
    await session.prompt('do the work'); // pi subscribes here, after the host
    session.emit({ type: 'agent_start' });
    session.emit({
      type: 'agent_end',
      messages: [assistantMessage({ content: [{ type: 'text', text: 'the final answer' }] })],
      willRetry: false,
    });
  } finally {
    console.warn = originalWarn;
  }

  // dispose()'s sweep force-closed the root span exactly once, so it still
  // exports — but as a FORCE_CLOSED span lacking the normal agent_end output.
  assert.equal(capture.spans.length, 1, 'the root span is force-closed exactly once by dispose()');
  const rootSpan = capture.spans[0];
  assert.equal(attrs(rootSpan)['openinference.span.kind'], 'AGENT');
  assert.equal(
    attrs(rootSpan)['traceroot.pi.force_closed'],
    true,
    'dispose() force-closed the root span, so it is marked force_closed',
  );
  assert.equal(
    attrs(rootSpan)['output.value'],
    undefined,
    "the run's completion output never made it onto the force-closed root span",
  );

  // Because dispose() beat pi's own close, the completion output could not be
  // recorded. Pi must SURFACE that the root span was force-closed by a
  // reentrant dispose() rather than silently doing nothing.
  assert.ok(
    warnings.some((w) => /reentrant dispose/i.test(w) && /agent_end/i.test(w)),
    'pi must warn that agent_end arrived after a reentrant dispose() force-closed the root span',
  );
});
