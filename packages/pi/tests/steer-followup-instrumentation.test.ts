/**
 * instrumentPiCodingAgent() must patch AgentSession.prototype.steer and
 * .followUp, not just .prompt — steer()/followUp() are standalone public
 * SDK entry points (see @earendil-works/pi-coding-agent's
 * dist/core/agent-session.d.ts:359/:367) a host can call directly without
 * ever calling prompt() on the session first. Before this fix, a host whose
 * first interaction with a session was steer()/followUp() got zero tracing:
 * subscribe() was never called (no listener attached), so every subsequent
 * AgentEvent — agent_start through agent_end — silently produced no spans
 * at all, for the entire lifetime of that session.
 *
 * Deliberately NOT asserting that steer()/followUp() text becomes a root
 * span's input.value: verified against the real, installed
 * @earendil-works/pi-agent-core@0.80.6 (dist/agent.js:169-176), Agent.steer()
 * / Agent.followUp() only ever enqueue into an internal queue — neither one
 * ever itself triggers a fresh run (only Agent.prompt()/continue() do, via
 * runPromptMessages()). Queuing their text into the prompt-only pendingInput
 * FIFO (consumed exclusively by agent_start) would misattribute it to
 * whatever LATER, unrelated run's agent_start happens to fire next — a
 * worse bug than the one being fixed here. Attaching the listener is the
 * correct, minimal fix for the actual "zero tracing" defect.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { instrumentPiCodingAgent } from '../src/instrumentation';
import { assistantMessage, attrs, CapturingExporter, makeFakeSessionClass } from './test-helpers';

// Extends the shared FakeAgentSession with the standalone steer()/followUp()
// entry points the base fixture deliberately omits (see test-helpers.ts, and
// shared-mode-instrumentation.test.ts's own makeSteerableSessionClass, which
// adds steer() the same way for the same reason), so a host whose first
// interaction is steer()/followUp() can be exercised here. Fresh per call,
// like makeFakeSessionClass itself, so prototype patches never stack across
// tests.
function makeSteerAndFollowUpSessionClass() {
  const Base = makeFakeSessionClass();
  return class SteerAndFollowUpAgentSession extends Base {
    async steer(_text: string, _images?: unknown[]): Promise<void> {}
    async followUp(_text: string, _images?: unknown[]): Promise<void> {}
  };
}

test('calling steer() as the FIRST interaction (no prior prompt() call) still attaches tracing', async () => {
  const capture = new CapturingExporter();
  const Session = makeSteerAndFollowUpSessionClass();
  const sdk = { AgentSession: Session };
  instrumentPiCodingAgent(sdk, { apiKey: 'k', _spanExporter: capture });
  const session = new Session();

  // No session.prompt(...) call anywhere — steer() is the only entry point
  // this host ever uses on this session.
  await session.steer('do X instead');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });

  assert.equal(
    capture.spans.length,
    1,
    'steer() must attach the span listener itself, not rely on prompt() having been called first',
  );
  assert.equal(attrs(capture.spans[0]!)['openinference.span.kind'], 'AGENT');
});

test('calling followUp() as the FIRST interaction (no prior prompt() call) still attaches tracing', async () => {
  const capture = new CapturingExporter();
  const Session = makeSteerAndFollowUpSessionClass();
  const sdk = { AgentSession: Session };
  instrumentPiCodingAgent(sdk, { apiKey: 'k', _spanExporter: capture });
  const session = new Session();

  await session.followUp('also check Y');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });

  assert.equal(
    capture.spans.length,
    1,
    'followUp() must attach the span listener itself, not rely on prompt() having been called first',
  );
  assert.equal(attrs(capture.spans[0]!)['openinference.span.kind'], 'AGENT');
});

test('a session already subscribed via prompt() does not get double-subscribed when steer()/followUp() are called later', async () => {
  const capture = new CapturingExporter();
  const Session = makeSteerAndFollowUpSessionClass();
  const sdk = { AgentSession: Session };
  instrumentPiCodingAgent(sdk, { apiKey: 'k', _spanExporter: capture });
  const session = new Session();

  await session.prompt('start the run');
  await session.steer('steer mid-run');
  await session.followUp('follow up after');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });

  assert.equal(
    capture.spans.length,
    1,
    'exactly one root span — proves steer()/followUp() reused the same listener prompt() already attached, instead of subscribing a second time',
  );
});
