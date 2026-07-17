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
import { trace } from '@opentelemetry/api';
import { SimpleSpanProcessor } from '@opentelemetry/sdk-trace-base';
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import { instrumentPiCodingAgent } from '../src/pi/instrumentation';
import {
  assistantMessage,
  attrs,
  CapturingExporter,
  makeFakeSessionClass,
} from './pi-test-helpers';

// Registers a real, freshly-registered global TracerProvider wired to
// `capture`, replacing the deleted private-exporter (`_spanExporter`)
// injection path — see pi-test-helpers.ts's makeRig() for the full
// isolation rationale behind calling trace.disable() first.
function registerCapturingProvider(capture: CapturingExporter): void {
  trace.disable();
  const provider = new NodeTracerProvider();
  provider.addSpanProcessor(new SimpleSpanProcessor(capture));
  provider.register();
}

// Extends the shared FakeAgentSession with the standalone steer()/followUp()
// entry points the base fixture deliberately omits (see test-helpers.ts, and
// provider-shared-mode-behavior.test.ts's own makeSteerableSessionClass, which
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

test('calling steer() as the FIRST interaction (no prior prompt() call) still attaches tracing — its run is ROOTLESS (bypasses prompt()), but its child spans still export', async () => {
  // FLIPPED for the new model: only the wrapped prompt() call opens a root
  // span (see instrumentation.ts's module header on the rootless-bypass
  // boundary policy) — steer()/followUp() never did, and agent_start no
  // longer synthesizes one either. Before this change, agent_start
  // unconditionally opened a root regardless of what triggered it, so this
  // test could assert on an exported AGENT span. Under the new model, a run
  // whose ONLY interaction was steer() (no enclosing prompt() call) produces
  // NO root span at all — but the underlying bug this test guards against
  // ("steer() as first interaction gets zero tracing at all") is still real,
  // so it is rephrased to prove the LISTENER is attached: a tool span fired
  // during this bypass run still exports (as a parentless mini-trace),
  // which could only happen if session.subscribe() had actually been called.
  const capture = new CapturingExporter();
  const Session = makeSteerAndFollowUpSessionClass();
  const sdk = { AgentSession: Session };
  registerCapturingProvider(capture);
  instrumentPiCodingAgent(sdk, {});
  const session = new Session();

  // No session.prompt(...) call anywhere — steer() is the only entry point
  // this host ever uses on this session.
  await session.steer('do X instead');
  session.emit({ type: 'agent_start' });
  session.emit({
    type: 'tool_execution_start',
    toolCallId: 'bypass-tool',
    toolName: 'bash',
    args: {},
  });
  session.emit({
    type: 'tool_execution_end',
    toolCallId: 'bypass-tool',
    toolName: 'bash',
    result: {},
    isError: false,
  });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });

  assert.equal(
    capture.spans.find((s) => attrs(s)['openinference.span.kind'] === 'AGENT'),
    undefined,
    'a run that bypasses prompt() entirely must never synthesize a root AGENT span',
  );
  const toolSpan = capture.spans.find((s) => attrs(s)['gen_ai.tool.call.id'] === 'bypass-tool');
  assert.ok(
    toolSpan,
    'steer() must still attach the span listener itself — otherwise this bypass run’s tool span ' +
      'would never have been captured at all, proving prompt() was not required first',
  );
  assert.equal(
    toolSpan!.parentSpanId,
    undefined,
    'with no root open, the bypass run’s tool span parents under ROOT_CONTEXT (a fresh, ' +
      'standalone parentless mini-trace)',
  );
});

test('calling followUp() as the FIRST interaction (no prior prompt() call) still attaches tracing — its run is ROOTLESS (bypasses prompt()), but its child spans still export', async () => {
  const capture = new CapturingExporter();
  const Session = makeSteerAndFollowUpSessionClass();
  const sdk = { AgentSession: Session };
  registerCapturingProvider(capture);
  instrumentPiCodingAgent(sdk, {});
  const session = new Session();

  await session.followUp('also check Y');
  session.emit({ type: 'agent_start' });
  session.emit({
    type: 'tool_execution_start',
    toolCallId: 'bypass-tool',
    toolName: 'bash',
    args: {},
  });
  session.emit({
    type: 'tool_execution_end',
    toolCallId: 'bypass-tool',
    toolName: 'bash',
    result: {},
    isError: false,
  });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });

  assert.equal(
    capture.spans.find((s) => attrs(s)['openinference.span.kind'] === 'AGENT'),
    undefined,
    'a run that bypasses prompt() entirely must never synthesize a root AGENT span',
  );
  const toolSpan = capture.spans.find((s) => attrs(s)['gen_ai.tool.call.id'] === 'bypass-tool');
  assert.ok(
    toolSpan,
    'followUp() must still attach the span listener itself, not rely on prompt() having been ' +
      'called first',
  );
});

test('a session already subscribed via prompt() does not get double-subscribed when steer()/followUp() are called later', async () => {
  const capture = new CapturingExporter();
  const Session = makeSteerAndFollowUpSessionClass();
  const sdk = { AgentSession: Session };
  registerCapturingProvider(capture);
  instrumentPiCodingAgent(sdk, {});
  const session = new Session();

  // prompt() is not awaited immediately (its promise settles only once its
  // final agent_end fires — see pi-test-helpers.ts's module header); steer()
  // and followUp() are independent, immediately-resolving calls that must
  // reuse the SAME listener prompt() already attached, not subscribe again.
  const done = session.prompt('start the run');
  await session.steer('steer mid-run');
  await session.followUp('follow up after');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
  await done;

  assert.equal(
    capture.spans.length,
    1,
    'exactly one root span — proves steer()/followUp() reused the same listener prompt() already attached, instead of subscribing a second time',
  );
});
