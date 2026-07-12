/**
 * A prompt() call that rejects (or throws) BEFORE its corresponding
 * agent_start ever fires must not leave a stale entry in the pendingInput
 * FIFO queue — see instrumentation-edge-cases.test.ts's existing
 * "a rejected prompt() ... never creates a dangling root span" test, which
 * proves the rejected call produces no span but never checks whether its
 * queued text leaks forward into a LATER, unrelated, successful prompt()
 * call on the same session.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { instrumentPiCodingAgent } from '../src/instrumentation';
import { assistantMessage, attrs, CapturingExporter, makeFakeSessionClass } from './test-helpers';

test('a prompt() call that rejects before agent_start does not leak its stale text into a LATER successful prompt() root span', async () => {
  const capture = new CapturingExporter();
  const Session = makeFakeSessionClass((text) => text === 'bad text that fails validation');
  const sdk = { AgentSession: Session };
  instrumentPiCodingAgent(sdk, { apiKey: 'k', _spanExporter: capture });
  const session = new Session();

  // First call: rejects before agent_start ever fires (e.g. a validation
  // failure inside Pi's own prompt() implementation).
  await assert.rejects(() => session.prompt('bad text that fails validation'));
  assert.equal(capture.spans.length, 0, 'the rejected call must not itself produce a span');

  // Second call: a genuine, unrelated, successful prompt on the SAME
  // session instance. This is the realistic case — a host app just retries
  // with different input, or a user submits a new message after a failed one.
  await session.prompt('good text that should actually run');
  session.emit({ type: 'agent_start' });
  session.emit({
    type: 'agent_end',
    messages: [assistantMessage({ content: [{ type: 'text', text: 'reply' }] })],
    willRetry: false,
  });

  const rootSpans = capture.spans.filter((s) => attrs(s)['openinference.span.kind'] === 'AGENT');
  assert.equal(rootSpans.length, 1, 'exactly one real run happened');
  assert.equal(
    attrs(rootSpans[0]!)['input.value'],
    'good text that should actually run',
    'the successful run must be attributed with ITS OWN prompt text, not the stale text left ' +
      'behind by the earlier rejected prompt() call that never reached agent_start',
  );
});
