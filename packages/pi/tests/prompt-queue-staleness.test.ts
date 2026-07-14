/**
 * A prompt() call whose run never starts — it hangs with no resolve/reject and
 * no agent_start ever fires — leaves its text sitting in the per-session
 * pendingInput FIFO queue forever. rejected-prompt-fifo-leak.test.ts already
 * covers the REJECTION path (the .catch() handler removes a rejected call's
 * entry), but a call that simply never settles has no such handler firing, so
 * its entry is never removed. A later, genuinely-new prompt() call on the same
 * session would then dequeue that abandoned entry at its own agent_start,
 * permanently misattributing every subsequent call's input text.
 *
 * The fix stamps each queued entry with an enqueue time and, at dequeue,
 * discards any head entry older than a bounded staleness window rather than
 * blindly trusting FIFO head position.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { assistantMessage, attrs, makeRig } from './test-helpers';

test('a stale queued prompt() (its run never started) is discarded instead of misattributing its text to a LATER run', async () => {
  const { capture, Session } = makeRig();
  const session = new Session();

  const realNow = Date.now;
  try {
    let clock = 1_700_000_000_000;
    Date.now = () => clock;

    // First prompt() queues its text, then its run never starts: no agent_start
    // ever fires for it and its promise resolved (so the rejection-only cleanup
    // never runs). Its entry is stranded in the FIFO queue.
    await session.prompt('STALE prompt whose run never started');

    // Time advances far past any reasonable staleness window before the next,
    // genuinely-new prompt() call on the same session instance.
    clock += 24 * 60 * 60 * 1000; // +24h

    await session.prompt('the genuine later prompt that DID start');
    session.emit({ type: 'agent_start' });
    session.emit({
      type: 'agent_end',
      messages: [assistantMessage({ content: [{ type: 'text', text: 'reply' }] })],
      willRetry: false,
    });
  } finally {
    Date.now = realNow;
  }

  const rootSpans = capture.spans.filter((s) => attrs(s)['openinference.span.kind'] === 'AGENT');
  assert.equal(rootSpans.length, 1, 'exactly one real run happened');
  assert.equal(
    attrs(rootSpans[0]!)['input.value'],
    'the genuine later prompt that DID start',
    'the later run must be attributed with ITS OWN text; the stale queued entry left behind by a ' +
      'prompt() call that never reached agent_start must be discarded, not dequeued as this run input',
  );
});

test('multiple stacked stale entries are all skipped so the first genuinely-fresh queued entry is used', async () => {
  const { capture, Session } = makeRig();
  const session = new Session();

  const realNow = Date.now;
  try {
    let clock = 1_700_000_000_000;
    Date.now = () => clock;

    // Two abandoned prompt() calls stack up in the queue, neither ever starting.
    await session.prompt('first abandoned');
    await session.prompt('second abandoned');

    clock += 24 * 60 * 60 * 1000; // both are now far past the staleness window

    await session.prompt('the only genuine one');
    session.emit({ type: 'agent_start' });
    session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
  } finally {
    Date.now = realNow;
  }

  const rootSpans = capture.spans.filter((s) => attrs(s)['openinference.span.kind'] === 'AGENT');
  assert.equal(rootSpans.length, 1);
  assert.equal(
    attrs(rootSpans[0]!)['input.value'],
    'the only genuine one',
    'the dequeue must skip past every stale head entry to reach the fresh one, not stop at the ' +
      'first (stale) entry',
  );
});

test('a still-fresh queued prompt() (within the staleness window) is NOT discarded', async () => {
  // Control for the tests above: the staleness guard must only drop genuinely
  // abandoned entries, never a normal queued prompt whose agent_start fires a
  // moment later. A too-aggressive window would silently blank out every real
  // run's input.value, so this pins the healthy FIFO path.
  const { capture, Session } = makeRig();
  const session = new Session();

  await session.prompt('a normal prompt');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });

  const rootSpans = capture.spans.filter((s) => attrs(s)['openinference.span.kind'] === 'AGENT');
  assert.equal(rootSpans.length, 1);
  assert.equal(
    attrs(rootSpans[0]!)['input.value'],
    'a normal prompt',
    'a queued prompt whose agent_start arrives promptly must still be used as its run input',
  );
});
