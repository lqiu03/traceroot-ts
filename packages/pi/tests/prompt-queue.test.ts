/**
 * Prompt-queue FIFO correctness tests, merged from four previously-separate
 * files that all exercised the same per-session pendingInput FIFO queue in
 * instrumentation.ts:
 *
 *  - prompt-queue-staleness.test.ts
 *  - rejected-prompt-fifo-leak.test.ts
 *  - steered-prompt-fifo-corruption.test.ts
 *  - agent-end-continuation-input-requeue.test.ts
 *
 * Also folds in two tests from the former
 * confirmed-bugfix-regressions.test.ts grab-bag (its Bugs 3 and
 * 4): the plain multi-slot FIFO ordering baseline, and the plain willRetry
 * reuse baseline. Neither duplicates existing coverage here — every other
 * FIFO/reuse test in this file exercises an adversarial variant (staleness,
 * rejection, steering, a second call already queued behind the retry, etc.),
 * but none of them pin down the simple, unadorned base case with an
 * otherwise-empty queue, which these two do.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { instrumentPiCodingAgent } from '../src/instrumentation';
import type { AgentEvent } from '../src/types';
import {
  assistantMessage,
  attrs,
  CapturingExporter,
  makeFakeSessionClass,
  makeRig,
} from './test-helpers';

// =============================================================================
// confirmed-bugfix-regressions.test.ts's Bug 3
//
// Plain FIFO baseline: two prompt() calls both queue before EITHER
// agent_start arrives, with no staleness/rejection/steering involved. With
// the old single-slot WeakMap design, the second prompt() call would
// silently overwrite the first's pending text; the fix needed a real
// multi-slot FIFO queue. None of the staleness/rejection/steering tests
// below exercise this bare two-call-both-succeed case.
// =============================================================================

test('two overlapping prompt() calls before their respective agent_start events each get their own correct input text, FIFO', async () => {
  const { capture, Session } = makeRig();
  const session = new Session();

  // Both prompt() calls fire before EITHER agent_start arrives — e.g. the
  // host app queues a second message while the first is still being
  // dispatched into the agent loop. With the old single-slot WeakMap, the
  // second prompt() call would silently overwrite the first's pending text.
  await session.prompt('first prompt text');
  await session.prompt('second prompt text');

  session.emit({ type: 'agent_start' });
  session.emit({
    type: 'agent_end',
    messages: [assistantMessage({ content: [{ type: 'text', text: 'first reply' }] })],
    willRetry: false,
  });

  session.emit({ type: 'agent_start' });
  session.emit({
    type: 'agent_end',
    messages: [assistantMessage({ content: [{ type: 'text', text: 'second reply' }] })],
    willRetry: false,
  });

  const rootSpans = capture.spans.filter((s) => attrs(s)['openinference.span.kind'] === 'AGENT');
  assert.equal(rootSpans.length, 2);
  assert.equal(
    attrs(rootSpans[0]!)['input.value'],
    'first prompt text',
    'the FIRST agent_start must claim the FIRST queued prompt() text, not the second',
  );
  assert.equal(attrs(rootSpans[0]!)['output.value'], 'first reply');
  assert.equal(
    attrs(rootSpans[1]!)['input.value'],
    'second prompt text',
    'the SECOND agent_start must claim the SECOND queued prompt() text — with the old single-slot ' +
      'design both roots would incorrectly show the same (second) text',
  );
  assert.equal(attrs(rootSpans[1]!)['output.value'], 'second reply');
});

// =============================================================================
// prompt-queue-staleness.test.ts
//
// A prompt() call whose run never starts — it hangs with no resolve/reject and
// no agent_start ever fires — leaves its text sitting in the per-session
// pendingInput FIFO queue forever. The rejected-prompt-fifo-leak tests below
// already cover the REJECTION path (the .catch() handler removes a rejected
// call's entry), but a call that simply never settles has no such handler
// firing, so its entry is never removed. A later, genuinely-new prompt() call
// on the same session would then dequeue that abandoned entry at its own
// agent_start, permanently misattributing every subsequent call's input text.
//
// The fix stamps each queued entry with an enqueue time and, at dequeue,
// discards any head entry older than a bounded staleness window rather than
// blindly trusting FIFO head position.
// =============================================================================

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

// =============================================================================
// rejected-prompt-fifo-leak.test.ts
//
// A prompt() call that rejects (or throws) BEFORE its corresponding
// agent_start ever fires must not leave a stale entry in the pendingInput
// FIFO queue — see instrumentation-edge-cases.test.ts's existing
// "a rejected prompt() ... never creates a dangling root span" test, which
// proves the rejected call produces no span but never checks whether its
// queued text leaks forward into a LATER, unrelated, successful prompt()
// call on the same session.
// =============================================================================

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

test('a prompt() call that throws SYNCHRONOUSLY (before returning a promise) does not leak its stale text into a LATER successful prompt() root span', async () => {
  // The rejection test above exercises the async path (originalPrompt returns
  // a promise that rejects, cleaned up by result.catch(removeIfStillQueued)).
  // This one exercises the DISTINCT synchronous branch in proto.prompt's
  // wrapper (the try/catch directly around originalPrompt.call): a prompt()
  // implementation that throws before ever constructing a promise. Without
  // that catch, the sync throw would propagate with the queued entry still in
  // the FIFO, and the next successful prompt()'s agent_start would dequeue
  // the dead call's text instead of its own.
  const capture = new CapturingExporter();
  const Session = class FakeAgentSession {
    sessionId = 'sess-1';
    private listeners: Array<(event: AgentEvent) => void> = [];
    // Deliberately NOT async: a sync throw here escapes as a throw, not a
    // rejected promise — the exact branch under test.
    prompt(text: string, _options?: unknown): Promise<void> {
      if (text === 'text that throws synchronously') {
        throw new Error('synchronous validation failure');
      }
      return Promise.resolve();
    }
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
  const sdk = { AgentSession: Session };
  instrumentPiCodingAgent(sdk, { apiKey: 'k', _spanExporter: capture });
  const session = new Session();

  assert.throws(
    () => session.prompt('text that throws synchronously'),
    /synchronous validation failure/,
    'the wrapper must rethrow the original synchronous error, not swallow it',
  );
  assert.equal(capture.spans.length, 0, 'the throwing call must not itself produce a span');

  await session.prompt('good text after the synchronous throw');
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
    'good text after the synchronous throw',
    'the successful run must carry ITS OWN text, not the entry stranded by the synchronous throw',
  );
});

// =============================================================================
// steered-prompt-fifo-corruption.test.ts
//
// A prompt() call that resolves successfully but never reaches agent_start
// must not leave a stale entry in the pendingInput FIFO queue either — not
// just the throw/reject case already covered above.
//
// Verified against the real, installed @earendil-works/pi-coding-agent@0.80.6
// (dist/core/agent-session.js:812-824): when `this.isStreaming` is true and
// `options.streamingBehavior` is set ('steer' or 'followUp'), prompt() calls
// `_queueSteer`/`_queueFollowUp` — which inject the message into the
// CURRENTLY-running Agent loop, not a new run — and returns. It never reaches
// `_runAgentPrompt`, the only call site that leads to a fresh agent_start.
// The promise resolves, not rejects.
//
// makeStreamingAwareFakeSessionClass below faithfully mirrors that gate:
// `isStreaming` is a real, mutable property (flipped by the test the same way
// the real SDK's own isStreaming lifecycle would — set on a fresh run,
// cleared once agent_end fires), and prompt() only "runs" (i.e. becomes the
// kind of call that will eventually produce agent_start) when NOT already
// streaming, or when streaming without a streamingBehavior option.
// =============================================================================

function makeStreamingAwareFakeSessionClass() {
  return class FakeAgentSession {
    sessionId = 'sess-1';
    // Mutable, mirroring the real SDK's `get isStreaming()` — true while a
    // run is in flight (from the moment prompt() starts a fresh run until
    // the test emits agent_end and clears it back to false).
    isStreaming = false;
    private listeners: Array<(event: AgentEvent) => void> = [];

    async prompt(
      _text: string,
      options?: { streamingBehavior?: 'steer' | 'followUp' },
    ): Promise<void> {
      // Faithful mirror of the verified real gate (agent-session.js:812-824):
      // while streaming, a call with streamingBehavior set queues into the
      // ALREADY-running loop and resolves without ever starting a new run —
      // no agent_start will ever follow this particular call.
      if (this.isStreaming && options?.streamingBehavior) {
        return;
      }
      // A genuinely new run: the real SDK flips isStreaming on before the
      // agent loop's own agent_start event fires.
      this.isStreaming = true;
    }

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

test("a steer()/followUp() call queued in during an active run (resolves, no agent_start) does not corrupt the NEXT run's input.value", async () => {
  const capture = new CapturingExporter();
  const Session = makeStreamingAwareFakeSessionClass();
  const sdk = { AgentSession: Session };
  instrumentPiCodingAgent(sdk, { apiKey: 'k', _spanExporter: capture });
  const session = new Session();

  // Run 1 starts normally.
  await session.prompt('run 1 text');
  session.emit({ type: 'agent_start' });

  // While run 1 is still active, a steered-in follow-up resolves
  // successfully with NO new agent_start — the SDK's documented interactive
  // steering/follow-up pattern.
  await session.prompt('steered-in text', { streamingBehavior: 'steer' });

  // Run 1 ends.
  session.emit({
    type: 'agent_end',
    messages: [assistantMessage({ content: [{ type: 'text', text: 'reply 1' }] })],
    willRetry: false,
  });
  session.isStreaming = false;

  // A genuinely new run 2 starts on the same session.
  await session.prompt('run 2 text');
  session.emit({ type: 'agent_start' });
  session.emit({
    type: 'agent_end',
    messages: [assistantMessage({ content: [{ type: 'text', text: 'reply 2' }] })],
    willRetry: false,
  });

  const rootSpans = capture.spans.filter((s) => attrs(s)['openinference.span.kind'] === 'AGENT');
  assert.equal(rootSpans.length, 2, 'run 1 and run 2 each got exactly one root span');
  assert.equal(attrs(rootSpans[0]!)['input.value'], 'run 1 text');
  assert.equal(
    attrs(rootSpans[1]!)['input.value'],
    'run 2 text',
    'run 2 must be attributed with ITS OWN prompt text, not the stale text left behind by the ' +
      'earlier steer()/followUp() queue-in call that resolved without ever reaching agent_start',
  );
});

/**
 * Two more real-SDK early-return paths that resolve successfully without
 * ever reaching _runAgentPrompt (and therefore without ever firing
 * agent_start) — independent of isStreaming/streamingBehavior above.
 *
 * Verified against the real, installed @earendil-works/pi-coding-agent@0.80.6
 * (dist/core/agent-session.js):
 *
 *  - Lines 783-790: `if (expandPromptTemplates && text.startsWith("/")) {
 *    const handled = await this._tryExecuteExtensionCommand(text); if
 *    (handled) { preflightResult?.(true); return; } }` — a leading "/"
 *    matched by a registered extension command executes immediately and
 *    returns, with no LLM turn and no agent_start, regardless of
 *    isStreaming. instrumentation.ts detects this precisely (no
 *    false-positive) via the SDK's own public `session.extensionRunner.
 *    getCommand(name)`.
 *  - Lines 794-799: `if (this._extensionRunner.hasHandlers("input")) { const
 *    inputResult = await this._extensionRunner.emitInput(...); if
 *    (inputResult.action === "handled") { preflightResult?.(true); return; }
 *    }` — an extension's 'input' hook can intercept and fully handle a
 *    prompt before it ever reaches the agent loop, also with no agent_start.
 *    instrumentation.ts can only detect this heuristically (via the SDK's
 *    own public `session.hasExtensionHandlers('input')`, true whenever ANY
 *    'input' hook is registered — not whether it will intercept THIS
 *    specific text, which is not knowable from outside without invoking the
 *    hook itself). See instrumentation.ts's own comment on
 *    mayBeHandledByInputHook for the accepted tradeoff this implies.
 */
function makeCommandAwareFakeSessionClass(registeredCommands: Set<string>) {
  return class FakeAgentSession {
    sessionId = 'sess-1';
    private listeners: Array<(event: AgentEvent) => void> = [];
    readonly extensionRunner = {
      getCommand: (name: string): object | undefined =>
        registeredCommands.has(name) ? {} : undefined,
    };

    async prompt(text: string, _options?: unknown): Promise<void> {
      // Mirrors the real prompt()'s extension-command early return
      // (783-790): a leading "/" whose command name is registered resolves
      // immediately, no run started, no agent_start will ever follow.
      if (text.startsWith('/')) {
        const spaceIndex = text.indexOf(' ');
        const commandName = spaceIndex === -1 ? text.slice(1) : text.slice(1, spaceIndex);
        if (registeredCommands.has(commandName)) {
          return;
        }
      }
      // Otherwise, a genuine run — the caller drives agent_start/agent_end
      // manually via emit(), same as every other fake session in this suite.
    }

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

test('a leading-"/" extension command match (resolves, no agent_start) does not corrupt the NEXT run\'s input.value', async () => {
  const capture = new CapturingExporter();
  const Session = makeCommandAwareFakeSessionClass(new Set(['mycommand']));
  const sdk = { AgentSession: Session };
  instrumentPiCodingAgent(sdk, { apiKey: 'k', _spanExporter: capture });
  const session = new Session();

  // Matches a registered extension command: the real SDK's
  // _tryExecuteExtensionCommand executes it and returns before ever reaching
  // _runAgentPrompt — no agent_start follows this call.
  await session.prompt('/mycommand do something');

  // A genuinely new, unrelated run on the same session.
  await session.prompt('a real prompt after the slash command');
  session.emit({ type: 'agent_start' });
  session.emit({
    type: 'agent_end',
    messages: [assistantMessage({ content: [{ type: 'text', text: 'reply' }] })],
    willRetry: false,
  });

  const rootSpans = capture.spans.filter((s) => attrs(s)['openinference.span.kind'] === 'AGENT');
  assert.equal(rootSpans.length, 1, 'the slash command itself produced no root span');
  assert.equal(
    attrs(rootSpans[0]!)['input.value'],
    'a real prompt after the slash command',
    "the real run must be attributed with ITS OWN prompt text, not the slash command's stale " +
      'queued text left behind by a call that resolved without ever reaching agent_start',
  );
});

test('an unregistered leading-"/" text (no matching extension command) is NOT proactively excluded, and still runs normally', async () => {
  // Guards against an overly-broad implementation that skips the queue for
  // ANY leading "/" text instead of only text matching a REGISTERED command
  // — verified against the real SDK: _tryExecuteExtensionCommand returns
  // false (not handled) when getCommand() finds nothing, and prompt() falls
  // through to the normal path, reaching agent_start like any other call.
  const capture = new CapturingExporter();
  const Session = makeStreamingAwareFakeSessionClass();
  const sdk = { AgentSession: Session };
  instrumentPiCodingAgent(sdk, { apiKey: 'k', _spanExporter: capture });
  const session = new Session();

  await session.prompt('/not-a-registered-command with some args');
  session.emit({ type: 'agent_start' });
  session.emit({
    type: 'agent_end',
    messages: [assistantMessage({ content: [{ type: 'text', text: 'reply' }] })],
    willRetry: false,
  });

  const rootSpans = capture.spans.filter((s) => attrs(s)['openinference.span.kind'] === 'AGENT');
  assert.equal(rootSpans.length, 1);
  assert.equal(
    attrs(rootSpans[0]!)['input.value'],
    '/not-a-registered-command with some args',
    'an unregistered slash-prefixed text must still be queued and attributed normally — the fix ' +
      'must not treat every leading "/" as an extension command',
  );
});

function makeInputHookAwareFakeSessionClass() {
  return class FakeAgentSession {
    sessionId = 'sess-1';
    // Toggled by the test to model a hook that is only registered for part
    // of the session's lifetime (e.g. a scoped moderation hook that
    // unregisters itself after acting once) — mirrors the SDK's own public
    // `hasExtensionHandlers('input')`, read fresh on every prompt() call.
    inputHandlerActive = false;
    private listeners: Array<(event: AgentEvent) => void> = [];

    hasExtensionHandlers(eventType: string): boolean {
      return this.inputHandlerActive && eventType === 'input';
    }

    async prompt(text: string, _options?: unknown): Promise<void> {
      // Mirrors the real prompt()'s input-hook-handled early return
      // (794-799): while the hook is active, this specific text resolves
      // immediately, no run started, no agent_start will ever follow.
      if (this.inputHandlerActive && text === 'a message an extension input hook intercepts') {
        return;
      }
      // Otherwise, a genuine run — the caller drives agent_start/agent_end
      // manually via emit(), same as every other fake session in this suite.
    }

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

test("an extension 'input' hook returning action:'handled' (resolves, no agent_start) does not corrupt a LATER, unrelated run's input.value", async () => {
  const capture = new CapturingExporter();
  const Session = makeInputHookAwareFakeSessionClass();
  const sdk = { AgentSession: Session };
  instrumentPiCodingAgent(sdk, { apiKey: 'k', _spanExporter: capture });
  const session = new Session();

  session.inputHandlerActive = true;
  // Matches an extension's 'input' hook returning action: 'handled': the
  // real SDK's emitInput()/handled branch returns before ever reaching
  // _runAgentPrompt — no agent_start follows this call either. Before the
  // fix, this call's text would still have been pushed onto pendingInput and
  // left stuck there forever (the promise resolves, not rejects).
  await session.prompt('a message an extension input hook intercepts');

  // The hook unregisters itself (a realistic, scoped usage pattern) before
  // the next call — same public signal (hasExtensionHandlers) the real SDK
  // itself would report as false once no 'input' handler remains.
  session.inputHandlerActive = false;

  // A genuinely new, unrelated run on the same session.
  await session.prompt('a real prompt after the intercepted message');
  session.emit({ type: 'agent_start' });
  session.emit({
    type: 'agent_end',
    messages: [assistantMessage({ content: [{ type: 'text', text: 'reply' }] })],
    willRetry: false,
  });

  const rootSpans = capture.spans.filter((s) => attrs(s)['openinference.span.kind'] === 'AGENT');
  assert.equal(rootSpans.length, 1, 'the intercepted message itself produced no root span');
  assert.equal(
    attrs(rootSpans[0]!)['input.value'],
    'a real prompt after the intercepted message',
    "the real run must be attributed with ITS OWN prompt text, not the intercepted message's " +
      'stale queued text left behind by a call that resolved without ever reaching agent_start',
  );
});

// =============================================================================
// agent-end-continuation-input-requeue.test.ts
//
// agent_end's re-queue of pendingInputText must not be gated on
// `event.willRetry` alone.
//
// Verified against the real, installed @earendil-works/pi-coding-agent@0.80.6
// (dist/core/agent-session.js): AgentSession._runAgentPrompt() runs
// `while (await this._handlePostAgentRun()) { await this.agent.continue(); }`,
// and `_handlePostAgentRun()` returns true for THREE independent reasons —
// (1) a retryable error (the only one reflected in `event.willRetry`, which
// `_willRetryAfterAgentEnd` computes purely from `_isRetryableError` on the
// last assistant message — and `_isRetryableError` explicitly EXCLUDES
// context-overflow: "Context overflow is handled by compaction, not retry"),
// (2) `_checkCompaction(msg)` returning true — an ordinary *successful*
// response whose context just crossed the compaction threshold, a routine,
// default-enabled path in any long session, not an error — and (3)
// `this.agent.hasQueuedMessages()`, i.e. an extension queued a follow-up
// during its own agent_end handler. `agent.continue()` emits a fresh
// `agent_start` unconditionally in every case, with willRetry never having
// been true for (2) or (3).
//
// Before the fix: instrumentation.ts's agent_end case only re-queued
// state.pendingInputText when `event.willRetry` was true, so a
// compaction-driven (or extension-queued-follow-up-driven) continuation
// silently discarded the input text instead of handing it to the
// immediately-following phantom agent_start — leaving that run's root span
// with input.value === undefined even though it is semantically still the
// same in-flight session.prompt() call.
// =============================================================================

test('an agent_end continuation with willRetry: false (auto-compaction or an extension-queued follow-up) still reuses the same input text on its phantom agent_start, which fires with no new prompt() call in between', async () => {
  const capture = new CapturingExporter();
  const Session = makeFakeSessionClass();
  const sdk = { AgentSession: Session };
  instrumentPiCodingAgent(sdk, { apiKey: 'test-key', _spanExporter: capture });
  const session = new Session();

  await session.prompt('a long task that crosses the compaction threshold');
  session.emit({ type: 'agent_start' });
  // The turn succeeds normally and crosses the auto-compaction threshold (or
  // an extension queues a follow-up from its agent_end handler) — either
  // way _handlePostAgentRun() returns true for a reason that has nothing to
  // do with the retry heuristic, so willRetry stays false even though
  // agent.continue() is about to fire a fresh agent_start.
  session.emit({
    type: 'agent_end',
    messages: [assistantMessage({ content: [{ type: 'text', text: 'partial progress' }] })],
    willRetry: false,
  });
  // The phantom continuation: agent_start fires again with NO new prompt()
  // call in between.
  session.emit({ type: 'agent_start' });
  session.emit({
    type: 'agent_end',
    messages: [assistantMessage({ content: [{ type: 'text', text: 'final result' }] })],
    willRetry: false,
  });

  const rootSpans = capture.spans.filter((s) => attrs(s)['openinference.span.kind'] === 'AGENT');
  assert.equal(rootSpans.length, 2, 'each agent_start gets its own root span');
  assert.equal(
    attrs(rootSpans[0]!)['input.value'],
    'a long task that crosses the compaction threshold',
    'the first attempt must carry the original prompt text',
  );
  assert.equal(
    attrs(rootSpans[1]!)['input.value'],
    'a long task that crosses the compaction threshold',
    'the continuation root span must still carry the SAME input text even though willRetry was ' +
      'false — the old code only re-queued pendingInputText when event.willRetry was true, so a ' +
      'compaction/extension-follow-up-driven continuation lost it entirely (input.value would be ' +
      'undefined here under the pre-fix code)',
  );
  assert.equal(attrs(rootSpans[1]!)['output.value'], 'final result');

  // A genuine follow-up prompt() call after the continuation finishes must
  // still get its own fresh text — proving the reuse does not leak forward
  // and make a later, unrelated prompt() replay the old text again.
  await session.prompt('a brand new followup prompt');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });

  const allRootSpans = capture.spans.filter((s) => attrs(s)['openinference.span.kind'] === 'AGENT');
  assert.equal(allRootSpans.length, 3);
  assert.equal(
    attrs(allRootSpans[2]!)['input.value'],
    'a brand new followup prompt',
    'after the continuation finished, a genuine new prompt() call must not accidentally replay ' +
      'the old reused text a third time',
  );
});

// -----------------------------------------------------------------------------
// confirmed-bugfix-regressions.test.ts's Bug 4
//
// Plain willRetry:true reuse baseline, queue otherwise empty: the compaction
// test above establishes the willRetry:false continuation-reuse path, and the
// test below establishes the willRetry:true retry-reuse path when a second,
// distinct call is ALREADY queued behind the retry (the harder, adversarial
// case) — but neither pins down the plain, unadorned baseline: a single
// prompt() call, a retry, and nothing else queued. This is exactly the
// "empty-queue-fallback heuristic" the test below's own comment refers back
// to, so it is kept here as that heuristic's positive control.
// -----------------------------------------------------------------------------

test('a retried run (willRetry: true) reuses the SAME input text on its retry agent_start, which fires with no new prompt() call in between', async () => {
  const { capture, Session } = makeRig();
  const session = new Session();

  await session.prompt('flaky prompt that needs a retry');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: true });
  // The agent loop retries: a fresh agent_start fires with NO new prompt()
  // call in between (the host app never re-sent the text).
  session.emit({ type: 'agent_start' });
  session.emit({
    type: 'agent_end',
    messages: [assistantMessage({ content: [{ type: 'text', text: 'succeeded on retry' }] })],
    willRetry: false,
  });

  const rootSpans = capture.spans.filter((s) => attrs(s)['openinference.span.kind'] === 'AGENT');
  assert.equal(rootSpans.length, 2, 'each attempt gets its own root span');
  assert.equal(
    attrs(rootSpans[0]!)['input.value'],
    'flaky prompt that needs a retry',
    'the first attempt must carry the original prompt text',
  );
  assert.equal(
    attrs(rootSpans[1]!)['input.value'],
    'flaky prompt that needs a retry',
    'the RETRY attempt must still carry the SAME input text — it must survive even though ' +
      'agent_end used to unconditionally delete the pendingInput entry regardless of willRetry',
  );
  assert.equal(attrs(rootSpans[1]!)['output.value'], 'succeeded on retry');

  // A genuine follow-up prompt() call after the retry succeeded must get its
  // own fresh text — proving the retry re-queue doesn't leak forward and
  // make a later, unrelated prompt() replay the old retried text again.
  await session.prompt('a brand new followup prompt');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
  const allRootSpans = capture.spans.filter((s) => attrs(s)['openinference.span.kind'] === 'AGENT');
  assert.equal(allRootSpans.length, 3);
  assert.equal(
    attrs(allRootSpans[2]!)['input.value'],
    'a brand new followup prompt',
    'after a successful retry, a genuine new prompt() call must not accidentally replay the old ' +
      'retried text a third time',
  );
});

test('a retry (willRetry: true) phantom agent_start reuses ITS OWN pendingInputText even when a second, genuinely distinct prompt() call is already queued behind it', async () => {
  // Regression coverage for the empty-queue-fallback heuristic above: that
  // heuristic alone is not enough once a second prompt() call is already
  // sitting in the FIFO when the retry's phantom agent_start fires, because
  // the queue is then non-empty and the plain "dequeue front, else fall back
  // to state.pendingInputText" logic incorrectly dequeues the OTHER call's
  // text instead of reusing this run's own. Unlike compaction/follow-up
  // continuations, a retry IS explicitly observable via event.willRetry, so
  // it can (and must) get a stronger, positive-priority guarantee instead of
  // relying on the queue happening to be empty.
  const capture = new CapturingExporter();
  const Session = makeFakeSessionClass();
  const sdk = { AgentSession: Session };
  instrumentPiCodingAgent(sdk, { apiKey: 'test-key', _spanExporter: capture });
  const session = new Session();

  // Both prompt() calls queue before A's agent_start ever fires — e.g. the
  // host app already queued a second, unrelated message while the first was
  // still being dispatched into the agent loop.
  await session.prompt('A');
  await session.prompt('B');

  // A's run starts and claims the front of the queue ('A'), leaving 'B'
  // still queued behind it.
  session.emit({ type: 'agent_start' });
  // A's run ends with a retryable error.
  session.emit({
    type: 'agent_end',
    messages: [assistantMessage({ content: [{ type: 'text', text: 'transient error' }] })],
    willRetry: true,
  });
  // The retry's phantom continuation: agent_start fires again with NO new
  // prompt() call in between. 'B' is still sitting at the front of the
  // queue at this point — the retry must not steal it.
  session.emit({ type: 'agent_start' });
  session.emit({
    type: 'agent_end',
    messages: [assistantMessage({ content: [{ type: 'text', text: 'A succeeded on retry' }] })],
    willRetry: false,
  });

  // B's own run finally gets its turn.
  session.emit({ type: 'agent_start' });
  session.emit({
    type: 'agent_end',
    messages: [assistantMessage({ content: [{ type: 'text', text: 'B result' }] })],
    willRetry: false,
  });

  const rootSpans = capture.spans.filter((s) => attrs(s)['openinference.span.kind'] === 'AGENT');
  assert.equal(rootSpans.length, 3, 'A, A-retry, and B each get their own root span');
  assert.equal(attrs(rootSpans[0]!)['input.value'], 'A', "A's first attempt carries A's text");
  assert.equal(
    attrs(rootSpans[1]!)['input.value'],
    'A',
    "the retry's phantom agent_start must reuse A's OWN text, not wrongly dequeue B's text just " +
      'because B was already queued in front of it',
  );
  assert.equal(attrs(rootSpans[1]!)['output.value'], 'A succeeded on retry');
  assert.equal(
    attrs(rootSpans[2]!)['input.value'],
    'B',
    "B's own run must still get attributed with B's text once it finally runs, proving the " +
      "retry's reservation does not permanently swallow B's queued entry",
  );
  assert.equal(attrs(rootSpans[2]!)['output.value'], 'B result');
});

// =============================================================================
// auto_retry_end reservation disarm
//
// The one-shot reservation armed by agent_end{willRetry:true} is only sound
// if a retry that never actually continues also DISARMS it. Verified against
// the real, installed @earendil-works/pi-coding-agent@0.80.6
// (dist/core/agent-session.js, _prepareRetry): session.abort() during the
// retry's backoff sleep aborts the sleep, which emits
// {type:'auto_retry_end', success:false} and returns false — so NO
// continuation agent_start ever fires for that retry. Without the disarm in
// instrumentation.ts's auto_retry_end case, the stale reservation would be
// consumed by the NEXT, genuinely-new run's agent_start, exporting that run's
// root span with the PREVIOUS run's input text and leaving the new run's own
// queued entry behind to shift every subsequent call off by one.
// =============================================================================

test('a cancelled retry (auto_retry_end success:false) disarms the reservation so the NEXT queued run keeps its own input text', async () => {
  const { capture, Session } = makeRig();
  const session = new Session();

  // Both calls queue before A's agent_start fires; B waits behind A.
  await session.prompt('A');
  await session.prompt('B');

  // A's run starts (claims 'A') and ends with a retryable error, arming the
  // retry reservation for what would be A's phantom continuation.
  session.emit({ type: 'agent_start' });
  session.emit({
    type: 'agent_end',
    messages: [assistantMessage({ content: [{ type: 'text', text: 'transient error' }] })],
    willRetry: true,
  });

  // The host aborts during the retry backoff: the SDK emits auto_retry_end
  // with success:false and the continuation agent_start NEVER fires.
  session.emit({ type: 'auto_retry_end', success: false, attempt: 1 });

  // B's own, genuinely distinct run finally starts. Its agent_start must
  // dequeue B's queued text — not consume the stale reservation and replay
  // A's text onto B's root span.
  session.emit({ type: 'agent_start' });
  session.emit({
    type: 'agent_end',
    messages: [assistantMessage({ content: [{ type: 'text', text: 'B result' }] })],
    willRetry: false,
  });

  const rootSpans = capture.spans.filter((s) => attrs(s)['openinference.span.kind'] === 'AGENT');
  assert.equal(rootSpans.length, 2, "A's aborted run and B's run each get exactly one root span");
  assert.equal(attrs(rootSpans[0]!)['input.value'], 'A');
  assert.equal(
    attrs(rootSpans[1]!)['input.value'],
    'B',
    "B's run must carry B's OWN text — a reservation left armed by the cancelled retry would " +
      "wrongly replay A's text here and strand B's queued entry for the next run",
  );
  assert.equal(attrs(rootSpans[1]!)['output.value'], 'B result');
});

test('auto_retry_end with success:true does NOT disarm anything mid-run, and a retries-exhausted failure (success:false after willRetry:false) is harmless', async () => {
  const { capture, Session } = makeRig();
  const session = new Session();

  // Run 1: a retryable error arms the reservation; the retry proceeds
  // normally (backoff completes, agent.continue() fires agent_start, which
  // consumes the reservation). During the retried run the SDK emits
  // auto_retry_end{success:true} on the first successful assistant message
  // (agent-session.js:353) — AFTER the reservation was already consumed, so
  // it must not disturb anything.
  await session.prompt('flaky prompt');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: true });
  session.emit({ type: 'agent_start' }); // retry continuation, consumes reservation
  session.emit({ type: 'auto_retry_end', success: true, attempt: 1 });
  session.emit({
    type: 'agent_end',
    messages: [assistantMessage({ content: [{ type: 'text', text: 'recovered' }] })],
    willRetry: false,
  });

  // Run 2 exhausts its retries: the final attempt's agent_end reports
  // willRetry:false (maxRetries reached, so _willRetryAfterAgentEnd is
  // false), then the SDK emits the terminal auto_retry_end{success:false}
  // (agent-session.js:753). No reservation was armed, so the disarm is a
  // no-op — and a fresh prompt() afterwards must still attribute normally.
  await session.prompt('doomed prompt');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
  session.emit({ type: 'auto_retry_end', success: false, attempt: 3, finalError: 'gave up' });

  await session.prompt('fresh prompt after exhaustion');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });

  const rootSpans = capture.spans.filter((s) => attrs(s)['openinference.span.kind'] === 'AGENT');
  assert.equal(rootSpans.length, 4);
  assert.equal(attrs(rootSpans[0]!)['input.value'], 'flaky prompt');
  assert.equal(
    attrs(rootSpans[1]!)['input.value'],
    'flaky prompt',
    'the successful retry continuation still reuses its own text; the mid-run ' +
      'auto_retry_end{success:true} must not have disturbed the already-consumed reservation',
  );
  assert.equal(attrs(rootSpans[1]!)['output.value'], 'recovered');
  assert.equal(attrs(rootSpans[2]!)['input.value'], 'doomed prompt');
  assert.equal(
    attrs(rootSpans[3]!)['input.value'],
    'fresh prompt after exhaustion',
    'after a retries-exhausted terminal failure, a genuinely new prompt() must attribute its own ' +
      'text — the no-op disarm on that path must not corrupt the queue',
  );
});
