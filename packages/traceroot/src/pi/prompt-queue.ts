/**
 * Owns the ONE invariant "which queued prompt() text belongs to which run" —
 * a per-session FIFO with retry-reservation semantics, guarding against
 * off-by-one input/output attribution. This concept was previously fragmented
 * across two storage locations and five call sites in instrumentation.ts; it
 * lives here, in one place, so the correlation logic can be reasoned about as
 * a single unit instead of a web of cross-referencing comments.
 *
 * Only imports from ./types — it touches no OTel/span machinery, so
 * instrumentation.ts can depend on it with no risk of an import cycle.
 *
 * ── Why a per-session FIFO, not a single slot ─────────────────────────────
 * A second prompt() call can fire before the first run's agent_start event
 * has arrived (overlapping runs on the same session), and each run's
 * agent_start must claim the text from the prompt() call that actually
 * triggered it, in order — never let a later prompt() call's text clobber an
 * earlier one's. One PromptQueue instance is created per AgentSession (via
 * getPromptQueue) and reached from both proto.prompt (enqueue side) and the
 * agent_start/agent_end/auto_retry_end event handlers (claim/reserve side).
 *
 * ── The three ways a fresh agent_start can arrive ─────────────────────────
 * Verified against the real, installed @earendil-works/pi-coding-agent@0.80.6
 * (dist/core/agent-session.js): AgentSession._runAgentPrompt() runs
 * `while (await this._handlePostAgentRun()) { await this.agent.continue(); }`,
 * and _handlePostAgentRun() returns true — firing a fresh agent_start with NO
 * corresponding new prompt() call — for THREE independent reasons:
 *   1. a retryable error (auto-retry),
 *   2. an ordinary auto-compaction continuation (_checkCompaction), or
 *   3. an extension-queued follow-up from its own agent_end handler
 *      (agent.hasQueuedMessages()).
 * None of the three push anything onto the queue. Retry is the ONE reason that
 * is explicitly observable on the event (agent_end's `willRetry`); compaction
 * and extension-queued-follow-up are indistinguishable from a genuinely new
 * call on this event's shape. This asymmetry drives claimForRun()'s two-tier
 * strategy below:
 *   - RETRY gets a strong, positive-priority reservation (reserveForRetry):
 *     the very next agent_start reuses the retrying run's OWN text even when a
 *     second, genuinely distinct prompt() call is already queued behind it.
 *   - COMPACTION / FOLLOW-UP get only the weaker empty-queue fallback: an
 *     empty queue at agent_start (with no reservation active) is treated as a
 *     continuation of whatever run just ran, reusing its remembered input
 *     text rather than a genuinely new call. It is the best available
 *     approximation, since the SDK exposes no field to tell those two apart
 *     from a new call.
 * See prompt-queue.test.ts for the exact cross-call corruption each tier
 * prevents. This enumeration must be kept in sync with the SDK and may miss
 * future continuation paths a later SDK version adds.
 */
import type { AgentSessionInstance, PromptOptions } from './types';

// A queued prompt() call's text, boxed in its own object rather than stored
// as a raw string: proto.prompt's rejection handler needs to remove EXACTLY
// the entry it pushed (by reference identity) if that specific call never
// reaches agent_start, without accidentally removing a different,
// still-pending queue entry that happens to hold an equal string value.
export interface QueuedPrompt {
  text: string;
  // Wall-clock time (Date.now()) this entry was enqueued, consulted at dequeue
  // to detect an entry that was stranded in the FIFO because its prompt() call
  // never reached agent_start (and, resolving rather than rejecting, never hit
  // removeIfStillQueued either). See MAX_QUEUED_PROMPT_AGE_MS.
  enqueuedAt: number;
}

// A queued prompt() entry older than this when a later agent_start arrives to
// claim it is treated as abandoned and skipped rather than trusted. A prompt()
// call whose run never starts — it hangs with no resolve/reject and no
// agent_start, so neither removeIfStillQueued (rejection-only) nor a normal
// agent_start dequeue ever removes it — would otherwise sit in the FIFO
// forever and be shifted out by some unrelated LATER run's agent_start,
// misattributing its text to that run and throwing every subsequent call off
// by one, permanently. This window is deliberately generous — far longer than
// any realistic single agent run a legitimately-queued overlapping prompt()
// could be waiting behind — so a genuinely-queued entry is never discarded,
// while a truly-stranded entry can only ever corrupt a call within this
// bounded window instead of cascading indefinitely.
const MAX_QUEUED_PROMPT_AGE_MS = 60 * 60 * 1000; // 1 hour

/**
 * Decides whether a prompt() call should be SKIPPED (not queued) because it
 * will never reach agent_start — the pure detection that used to be inlined
 * in proto.prompt. Depends only on the session's public surface and the
 * call's options, never on queue state, so it is a free function rather than
 * a PromptQueue method.
 *
 * Choosing a proactive (enumerate-the-early-returns) strategy here rather than
 * a reactive (clean up by reference identity once THIS call's own promise
 * settles) one is deliberate and verified empirically, not assumed. Reading
 * the real, installed @earendil-works/pi-coding-agent@0.80.6 source alone
 * suggests agent_start is guaranteed to have already fired (and shifted some
 * queue entry) by the time prompt()'s returned promise settles, which would
 * make `result.finally(removeIfStillQueued)` sound in theory. It is NOT sound
 * in practice for this FIFO: the very feature this queue exists for (two
 * prompt() calls queuing before EITHER's agent_start has fired — see the "two
 * overlapping prompt() calls" and retry/compaction continuation tests) means a
 * still-genuinely-in-flight call's own promise can settle independently of
 * when ITS OWN agent_start arrives. Trying `result.finally(...)` and running
 * the full suite reproduced exactly that: 13 unrelated, previously-green tests
 * failed because their queued entries were reactively stripped before their
 * own later-emitted agent_start could claim them. Proactive enumeration —
 * covering as many early-return paths as can be determined SYNCHRONOUSLY and
 * precisely from public SDK surface — is the safe option. This must be kept in
 * sync with the SDK and may miss future early-return paths a later version
 * adds.
 */
export function shouldSkipQueue(
  session: AgentSessionInstance,
  text: string,
  options: PromptOptions | undefined,
): boolean {
  // (a) isStreaming + streamingBehavior — verified against the real, installed
  // SDK (dist/core/agent-session.js:812-824): when `this.isStreaming` is true
  // and `options.streamingBehavior` is set, prompt() calls
  // `_queueSteer`/`_queueFollowUp` (which inject the message into the
  // CURRENTLY-running Agent loop, not a new run) and returns — `_runAgentPrompt`
  // is never reached. `isStreaming` is read synchronously, matching the real
  // prompt()'s own synchronous read of the same getter.
  const isStreamedQueueOnly = session.isStreaming === true && !!options?.streamingBehavior;
  // (b) a leading "/" matched by a registered extension command — verified
  // against the real, installed SDK (dist/core/agent-session.js:783-790):
  // `if (expandPromptTemplates && text.startsWith("/")) { const handled =
  // await this._tryExecuteExtensionCommand(text); if (handled) {
  // preflightResult?.(true); return; } }`. This is decidable precisely (not
  // just heuristically) from outside: _tryExecuteExtensionCommand's own command
  // lookup (agent-session.js:903-908) parses the command name identically to
  // below, and its try/catch (913-925) means ANY registered command — even one
  // whose handler throws — still returns true. So a truthy `getCommand()`
  // lookup via the SDK's own public `session.extensionRunner` getter
  // (agent-session.js:2629-2630; ExtensionRunner.getCommand is itself public —
  // dist/core/extensions/runner.d.ts:128) deterministically means this call
  // will never reach agent_start, with no false-positive case.
  const expandPromptTemplates = options?.expandPromptTemplates ?? true;
  let matchesExtensionCommand = false;
  if (expandPromptTemplates && typeof text === 'string' && text.startsWith('/')) {
    const spaceIndex = text.indexOf(' ');
    const commandName = spaceIndex === -1 ? text.slice(1) : text.slice(1, spaceIndex);
    matchesExtensionCommand = !!session.extensionRunner?.getCommand?.(commandName);
  }
  // (c) any extension with an 'input' hook returning action: 'handled' —
  // verified against the real, installed SDK (dist/core/agent-session.js:
  // 794-799): `if (this._extensionRunner.hasHandlers("input")) { const
  // inputResult = await this._extensionRunner.emitInput(...); if
  // (inputResult.action === "handled") { preflightResult?.(true); return; } }`.
  // Unlike (b), this is NOT precisely decidable from outside: whether a
  // specific text ends up "handled" depends on the registered hook function's
  // own logic, which this patch layer cannot inspect or pre-invoke (calling it
  // ourselves would double-invoke a potentially side-effecting extension). The
  // best available signal is the SDK's own public
  // `session.hasExtensionHandlers('input')` (agent-session.js:618, mirrors the
  // exact same `hasHandlers("input")` check prompt() itself makes) — true
  // whenever ANY 'input' hook is registered, whether or not it will actually
  // intercept THIS text. This is a deliberate, documented best-effort
  // heuristic: a session with an 'input' hook that only intercepts SOME
  // messages will, for every message it does NOT intercept, still skip queuing
  // — losing that real run's input.value (falling back to claimForRun's
  // empty-queue heuristic) rather than corrupting a LATER call's attribution.
  // That graceful degradation is preferred over the cross-call corruption this
  // queue exists to prevent, given the SDK exposes no way to precisely predict
  // a specific hook's decision without invoking it.
  const mayBeHandledByInputHook = session.hasExtensionHandlers?.('input') === true;
  return isStreamedQueueOnly || matchesExtensionCommand || mayBeHandledByInputHook;
}

/**
 * Per-session input-attribution state: the FIFO queue itself plus the two
 * fields that used to live on SessionSpanState (the remembered input text and
 * the one-shot retry reservation). One instance per AgentSession, obtained via
 * getPromptQueue().
 */
export class PromptQueue {
  private readonly queue: QueuedPrompt[] = [];
  // The input text the current run's root span was opened with — remembered so
  // a continuation agent_start (retry/compaction/follow-up, none of which
  // enqueue anything) can reuse it. See the module header's "three ways a
  // fresh agent_start can arrive".
  private pendingInputText: string | undefined = undefined;
  // One-shot reservation armed by agent_end{willRetry:true}, consumed by
  // exactly the next agent_start. See the module header for why only the
  // explicitly-observable retry case gets this strong "reuse regardless of
  // what's queued" guarantee.
  private reserveInputForRetry = false;

  /**
   * Adds a prompt()'s text to the tail of the FIFO and returns the boxed entry
   * so the caller can later remove EXACTLY it (by reference identity) if the
   * call never reaches agent_start.
   */
  enqueue(text: string): QueuedPrompt {
    const entry: QueuedPrompt = { text, enqueuedAt: Date.now() };
    this.queue.push(entry);
    return entry;
  }

  /**
   * Removes a specific still-queued entry (by reference identity, never by
   * string match) if it is still present. A no-op for an undefined entry or
   * one already claimed by an agent_start.
   *
   * A prompt() call that never reaches agent_start (a synchronous throw, or its
   * returned Promise rejecting — e.g. a validation failure inside Pi's own
   * prompt() before the agent loop starts) must not leave its text sitting in
   * the FIFO: agent_start will never fire to claim it, so it would otherwise
   * become the wrong (stale) input text attached to whatever LATER,
   * genuinely-successful prompt() call claims it instead — and every prompt()
   * after that would be off by one, permanently. Callers gate this on rejection
   * only (.catch(), not .finally()) — see shouldSkipQueue's proactive-vs-
   * reactive note for why widening it to resolve is unsafe for this FIFO's
   * overlapping-call semantics.
   */
  removeIfStillQueued(entry: QueuedPrompt | undefined): void {
    if (!entry) return;
    const idx = this.queue.indexOf(entry);
    if (idx !== -1) this.queue.splice(idx, 1);
  }

  /**
   * Resolves the input text for a run whose agent_start just fired, and
   * remembers it for any continuation that follows. This is the entire
   * reservation-vs-dequeue-vs-remembered-text decision (see the module header
   * for the full rationale):
   *   1. If a retry reservation is armed, consume it and reuse the remembered
   *      text — this wins even over a non-empty queue.
   *   2. Otherwise take the oldest genuinely-fresh queued entry, if any.
   *   3. Otherwise (empty queue, no reservation) fall back to the remembered
   *      text — this run is a continuation of the previous one.
   */
  claimForRun(): string | undefined {
    let inputText: string | undefined;
    if (this.reserveInputForRetry) {
      this.reserveInputForRetry = false;
      inputText = this.pendingInputText;
    } else {
      const queuedEntry = this.dequeueFreshQueuedPrompt();
      inputText = queuedEntry ? queuedEntry.text : this.pendingInputText;
    }
    this.pendingInputText = inputText;
    return inputText;
  }

  /**
   * Arms the one-shot retry reservation. Called by agent_end when
   * event.willRetry is true; consumed by exactly the next agent_start's
   * claimForRun(). The remembered input text is deliberately NOT cleared on
   * agent_end regardless of willRetry, so the empty-queue fallback still works
   * for compaction/follow-up continuations (see the module header).
   */
  reserveForRetry(): void {
    this.reserveInputForRetry = true;
  }

  /**
   * Disarms the retry reservation. Called by auto_retry_end when the SDK
   * cancelled or exhausted the retry, so no continuation agent_start will
   * consume it — clearing it prevents a LATER, genuinely-new prompt() call's
   * agent_start from mis-consuming the previous run's input.value.
   */
  clearReservation(): void {
    this.reserveInputForRetry = false;
  }

  /**
   * Shifts the oldest genuinely-fresh entry off the FIFO, discarding any head
   * entries that have gone stale (their prompt() call never reached agent_start
   * — see MAX_QUEUED_PROMPT_AGE_MS). Skips past every stale head entry rather
   * than stopping at the first, so a run of stranded entries can never block
   * the fresh one queued behind them. Returns undefined when the queue holds
   * nothing but stale entries (or is empty), so claimForRun falls back to the
   * remembered text instead of trusting stale text. Warns once per dequeue that
   * discarded anything, so a real "prompt never started" anomaly is visible.
   */
  private dequeueFreshQueuedPrompt(): QueuedPrompt | undefined {
    const now = Date.now();
    let discarded = 0;
    let entry = this.queue.shift();
    while (entry && now - entry.enqueuedAt > MAX_QUEUED_PROMPT_AGE_MS) {
      discarded += 1;
      entry = this.queue.shift();
    }
    if (discarded > 0) {
      console.warn(
        `[traceroot-pi] discarded ${discarded} stale queued prompt(s) whose run never started ` +
          '(older than the staleness window) instead of misattributing their text to this run.',
      );
    }
    return entry;
  }
}

/**
 * Returns the session's PromptQueue, lazily creating and storing one on first
 * use. Mirrors the WeakMap-per-session pattern used for SessionSpanState — the
 * enqueue side (proto.prompt) and the claim side (the event handlers) both go
 * through this, so they always share one instance regardless of which side
 * touches the session first.
 */
export function getPromptQueue(
  queues: WeakMap<AgentSessionInstance, PromptQueue>,
  session: AgentSessionInstance,
): PromptQueue {
  let queue = queues.get(session);
  if (!queue) {
    queue = new PromptQueue();
    queues.set(session, queue);
  }
  return queue;
}
