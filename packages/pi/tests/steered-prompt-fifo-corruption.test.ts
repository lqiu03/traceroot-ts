/**
 * A prompt() call that resolves successfully but never reaches agent_start
 * must not leave a stale entry in the pendingInput FIFO queue either — not
 * just the throw/reject case already covered by
 * rejected-prompt-fifo-leak.test.ts.
 *
 * Verified against the real, installed @earendil-works/pi-coding-agent@0.80.6
 * (dist/core/agent-session.js:812-824): when `this.isStreaming` is true and
 * `options.streamingBehavior` is set ('steer' or 'followUp'), prompt() calls
 * `_queueSteer`/`_queueFollowUp` — which inject the message into the
 * CURRENTLY-running Agent loop, not a new run — and returns. It never reaches
 * `_runAgentPrompt`, the only call site that leads to a fresh agent_start.
 * The promise resolves, not rejects.
 *
 * FakeAgentSession below faithfully mirrors that gate: `isStreaming` is a
 * real, mutable property (flipped by the test the same way the real SDK's
 * own isStreaming lifecycle would — set on a fresh run, cleared once
 * agent_end fires), and prompt() only "runs" (i.e. becomes the kind of call
 * that will eventually produce agent_start) when NOT already streaming, or
 * when streaming without a streamingBehavior option.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import type { ExportResult } from '@opentelemetry/core';
import { ExportResultCode } from '@opentelemetry/core';
import type { ReadableSpan, SpanExporter } from '@opentelemetry/sdk-trace-base';
import { instrumentPiCodingAgent } from '../src/instrumentation';
import type { AgentEvent, AssistantMessage } from '../src/types';

class CapturingExporter implements SpanExporter {
  readonly spans: ReadableSpan[] = [];
  export(spans: ReadableSpan[], resultCallback: (result: ExportResult) => void): void {
    this.spans.push(...spans);
    resultCallback({ code: ExportResultCode.SUCCESS });
  }
  async shutdown(): Promise<void> {}
}

function makeFakeSessionClass() {
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

test("a steer()/followUp() call queued in during an active run (resolves, no agent_start) does not corrupt the NEXT run's input.value", async () => {
  const capture = new CapturingExporter();
  const Session = makeFakeSessionClass();
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
  const Session = makeFakeSessionClass();
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
