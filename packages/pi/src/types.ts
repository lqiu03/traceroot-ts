/**
 * Hand-transcribed local mirror of the `@earendil-works/pi-coding-agent` /
 * `@earendil-works/pi-agent-core` surface this package touches.
 *
 * Deliberately NOT imported from the real packages, matching the confirmed
 * convention in packages/traceroot/src/claude-agent-sdk.ts (which hand-rolls
 * a structural ClaudeAgentSDKModule type rather than depending on
 * @anthropic-ai/claude-agent-sdk) and traceroot-pi-extension's own
 * src/types.ts. Keeps this package free of any dependency — even a
 * type-only devDependency — on Pi's own package, and easy to re-verify
 * against a new Pi version by hand.
 *
 * Every field here was confirmed against the real published .d.ts for
 * @earendil-works/pi-coding-agent@0.80.6, @earendil-works/pi-agent-core@0.80.6,
 * and @earendil-works/pi-ai@0.80.6 (npm pack + inspection), not guessed.
 */

export interface Usage {
  input: number;
  output: number;
  cacheRead: number;
  cacheWrite: number;
  reasoning?: number;
  totalTokens: number;
  cost: {
    input: number;
    output: number;
    cacheRead: number;
    cacheWrite: number;
    total: number;
  };
}

export type StopReason = 'stop' | 'length' | 'toolUse' | 'error' | 'aborted';

export interface AssistantMessage {
  role: 'assistant';
  content: unknown[];
  api: string;
  provider: string;
  model: string;
  responseModel?: string;
  responseId?: string;
  usage: Usage;
  stopReason: StopReason;
  errorMessage?: string;
  timestamp: number;
}

export interface UserMessage {
  role: 'user';
  content: unknown;
  timestamp: number;
}

export interface ToolResultMessage {
  role: 'toolResult';
  toolCallId: string;
  toolName: string;
  content: unknown[];
  details?: unknown;
  isError: boolean;
  timestamp: number;
}

/**
 * `@earendil-works/pi-agent-core`'s real `AgentMessage` is
 * `Message | CustomAgentMessages[keyof CustomAgentMessages]`, and
 * `@earendil-works/pi-coding-agent`'s `dist/core/messages.d.ts` augments
 * `CustomAgentMessages` with four additional roles this package doesn't
 * otherwise model: `bashExecution`, `custom`, `branchSummary`,
 * `compactionSummary` (npm pack + inspection of both packages@0.80.6).
 * `sendCustomMessage()` emits a live `message_start`/`message_end` pair with
 * `role: 'custom'` while idle (dist/core/agent-session.js:~1074), so these
 * are reachable at runtime, not merely declared.
 *
 * Every consumer in this package narrows by `.role` before touching any
 * other field (`message.role === 'assistant'` / `!== 'assistant'`), so
 * these four are never dereferenced beyond `.role` here — kept minimal
 * rather than fabricating field shapes this package doesn't act on.
 */
export interface OtherAgentMessage {
  role: 'bashExecution' | 'custom' | 'branchSummary' | 'compactionSummary';
  timestamp?: number;
}

export type AgentMessage = AssistantMessage | UserMessage | ToolResultMessage | OtherAgentMessage;

/**
 * The event union emitted by `AgentSession.subscribe()` and the lower-level
 * `Agent.subscribe()`. `AgentSession.subscribe()` is a strict superset
 * (adds `willRetry` to `agent_end`, plus session-level events this package
 * does not use) — only the shared, raw `AgentEvent` shapes are mirrored here.
 */
export type AgentEvent =
  | { type: 'agent_start' }
  | { type: 'agent_end'; messages: AgentMessage[]; willRetry?: boolean }
  | { type: 'turn_start' }
  | { type: 'turn_end'; message: AgentMessage; toolResults: ToolResultMessage[] }
  | { type: 'message_start'; message: AgentMessage }
  | { type: 'message_update'; message: AgentMessage }
  | { type: 'message_end'; message: AgentMessage }
  | { type: 'tool_execution_start'; toolCallId: string; toolName: string; args: unknown }
  | {
      type: 'tool_execution_update';
      toolCallId: string;
      toolName: string;
      args: unknown;
      partialResult: unknown;
    }
  | {
      type: 'tool_execution_end';
      toolCallId: string;
      toolName: string;
      result: unknown;
      isError: boolean;
    };

export interface PromptOptions {
  expandPromptTemplates?: boolean;
  images?: unknown[];
  streamingBehavior?: 'steer' | 'followUp';
  source?: unknown;
  preflightResult?: (success: boolean) => void;
}

export interface AgentSessionInstance {
  readonly sessionId?: string;
  /**
   * True while a run is actively executing. Confirmed against the real,
   * installed @earendil-works/pi-coding-agent@0.80.6
   * (dist/core/agent-session.js:572, `get isStreaming()`) and used by
   * prompt() itself (agent-session.js:812) to decide whether to queue via
   * steer()/followUp() (see PromptOptions.streamingBehavior) instead of
   * starting a fresh run. instrumentation.ts's proto.prompt wrapper reads
   * this same getter to avoid queuing a pendingInput FIFO entry for a call
   * that will never reach agent_start.
   */
  readonly isStreaming?: boolean;
  prompt(text: string, options?: PromptOptions): Promise<void>;
  /**
   * Whether any extension has registered a handler for the given event type
   * (e.g. `'input'`). Confirmed public on the real, installed
   * @earendil-works/pi-coding-agent@0.80.6 (dist/core/agent-session.d.ts:618,
   * `hasExtensionHandlers(eventType: string): boolean`) — mirrors the exact
   * `this._extensionRunner.hasHandlers("input")` check prompt() itself makes
   * (dist/core/agent-session.js:794) before dispatching to an 'input' hook.
   * Optional here, like steer()/followUp()/dispose(), so a minimal/partial
   * double never disables prompt instrumentation over a missing method.
   * instrumentation.ts's proto.prompt wrapper uses this as a best-effort
   * signal (NOT a precise one — see its own comment) that a call might be
   * fully intercepted by an 'input' hook and never reach agent_start.
   */
  hasExtensionHandlers?(eventType: string): boolean;
  /**
   * The session's extension runner, exposing (among other things)
   * `getCommand(name)` — confirmed public on the real, installed
   * @earendil-works/pi-coding-agent@0.80.6 (dist/core/agent-session.d.ts:622,
   * `get extensionRunner(): ExtensionRunner`; dist/core/extensions/
   * runner.d.ts:128, `getCommand(name: string): ResolvedCommand | undefined`
   * on ExtensionRunner itself). Only the one method this package actually
   * calls is declared here — see this package's own types.ts header comment
   * on why the full real SDK type is deliberately not imported.
   * instrumentation.ts's proto.prompt wrapper uses this to deterministically
   * detect a leading "/" prompt that a registered extension command will
   * fully handle, so it never reaches agent_start.
   */
  readonly extensionRunner?: {
    getCommand?(name: string): unknown;
  };
  /**
   * Queue a steering message while the agent is running — delivered after
   * the current assistant turn finishes its tool calls, before the next LLM
   * call. A standalone public entry point distinct from prompt(text, {
   * streamingBehavior: 'steer' }): a host can call this directly without
   * ever having called prompt() on the session first. Optional here (rather
   * than required, like prompt/subscribe) so a minimal/partial double never
   * disables prompt instrumentation over a missing, unrelated method —
   * mirrors dispose()'s own optionality in this interface.
   */
  steer?(text: string, images?: unknown[]): Promise<void>;
  /**
   * Queue a follow-up message to be processed once the agent has no more
   * tool calls or steering messages left. Same standalone-entry-point
   * caveat as steer() above.
   */
  followUp?(text: string, images?: unknown[]): Promise<void>;
  subscribe(listener: (event: AgentEvent) => void): () => void;
  /**
   * Removes every listener registered via subscribe() and disconnects from
   * the underlying Agent. Confirmed against the real, installed
   * @earendil-works/pi-coding-agent@0.80.6 dist/core/agent-session.js:
   * dispose() reassigns the private `_eventListeners` array (the same array
   * subscribe() pushes into and _emit() reads on every dispatch) to a fresh
   * empty array, so no subscribe() listener — including instrumentation.ts's
   * own — is ever invoked again after dispose() runs. See instrumentation.ts's
   * module header comment for the full verified call chain.
   */
  dispose(): void;
}

export interface AgentSessionConstructor {
  prototype: AgentSessionInstance;
}

/** Structural shape of the imported `@earendil-works/pi-coding-agent` module namespace. */
export interface PiCodingAgentModule {
  AgentSession?: AgentSessionConstructor;
  [key: string]: unknown;
}
