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

export type AgentMessage = AssistantMessage | UserMessage | ToolResultMessage;

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
  prompt(text: string, options?: PromptOptions): Promise<void>;
  subscribe(listener: (event: AgentEvent) => void): () => void;
}

export interface AgentSessionConstructor {
  prototype: AgentSessionInstance;
}

/** Structural shape of the imported `@earendil-works/pi-coding-agent` module namespace. */
export interface PiCodingAgentModule {
  AgentSession?: AgentSessionConstructor;
  [key: string]: unknown;
}
