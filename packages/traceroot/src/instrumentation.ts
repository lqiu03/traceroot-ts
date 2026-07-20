// src/instrumentation.ts
import { registerInstrumentations, type Instrumentation } from '@opentelemetry/instrumentation';
import { instrumentPiCodingAgent, type PiInstrumentationConfig } from './pi';
import type { InitializeOptions, PiCodingAgentInstrumentation } from './types';
import { wireOpenAIAgentsProcessor } from './openai-agents';
import { wireClaudeAgentSDKInstrumentation } from './claude-agent-sdk';

type InstrumentationWithManualPatch = Instrumentation & {
  manuallyInstrument(moduleRef: unknown): void;
};

type InstrumentationCtor = new () => InstrumentationWithManualPatch;

const OPENINFERENCE_PACKAGES = {
  openAI: ['@arizeai/openinference-instrumentation-openai', 'OpenAIInstrumentation'],
  anthropic: ['@arizeai/openinference-instrumentation-anthropic', 'AnthropicInstrumentation'],
  langchain: ['@arizeai/openinference-instrumentation-langchain', 'LangChainInstrumentation'],
  bedrock: ['@arizeai/openinference-instrumentation-bedrock', 'BedrockInstrumentation'],
} as const;

function loadInstrumentation(pkg: string, exportName: string): InstrumentationCtor | null {
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const mod = require(pkg) as Record<string, unknown>;
    const ctor = mod[exportName];
    if (typeof ctor !== 'function') return null;
    return ctor as InstrumentationCtor;
  } catch {
    return null;
  }
}

/**
 * An `instrumentModules.piCodingAgent` value is the `{ module, config }`
 * wrapper form (as opposed to a raw `@earendil-works/pi-coding-agent` module
 * ref) when it carries a `module` property and is NOT itself a pi module
 * namespace. The negative `AgentSession` check guards against a future pi
 * module that also exports a `module` symbol: the real pi namespace exposes
 * `AgentSession`, the wrapper does not, so the two stay distinguishable.
 */
function isPiCodingAgentWrapper(entry: unknown): entry is PiCodingAgentInstrumentation {
  return (
    typeof entry === 'object' &&
    entry !== null &&
    'module' in entry &&
    (entry as { module?: unknown }).module != null &&
    !('AgentSession' in entry)
  );
}

/**
 * Unwraps the { module, config } form if given, then delegates directly to
 * the in-tree instrumentPiCodingAgent() (./pi) -- pi is a
 * core module now, not a separately-installed package, so there's no dynamic
 * require() or missing-optional-peer diagnosis.
 *
 * Unlike claudeAgentSDK, pi keeps its own `config` (captureContent/
 * captureToolIo -- see PiCodingAgentInstrumentation in types.ts) and passes
 * it through unmerged; there is no apiKey/baseUrl to thread, since this
 * in-tree integration builds no export pipeline of its own (see
 * pi.ts's module header).
 *
 * A failure thrown by instrumentPiCodingAgent() itself is intentionally NOT
 * caught here -- it surfaces, same as the OpenInference loop below.
 */
function wirePiCodingAgentInstrumentation(entry: unknown): void {
  let mod: unknown = entry;
  let explicitConfig: PiInstrumentationConfig | undefined;
  if (isPiCodingAgentWrapper(entry)) {
    mod = entry.module;
    explicitConfig = entry.config;
  }
  instrumentPiCodingAgent(mod, explicitConfig);
}

/**
 * Wires OpenInference instrumentations based on the instrumentModules option:
 *
 * - undefined  -> RITM auto-instrumentation for all supported modules (CJS only)
 * - {}         -> no instrumentation
 * - { openAI } -> manual patch only the provided module refs
 *
 * Called once by TraceRoot.initialize().
 *
 * All OpenInference instrumentations are lazy-loaded via require() so that
 * missing peer dependencies don't crash initialization.
 */
export function wireInstrumentations(
  instrumentModules: InitializeOptions['instrumentModules'],
): void {
  if (instrumentModules === undefined) {
    // Auto-instrumentation via require-in-the-middle (CJS only).
    // ESM users must pass explicit module refs.
    const instrs: Instrumentation[] = [];
    for (const [pkg, exportName] of Object.values(OPENINFERENCE_PACKAGES)) {
      const Ctor = loadInstrumentation(pkg, exportName);
      if (Ctor) instrs.push(new Ctor());
    }
    if (instrs.length > 0) {
      registerInstrumentations({ instrumentations: instrs });
    }
    return;
  }

  const instrs: Instrumentation[] = [];

  for (const [key, [pkg, exportName]] of Object.entries(OPENINFERENCE_PACKAGES)) {
    const moduleRef = instrumentModules[key as keyof typeof OPENINFERENCE_PACKAGES];
    if (!moduleRef) continue;
    const Ctor = loadInstrumentation(pkg, exportName);
    if (!Ctor) {
      throw new Error(`[TraceRoot] Failed to load ${pkg}. Install it: npm install ${pkg}`);
    }
    const instr = new Ctor();
    instrs.push(instr);
    instr.manuallyInstrument(moduleRef);
  }

  if (instrumentModules.claudeAgentSDK) {
    wireClaudeAgentSDKInstrumentation(instrumentModules.claudeAgentSDK);
  }
  if (instrumentModules.openaiAgents) {
    wireOpenAIAgentsProcessor(instrumentModules.openaiAgents);
  }
  if (instrumentModules.piCodingAgent) {
    wirePiCodingAgentInstrumentation(instrumentModules.piCodingAgent);
  }

  if (instrs.length > 0) {
    registerInstrumentations({ instrumentations: instrs });
  }
}
