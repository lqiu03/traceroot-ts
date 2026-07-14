// src/instrumentation.ts
import { registerInstrumentations, type Instrumentation } from '@opentelemetry/instrumentation';
import type { InitializeOptions } from './types';
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
 * Lazy-loads @traceroot-ai/pi and delegates to its instrumentPiCodingAgent().
 * Deliberately a bare 1-argument call -- no tracer or provider is passed
 * explicitly. instrumentPiCodingAgent() auto-discovers the
 * already-registered global provider on its own (this function only ever
 * runs from inside wireInstrumentations(), which TraceRoot.initialize()
 * only calls AFTER _provider.register() has already completed -- see
 * traceroot.ts:160-162 -- so the provider is guaranteed to be live and
 * registered by the time instrumentPiCodingAgent() checks for it).
 *
 * Only the require() is wrapped in try/catch, matching loadInstrumentation()'s
 * contract: a missing optional peer package warns and no-ops rather than
 * crashing initialize(), exactly like a missing OpenInference package does.
 * A failure thrown by instrumentPiCodingAgent() itself is intentionally NOT
 * caught -- it surfaces, just as a throwing new Ctor()/manuallyInstrument()
 * does in the OpenInference loop below.
 */
function wirePiCodingAgentInstrumentation(mod: unknown): void {
  let instrumentPiCodingAgent: ((sdk: unknown, config?: unknown) => unknown) | undefined;
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const pkg = require('@traceroot-ai/pi') as { instrumentPiCodingAgent?: unknown };
    if (typeof pkg.instrumentPiCodingAgent === 'function') {
      instrumentPiCodingAgent = pkg.instrumentPiCodingAgent as typeof instrumentPiCodingAgent;
    }
  } catch {
    console.warn(
      '[TraceRoot] instrumentModules.piCodingAgent was provided but @traceroot-ai/pi is not ' +
        'installed. Install it: npm install @traceroot-ai/pi',
    );
    return;
  }
  if (!instrumentPiCodingAgent) {
    console.warn(
      '[TraceRoot] @traceroot-ai/pi is installed but does not export instrumentPiCodingAgent(). ' +
        'Check your installed version.',
    );
    return;
  }
  instrumentPiCodingAgent(mod);
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
