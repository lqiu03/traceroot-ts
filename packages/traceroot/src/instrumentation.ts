// src/instrumentation.ts
import { registerInstrumentations, type Instrumentation } from '@opentelemetry/instrumentation';
import type { PiInstrumentationConfig } from '@traceroot-ai/pi';
import type { InitializeOptions, PiCodingAgentInstrumentation } from './types';
import { wireOpenAIAgentsProcessor } from './openai-agents';
import { wireClaudeAgentSDKInstrumentation } from './claude-agent-sdk';

/**
 * initialize()'s own resolved apiKey/baseUrl, threaded down to the pi
 * package so a host that configured TraceRoot programmatically (never via the
 * TRACEROOT_API_KEY env var) still gets pi instrumented.
 */
interface PiInstrumentationDefaults {
  apiKey?: string;
  baseUrl?: string;
}

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
 * An `instrumentModules.piCodingAgent` value is the bare `{ module, config }`
 * wrapper form (as opposed to a raw `@earendil-works/pi-coding-agent` module
 * ref) when it carries a `module` property and is NOT itself a pi module
 * namespace. The negative `AgentSession` check guards the (unlikely) case of a
 * future pi module that happens to also export a `module` symbol: the real pi
 * namespace exposes `AgentSession`, the wrapper does not, so the two are
 * always distinguishable.
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
 * Lazy-loads @traceroot-ai/pi and delegates to its instrumentPiCodingAgent(),
 * threading initialize()'s own resolved apiKey/baseUrl (and any explicit
 * per-module config) down to it.
 *
 * pi still auto-discovers the already-registered global provider on its own
 * (this function only ever runs from inside wireInstrumentations(), which
 * TraceRoot.initialize() only calls AFTER _provider.register() has completed
 * -- see traceroot.ts -- so the provider is guaranteed live by the time
 * instrumentPiCodingAgent() checks for it). But pi's own config-resolution
 * still gates the private-provider fallback on an apiKey, and a host that
 * configured TraceRoot programmatically never populates TRACEROOT_API_KEY --
 * so the apiKey/baseUrl MUST be passed explicitly here, or pi silently no-ops.
 *
 * Only the require() is wrapped in try/catch, matching loadInstrumentation()'s
 * contract: a missing optional peer package warns and no-ops rather than
 * crashing initialize(), exactly like a missing OpenInference package does.
 * A failure thrown by instrumentPiCodingAgent() itself is intentionally NOT
 * caught -- it surfaces, just as a throwing new Ctor()/manuallyInstrument()
 * does in the OpenInference loop below.
 */
function wirePiCodingAgentInstrumentation(
  entry: unknown,
  defaults: PiInstrumentationDefaults,
): void {
  let instrumentPiCodingAgent:
    | ((sdk: unknown, config?: PiInstrumentationConfig) => unknown)
    | undefined;
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

  // Unwrap the { module, config } form if given, then merge: an explicit
  // per-module apiKey/baseUrl wins, otherwise initialize()'s resolved defaults
  // fill in. captureContent/captureToolIo (and any other config field) pass
  // straight through from the explicit config so the caller's PII controls
  // reach pi verbatim.
  let mod: unknown = entry;
  let explicitConfig: PiInstrumentationConfig | undefined;
  if (isPiCodingAgentWrapper(entry)) {
    mod = entry.module;
    explicitConfig = entry.config;
  }
  const config: PiInstrumentationConfig = {
    ...explicitConfig,
    apiKey: explicitConfig?.apiKey ?? defaults.apiKey,
    baseUrl: explicitConfig?.baseUrl ?? defaults.baseUrl,
  };
  instrumentPiCodingAgent(mod, config);
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
  piDefaults: PiInstrumentationDefaults = {},
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
    wirePiCodingAgentInstrumentation(instrumentModules.piCodingAgent, piDefaults);
  }

  if (instrs.length > 0) {
    registerInstrumentations({ instrumentations: instrs });
  }
}
