// src/traceroot.ts
import {
  context,
  diag,
  DiagConsoleLogger,
  DiagLogLevel,
  propagation,
  trace,
} from '@opentelemetry/api';
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-proto';
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import type { SDKRegistrationConfig } from '@opentelemetry/sdk-trace-base';
import {
  OpenInferenceBatchSpanProcessor,
  OpenInferenceSimpleSpanProcessor,
} from '@arizeai/openinference-vercel';
import { InitializeOptions } from './types';
import { SDK_NAME, SDK_VERSION, TraceRootSpanProcessor } from './processor';
import { wireInstrumentations } from './instrumentation';
import { DEFAULT_FLUSH_AT, DEFAULT_FLUSH_INTERVAL_SEC, DEFAULT_TIMEOUT_SEC } from './constants';
import { _resetObserveState } from './observe';
import {
  autoDetectGitContext,
  getGitRoot,
  harvestCiGitContext,
  gitContextFromFiles,
  _resetGitContextCache,
} from './git_context';

const DEFAULT_BASE_URL = 'https://app.traceroot.ai';

let _isInitialized = false;
let _provider: NodeTracerProvider | undefined;
// The beforeExit listener initialize() installed, so shutdown()/_resetForTesting()
// can remove exactly that one instead of leaving it to fire (harmlessly, but as an
// accumulating listener) on every future process exit after a shutdown.
let _beforeExitHandler: (() => void) | undefined;
// The exact ContextManager / TextMapPropagator instances TraceRoot ATTEMPTED to
// register at initialize() time. NodeTracerProvider.register(config) mutates the
// config object it's handed, filling in config.contextManager (always) and
// config.propagator (from env defaults, may stay undefined) with the concrete
// instances it tried to install — regardless of whether those specific
// registrations actually WON the process-wide first-write-wins slot. Captured
// here so shutdown() can verify, per-slot and by reference identity, whether
// TraceRoot still owns each of context/propagation before resetting it. See the
// three ownership predicates below.
let _registeredContextManager: unknown;
let _registeredPropagator: unknown;

/**
 * OpenTelemetry's global registry (@opentelemetry/api's internal
 * global-utils.js) tracks `trace`, `context`, and `propagation` as THREE
 * SEPARATE first-write-wins slots. TraceRoot can win some and lose others — a
 * host can win the `context` slot on its own (e.g. context.setGlobalContextManager()
 * directly, with no tracer provider) while TraceRoot still wins `trace` and
 * `propagation`. So ownership of each slot must be checked independently before
 * tearing it down; a single combined check would let shutdown() destroy a slot
 * TraceRoot never owned.
 */

/**
 * True when `provider` is the concrete TracerProvider the OTel global proxy
 * currently delegates to — i.e. TraceRoot's registration still owns the `trace`
 * slot, so it's safe to reset it on teardown. Duck-typed on getDelegate()
 * (matching pi's own provider detection) rather than instanceof
 * ProxyTracerProvider, which is defeated by a dual-copy @opentelemetry/api
 * install.
 */
function isActiveGlobalDelegate(provider: NodeTracerProvider): boolean {
  const globalProvider = trace.getTracerProvider() as { getDelegate?: () => unknown };
  const delegate =
    typeof globalProvider.getDelegate === 'function'
      ? globalProvider.getDelegate()
      : globalProvider;
  return delegate === provider;
}

/**
 * True when the ContextManager currently installed in the global `context` slot
 * is the exact instance TraceRoot registered — i.e. TraceRoot won the `context`
 * slot and it hasn't since been replaced. context._getContextManager() is an
 * underscore-prefixed-by-convention runtime accessor on the ContextAPI
 * singleton (not truly private — same duck-typing isActiveGlobalDelegate() uses
 * for getDelegate()); reached via bracket notation since the .d.ts doesn't
 * expose it. Returns false when TraceRoot never registered one.
 */
function isActiveContextManager(): boolean {
  if (_registeredContextManager === undefined) return false;
  const current = (
    context as unknown as { _getContextManager?: () => unknown }
  )._getContextManager?.();
  return current === _registeredContextManager;
}

/**
 * True when the TextMapPropagator currently installed in the global
 * `propagation` slot is the exact instance TraceRoot registered — i.e. TraceRoot
 * won the `propagation` slot and it hasn't since been replaced.
 * propagation._getGlobalPropagator() is the analogous runtime accessor on the
 * PropagationAPI singleton. Returns false when TraceRoot never registered one
 * (e.g. OTEL_PROPAGATORS resolved to nothing, leaving config.propagator unset).
 */
function isActivePropagator(): boolean {
  if (_registeredPropagator === undefined) return false;
  const current = (
    propagation as unknown as { _getGlobalPropagator?: () => unknown }
  )._getGlobalPropagator?.();
  return current === _registeredPropagator;
}

/**
 * Resets exactly the global trace/context/propagation slots THIS module's
 * registration still owns, per the three independent ownership predicates
 * above, and clears the bookkeeping fields that back them. Shared by
 * shutdown() (the normal teardown path) and initialize()'s wiring-failure
 * rollback (an abnormal teardown of a provider that only ever got as far as
 * register()) — both need the identical per-slot ownership dance, and
 * duplicating it would leave two copies of a subtle invariant to keep in
 * sync. Does NOT touch _isInitialized, _provider, or call provider.shutdown()
 * — callers own those since the two call sites differ on exactly those
 * points (sync vs async, and what "provider" even means at that point).
 */
function _releaseGlobalSlots(provider: NodeTracerProvider | undefined): void {
  if (provider && isActiveGlobalDelegate(provider)) {
    trace.disable();
  }
  if (isActiveContextManager()) {
    context.disable();
  }
  if (isActivePropagator()) {
    propagation.disable();
  }
  _registeredContextManager = undefined;
  _registeredPropagator = undefined;
}

export class TraceRoot {
  private constructor() {}

  static isInitialized(): boolean {
    return _isInitialized;
  }

  static initialize(options: InitializeOptions = {}): void {
    const enabled = options.enabled ?? process.env['TRACEROOT_ENABLED'] !== 'false';
    if (!enabled) {
      return;
    }

    if (_isInitialized) {
      console.warn('[TraceRoot] Already initialized. Skipping duplicate initialize() call.');
      return;
    }

    const apiKey = options.apiKey ?? process.env['TRACEROOT_API_KEY'];
    if (!apiKey) {
      console.warn(
        '[TraceRoot] No API key provided. Set TRACEROOT_API_KEY env var or pass apiKey to initialize(). ' +
          'Spans will be emitted but export will fail.',
      );
    }

    const logLevelMap: Record<string, DiagLogLevel> = {
      debug: DiagLogLevel.DEBUG,
      info: DiagLogLevel.INFO,
      warn: DiagLogLevel.WARN,
      error: DiagLogLevel.ERROR,
    };
    diag.setLogger(
      new DiagConsoleLogger(),
      logLevelMap[options.logLevel ?? 'error'] ?? DiagLogLevel.ERROR,
    );

    const baseUrl = (
      options.baseUrl ??
      process.env['TRACEROOT_HOST_URL'] ??
      DEFAULT_BASE_URL
    ).replace(/\/$/, '');

    const headers: Record<string, string> = {
      'x-traceroot-sdk-name': SDK_NAME,
      'x-traceroot-sdk-version': SDK_VERSION,
    };
    if (apiKey) headers['Authorization'] = `Bearer ${apiKey}`;

    const exporter = new OTLPTraceExporter({
      url: `${baseUrl}/api/v1/public/traces`,
      headers,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      compression: 'gzip' as any,
    });

    const environment = options.environment ?? process.env['TRACEROOT_ENVIRONMENT'];

    // `|| undefined` so an empty option/env var ('') is treated as unset —
    // otherwise it would block fallback detection and suppress the warning.
    let gitRepo = options.gitRepo || process.env['TRACEROOT_GIT_REPO'] || undefined;
    let gitRef = options.gitRef || process.env['TRACEROOT_GIT_REF'] || undefined;

    // CI/platform env vars — production path (no .git needed).
    if (gitRepo === undefined || gitRef === undefined) {
      const ci = harvestCiGitContext();
      gitRepo ??= ci.gitRepo;
      gitRef ??= ci.gitRef;
    }

    // .git read as files — dev fallback, no git binary required.
    if (gitRepo === undefined || gitRef === undefined) {
      const fromFiles = gitContextFromFiles();
      gitRepo ??= fromFiles.gitRepo;
      gitRef ??= fromFiles.gitRef;
    }

    // git subprocess — last resort (handles run-from-subdirectory).
    if (gitRepo === undefined || gitRef === undefined) {
      const autoGit = autoDetectGitContext();
      gitRepo ??= autoGit.gitRepo;
      gitRef ??= autoGit.gitRef;
    }

    // Warm git-root cache for per-span source paths (cached/idempotent).
    getGitRoot();

    // Honest absence: warn once, never fabricate.
    if (gitRepo === undefined || gitRef === undefined) {
      console.warn(
        '[TraceRoot] git context incomplete (repo=' +
          (gitRepo ?? 'unset') +
          ', ref=' +
          (gitRef ?? 'unset') +
          '). The AI agent needs both to correlate traces to source. ' +
          'Set TRACEROOT_GIT_REPO / TRACEROOT_GIT_REF (see ' +
          'https://docs.traceroot.ai/tracing/git-context).',
      );
    }

    // Flush/batch tuning — env vars take precedence over SDK defaults.
    const flushIntervalSec = Number(
      process.env['TRACEROOT_FLUSH_INTERVAL'] || DEFAULT_FLUSH_INTERVAL_SEC,
    );
    const flushAt = Number(process.env['TRACEROOT_FLUSH_AT'] || DEFAULT_FLUSH_AT);
    const timeoutSec = Number(process.env['TRACEROOT_TIMEOUT'] || DEFAULT_TIMEOUT_SEC);

    // ── Vercel AI SDK: wrap the exporter in OpenInference span processors ──────
    // These processors enrich spans emitted by Vercel AI SDK's experimental_telemetry
    // with OpenInference semantic conventions (model name, token counts, IO, etc.)
    // before they reach the OTLP exporter. All other SDK spans pass through unchanged.
    const innerProcessor = options.disableBatch
      ? new OpenInferenceSimpleSpanProcessor({ exporter })
      : new OpenInferenceBatchSpanProcessor({
          exporter,
          config: {
            scheduledDelayMillis: flushIntervalSec * 1000,
            maxExportBatchSize: flushAt,
            exportTimeoutMillis: timeoutSec * 1000,
          },
        });

    _provider = new NodeTracerProvider();
    _provider.addSpanProcessor(
      new TraceRootSpanProcessor(innerProcessor, { environment, gitRepo, gitRef }),
    );
    // Pass our own config object into register() so we can read back the exact
    // ContextManager / TextMapPropagator instances it tried to install (it
    // mutates this object in place — see _registeredContextManager's comment).
    // These are what shutdown()'s per-slot ownership checks compare against.
    const registerConfig: SDKRegistrationConfig = {};
    _provider.register(registerConfig);
    _registeredContextManager = registerConfig.contextManager;
    _registeredPropagator = registerConfig.propagator;

    try {
      wireInstrumentations(options.instrumentModules);
    } catch (error) {
      // register() above already won the global trace/context/propagation
      // slots. If wiring then throws (e.g. a misshaped instrumentModules
      // entry), leaving that registration in place would break the
      // "a registered global provider <=> _isInitialized === true" invariant:
      // _isInitialized stays false, so the double-init guard never fires, but
      // a retried initialize()'s new provider would lose the first-write-wins
      // race for slots this orphaned provider still holds — silently
      // stranding the process without a working export pipeline. Tear down
      // exactly what this call registered before re-throwing.
      const orphaned = _provider;
      _releaseGlobalSlots(orphaned);
      _provider = undefined;
      void orphaned?.shutdown().catch(() => {});
      throw error;
    }

    _isInitialized = true;
    _beforeExitHandler = () => {
      void _provider?.forceFlush();
    };
    process.once('beforeExit', _beforeExitHandler);
  }

  static async flush(): Promise<void> {
    await _provider?.forceFlush();
  }

  static async shutdown(): Promise<void> {
    const provider = _provider;
    if (_beforeExitHandler) {
      process.removeListener('beforeExit', _beforeExitHandler);
      _beforeExitHandler = undefined;
    }
    await provider?.shutdown();
    _isInitialized = false;
    _provider = undefined;
    _resetObserveState();
    // trace/context/propagation .disable() reset OTel's PROCESS-WIDE global
    // singletons, not just this provider. Without them, OTel's global
    // registration stays pinned to the now-shut-down provider: a subsequent
    // initialize()'s register() is silently rejected (registration is
    // first-write-wins), so every tracer -- new and old -- resolves to the
    // dead provider and its spans are never exported for the rest of the
    // process. But reset each slot ONLY when TraceRoot still owns THAT slot.
    // trace/context/propagation are THREE independent first-write-wins slots
    // (see the ownership predicates above), so ownership is checked per-slot,
    // not as one combined check: a host can win the `context` slot on its own
    // (context.setGlobalContextManager() directly) while TraceRoot still owns
    // `trace` and `propagation`. A single combined gate keyed only on the
    // trace-delegate check would then see "true" and call context.disable()
    // anyway, silently wiping a context manager TraceRoot never owned. Gating
    // each disable() on its own ownership predicate tears down exactly what
    // TraceRoot registered and nothing else. (A host that shares TraceRoot's
    // OWN registered instances without registering its own cannot be
    // distinguished here and will still see the reset -- an accepted
    // limitation.) _resetForTesting() below is unconditional because tests
    // always want a clean slate.
    _releaseGlobalSlots(provider);
  }
}

/** @internal */
export function _resetForTesting(): void {
  _isInitialized = false;
  _provider = undefined;
  if (_beforeExitHandler) {
    process.removeListener('beforeExit', _beforeExitHandler);
    _beforeExitHandler = undefined;
  }
  _registeredContextManager = undefined;
  _registeredPropagator = undefined;
  _resetObserveState();
  _resetGitContextCache();
  trace.disable();
  context.disable();
  propagation.disable();
}
