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

/**
 * True when `provider` is the concrete TracerProvider the OTel global proxy
 * currently delegates to — i.e. TraceRoot's registration is still the active
 * one, so it's safe to reset the process-wide global singletons on teardown.
 * Duck-typed on getDelegate() (matching pi's own provider detection) rather
 * than instanceof ProxyTracerProvider, which is defeated by a dual-copy
 * @opentelemetry/api install.
 */
function isActiveGlobalDelegate(provider: NodeTracerProvider): boolean {
  const globalProvider = trace.getTracerProvider() as { getDelegate?: () => unknown };
  const delegate =
    typeof globalProvider.getDelegate === 'function'
      ? globalProvider.getDelegate()
      : globalProvider;
  return delegate === provider;
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
    _provider.register();

    // Thread the resolved apiKey/baseUrl so a lazily-wired pi pipeline gets
    // them even when the host configured TraceRoot programmatically and never
    // set TRACEROOT_API_KEY (see wirePiCodingAgentInstrumentation()).
    wireInstrumentations(options.instrumentModules, { apiKey, baseUrl });

    _isInitialized = true;
    process.once('beforeExit', () => {
      void _provider?.forceFlush();
    });
  }

  static async flush(): Promise<void> {
    await _provider?.forceFlush();
  }

  static async shutdown(): Promise<void> {
    const provider = _provider;
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
    // process. But reset them ONLY when this provider is still the active
    // global delegate -- i.e. TraceRoot won the first-write-wins registration
    // for all three at initialize() time. If a different OTel consumer owns
    // the global (it registered before us, so our own register() was
    // rejected), disabling here would silently wipe ITS context/propagation
    // too; leave the globals alone and tear down only our own provider. (A
    // host that shares TraceRoot's OWN provider without registering its own
    // cannot be distinguished here and will still see the reset -- an accepted
    // limitation.) Mirrors _resetForTesting()'s cleanup below, which is
    // unconditional because tests always want a clean slate.
    if (provider && isActiveGlobalDelegate(provider)) {
      trace.disable();
      context.disable();
      propagation.disable();
    }
  }
}

/** @internal */
export function _resetForTesting(): void {
  _isInitialized = false;
  _provider = undefined;
  _resetObserveState();
  _resetGitContextCache();
  trace.disable();
  context.disable();
  propagation.disable();
}
