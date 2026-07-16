// src/reresolving-tracer.ts — a tracer handle that re-resolves the global
// provider on every span-open, shared by the in-tree integrations
// (claude-agent-sdk, pi).
//
// trace.disable() (called by TraceRoot.shutdown()) swaps the OTel API's
// internal ProxyTracerProvider for a brand-new instance rather than mutating
// the old one, so a Tracer captured once at wrap time (via a single
// trace.getTracer() call) stays permanently bound to the old, now-detached
// provider -- after a shutdown()/initialize() cycle every span it opens goes
// silently dark. An integration's wrap-once guard also means it is only ever
// wrapped once, so there is no later re-wrap to pick up a fresh tracer.
// Re-resolving through the global `trace` facade on every span-open instead
// mirrors ProxyTracer's own lazy-delegate-rebind pattern one level up:
// whatever TracerProvider is globally active at the moment a span is opened is
// the one that span routes to. In steady state (no disable() ever called) this
// behaves identically to a tracer captured once.
import { trace } from '@opentelemetry/api';
import type { Context, Span, SpanOptions, Tracer } from '@opentelemetry/api';

// Callers only ever call startSpan on the returned handle, so this narrows to
// just that method rather than carrying the full Tracer surface.
export type SpanFactory = Pick<Tracer, 'startSpan'>;

export function createReresolvingTracer(name: string, version?: string): SpanFactory {
  const resolve = (): Tracer => trace.getTracer(name, version);
  return {
    startSpan(spanName: string, options?: SpanOptions, ctx?: Context): Span {
      return resolve().startSpan(spanName, options, ctx);
    },
  };
}
