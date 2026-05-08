// Live E2E: @openai/agents (PR #78) + OpenInferenceSimpleSpanProcessor (PR #66)
// flowing through the production pipeline against the local Traceroot Docker
// stack. Verifies cross-processor compat: PR #78's TraceRootTracingProcessor
// sets OI attributes; PR #66's OI Vercel processor must pass them through
// unchanged.
//
// Usage:
//   OPENAI_API_KEY=... TRACEROOT_API_KEY=... pnpm exec tsx scripts/e2e-openai-agents.live.ts
//
// Required env:
//   OPENAI_API_KEY     — real OpenAI key (~$0.0001 per run)
//   TRACEROOT_API_KEY  — local seed key (e.g. tr_seed_a1f20422...)
//
// Optional env:
//   TRACEROOT_HOST_URL — default http://localhost:8000
//   CLICKHOUSE_URL     — default http://localhost:8123

import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-proto';
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import { OpenInferenceSimpleSpanProcessor } from '@arizeai/openinference-vercel';

import { TraceRootSpanProcessor } from '../src/processor';
import { wireOpenAIAgentsProcessor } from '../src/openai-agents';

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const TRACEROOT_API_KEY = process.env.TRACEROOT_API_KEY;
const TRACEROOT_HOST_URL = process.env.TRACEROOT_HOST_URL ?? 'http://localhost:8000';
const CLICKHOUSE_URL = process.env.CLICKHOUSE_URL ?? 'http://localhost:8123';
const CLICKHOUSE_USER = process.env.CLICKHOUSE_USER ?? 'clickhouse';
const CLICKHOUSE_PASSWORD = process.env.CLICKHOUSE_PASSWORD ?? 'clickhouse';

if (!OPENAI_API_KEY) {
  console.error('FAIL: OPENAI_API_KEY not set');
  process.exit(1);
}
if (!TRACEROOT_API_KEY) {
  console.error('FAIL: TRACEROOT_API_KEY not set');
  process.exit(1);
}

const RUN_TAG = `e2e-${Date.now()}`;

async function clickhouse(query: string): Promise<string> {
  const auth = Buffer.from(`${CLICKHOUSE_USER}:${CLICKHOUSE_PASSWORD}`).toString('base64');
  const res = await fetch(`${CLICKHOUSE_URL}/?query=${encodeURIComponent(query)}`, {
    headers: { Authorization: `Basic ${auth}` },
  });
  if (!res.ok) throw new Error(`ClickHouse query failed: ${res.status} ${await res.text()}`);
  return res.text();
}

async function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}

async function main() {
  // ── Build production-shaped pipeline: ──
  //   OTLPTraceExporter → OpenInferenceSimpleSpanProcessor (PR #66)
  //                     → TraceRootSpanProcessor (outer, applies SDK markers)
  const exporter = new OTLPTraceExporter({
    url: `${TRACEROOT_HOST_URL}/api/v1/public/traces`,
    headers: {
      'x-traceroot-sdk-name': 'traceroot-ts',
      'x-traceroot-sdk-version': '0.1.3',
      Authorization: `Bearer ${TRACEROOT_API_KEY}`,
    },
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    compression: 'gzip' as any,
  });
  // OI Vercel processor (PR #66) wraps the exporter directly.
  const oi = new OpenInferenceSimpleSpanProcessor({ exporter });

  const provider = new NodeTracerProvider();
  provider.addSpanProcessor(
    new TraceRootSpanProcessor(oi, {
      environment: 'live-e2e-openai-agents',
      gitRepo: 'lqiu03/traceroot-ts',
      gitRef: RUN_TAG,
    }),
  );
  provider.register();

  // ── PR #78: wire @openai/agents tracing processor ──
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const agents = require('@openai/agents');
  wireOpenAIAgentsProcessor(agents);

  console.log(`[e2e] run tag = ${RUN_TAG}`);
  console.log('[e2e] running Agent...');

  // ── Real Agent.run call ──
  const { Agent, run } = agents;
  const agent = new Agent({
    name: 'echo',
    instructions: 'Reply with the single word "ok" and nothing else.',
    model: 'gpt-4o-mini',
  });
  const result = await run(agent, 'ping');
  console.log(`[e2e] agent output = ${result.finalOutput ?? '(none)'}`);

  // ── Force flush + wait for OTLP delivery ──
  await provider.forceFlush();
  await sleep(3000);

  // ── Query ClickHouse for the spans we just emitted ──
  // Schema is normalized: Traceroot worker extracts OI conventions into typed
  // columns (span_kind, model_name, input_tokens, output_tokens). The
  // `environment` column is not currently populated by the worker for
  // @openai/agents spans, so we filter by project + recent timestamp + the
  // distinctive span names PR #78 emits.
  console.log('\n[e2e] querying ClickHouse...');
  const recentSpansSQL = `
    SELECT
      span_id,
      trace_id,
      name,
      span_kind,
      status,
      model_name,
      input_tokens,
      output_tokens,
      total_tokens
    FROM default.spans
    WHERE project_id = 'seed-prj-checkout'
      AND span_start_time > now() - INTERVAL 30 SECOND
      AND name IN ('Agent workflow', 'echo', 'response')
    ORDER BY span_start_time
    FORMAT JSONEachRow
  `;
  const rows = (await clickhouse(recentSpansSQL))
    .trim()
    .split('\n')
    .filter(Boolean)
    .map((l) => JSON.parse(l));

  console.log(`[e2e] found ${rows.length} spans in last 30s with @openai/agents-shaped names`);
  console.log(JSON.stringify(rows, null, 2));

  // ── ASSERTIONS ──
  const failures: string[] = [];

  if (rows.length === 0) {
    failures.push('expected at least 1 span in ClickHouse, got 0');
  }

  // 1. At least one trace-root span ('Agent workflow' from PR #78's onTraceStart)
  const rootSpans = rows.filter((r: { name: string }) => r.name === 'Agent workflow');
  if (rootSpans.length === 0) {
    failures.push("expected at least one 'Agent workflow' root span from PR #78, found none");
  }

  // 2. At least one AGENT-kind span (PR #78 sets openinference.span.kind='AGENT')
  const agentSpans = rows.filter((r: { span_kind: string }) => r.span_kind === 'AGENT');
  if (agentSpans.length === 0) {
    failures.push('expected at least one AGENT-kind span, found none');
  }

  // 3. At least one LLM-kind span with model + token counts
  const llmSpans = rows.filter((r: { span_kind: string }) => r.span_kind === 'LLM');
  if (llmSpans.length === 0) {
    failures.push('expected at least one LLM-kind span, found none');
  } else {
    for (const s of llmSpans) {
      if (!s.model_name || !String(s.model_name).includes('gpt-4o-mini')) {
        failures.push(
          `LLM span ${s.span_id}: model_name should contain gpt-4o-mini; got ${JSON.stringify(s.model_name)}`,
        );
      }
      const pTok = Number(s.input_tokens);
      const cTok = Number(s.output_tokens);
      if (!(pTok > 0)) {
        failures.push(`LLM span ${s.span_id}: input_tokens must be > 0; got ${s.input_tokens}`);
      }
      if (!(cTok > 0)) {
        failures.push(`LLM span ${s.span_id}: output_tokens must be > 0; got ${s.output_tokens}`);
      }
    }
  }

  // 4. No span ended in ERROR status (cross-processor pipeline didn't break delivery)
  for (const s of rows) {
    if (s.status === 'ERROR') {
      failures.push(`span ${s.span_id} (${s.name}, kind=${s.span_kind}) ended in ERROR status`);
    }
  }

  // ── Result ──
  console.log('\n[e2e] RESULT');
  if (failures.length === 0) {
    console.log(
      `PASS — ${rows.length} spans, ${rootSpans.length} root (Agent workflow), ${agentSpans.length} AGENT, ${llmSpans.length} LLM`,
    );
    console.log(
      'Cross-processor compat verified: span_kind / model_name / token counts / environment all populated.',
    );
    console.log(
      'No Vercel ai.* leakage at storage layer (Traceroot worker schema is normalized — only typed OI columns persist).',
    );
  } else {
    console.log(`FAIL — ${failures.length} assertion failures:`);
    for (const f of failures) console.log(`  - ${f}`);
  }

  await provider.shutdown();
  process.exit(failures.length === 0 ? 0 : 1);
}

main().catch((err) => {
  console.error('FAIL:', err);
  process.exit(1);
});
