import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { SpanType, TracingEventType } from '@mastra/core/observability';
import type { AnyExportedSpan } from '@mastra/core/observability';
import type { SpanExporter, ReadableSpan } from '@opentelemetry/sdk-trace-base';
import { TraceRootExporter } from '../src/exporter';

// ---------------------------------------------------------------------------
// Capturing exporter — injected via _spanExporter to avoid OTLP network calls
// ---------------------------------------------------------------------------

class CapturingExporter implements SpanExporter {
  readonly spans: ReadableSpan[] = [];
  export(spans: ReadableSpan[], cb: (result: { code: number }) => void): void {
    this.spans.push(...spans);
    cb({ code: 0 });
  }
  async shutdown(): Promise<void> {}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

let _seq = 0;
function uid(): string {
  return (++_seq).toString(16).padStart(16, '0');
}

function makeRig() {
  const capture = new CapturingExporter();
  const exporter = new TraceRootExporter({
    apiKey: 'test-key',
    disableBatch: true,
    _spanExporter: capture,
  });
  // init sets up resource/scope; falls back to defaults if skipped, but call it for correctness
  exporter.init({ config: { serviceName: 'test' } } as Parameters<typeof exporter.init>[0]);
  return { exporter, capture };
}

function makeSpan(opts: {
  id?: string;
  traceId?: string;
  parentSpanId?: string;
  name?: string;
  type?: SpanType;
  isEvent?: boolean;
}): AnyExportedSpan {
  const now = new Date();
  return {
    id: opts.id ?? uid(),
    traceId: opts.traceId ?? uid(),
    parentSpanId: opts.parentSpanId,
    name: opts.name ?? 'span',
    type: opts.type ?? SpanType.AGENT_RUN,
    isEvent: opts.isEvent ?? false,
    isRootSpan: !opts.parentSpanId,
    startTime: now,
    endTime: new Date(now.getTime() + 10),
    input: undefined,
    output: undefined,
    metadata: undefined,
    errorInfo: undefined,
    attributes: undefined,
    requestContext: undefined,
  } as unknown as AnyExportedSpan;
}

async function startSpan(exporter: TraceRootExporter, span: AnyExportedSpan): Promise<void> {
  await (
    exporter as unknown as { _exportTracingEvent(e: unknown): Promise<void> }
  )._exportTracingEvent({
    type: TracingEventType.SPAN_STARTED,
    exportedSpan: span,
  });
}

async function endSpan(exporter: TraceRootExporter, span: AnyExportedSpan): Promise<void> {
  await (
    exporter as unknown as { _exportTracingEvent(e: unknown): Promise<void> }
  )._exportTracingEvent({
    type: TracingEventType.SPAN_ENDED,
    exportedSpan: span,
  });
}

async function runSpan(exporter: TraceRootExporter, span: AnyExportedSpan): Promise<void> {
  await startSpan(exporter, span);
  await endSpan(exporter, span);
}

function getAttrs(capture: CapturingExporter, index?: number): Record<string, unknown> {
  const i = index ?? capture.spans.length - 1;
  return (capture.spans[i]?.attributes ?? {}) as Record<string, unknown>;
}

function pathMaps(exporter: TraceRootExporter) {
  const e = exporter as unknown as {
    _namePathBySpanId: Map<string, string[]>;
    _idsPathBySpanId: Map<string, string[]>;
  };
  return { name: e._namePathBySpanId, ids: e._idsPathBySpanId };
}

// ---------------------------------------------------------------------------
// 1. Path computation
// ---------------------------------------------------------------------------

describe('path computation', () => {
  it('root span: path=[name], ids=[]', async () => {
    const { exporter, capture } = makeRig();
    const traceId = uid();
    const root = makeSpan({ traceId, name: 'agent-run' });
    await runSpan(exporter, root);
    const attrs = getAttrs(capture);
    assert.deepEqual(attrs['traceroot.span.path'], ['agent-run']);
    assert.deepEqual(attrs['traceroot.span.ids_path'], []);
  });

  it('child span: path=[root,child], ids=[root.id]', async () => {
    const { exporter, capture } = makeRig();
    const traceId = uid();
    const root = makeSpan({ traceId, name: 'agent-run' });
    const child = makeSpan({ traceId, parentSpanId: root.id, name: 'model-gen' });
    await startSpan(exporter, root);
    await startSpan(exporter, child);
    await endSpan(exporter, child);
    const attrs = getAttrs(capture);
    assert.deepEqual(attrs['traceroot.span.path'], ['agent-run', 'model-gen']);
    assert.deepEqual(attrs['traceroot.span.ids_path'], [root.id]);
  });

  it('grandchild span: 3-level path and 2-level ids', async () => {
    const { exporter, capture } = makeRig();
    const traceId = uid();
    const root = makeSpan({ traceId, name: 'agent-run' });
    const child = makeSpan({ traceId, parentSpanId: root.id, name: 'model-gen' });
    const grandchild = makeSpan({ traceId, parentSpanId: child.id, name: 'model-step' });
    await startSpan(exporter, root);
    await startSpan(exporter, child);
    await startSpan(exporter, grandchild);
    await endSpan(exporter, grandchild);
    const attrs = getAttrs(capture);
    assert.deepEqual(attrs['traceroot.span.path'], ['agent-run', 'model-gen', 'model-step']);
    assert.deepEqual(attrs['traceroot.span.ids_path'], [root.id, child.id]);
  });

  it('siblings get independent paths', async () => {
    const { exporter, capture } = makeRig();
    const traceId = uid();
    const root = makeSpan({ traceId, name: 'agent-run' });
    const s1 = makeSpan({ traceId, parentSpanId: root.id, name: 'step-1' });
    const s2 = makeSpan({ traceId, parentSpanId: root.id, name: 'step-2' });
    await startSpan(exporter, root);
    await startSpan(exporter, s1);
    await startSpan(exporter, s2);
    await endSpan(exporter, s1);
    await endSpan(exporter, s2);
    assert.deepEqual(getAttrs(capture, 0)['traceroot.span.path'], ['agent-run', 'step-1']);
    assert.deepEqual(getAttrs(capture, 1)['traceroot.span.path'], ['agent-run', 'step-2']);
    assert.deepEqual(getAttrs(capture, 0)['traceroot.span.ids_path'], [root.id]);
    assert.deepEqual(getAttrs(capture, 1)['traceroot.span.ids_path'], [root.id]);
  });

  it('multi-child tree: each child gets correct parent id in ids_path', async () => {
    const { exporter, capture } = makeRig();
    const traceId = uid();
    const root = makeSpan({ traceId, name: 'root' });
    const c1 = makeSpan({ traceId, parentSpanId: root.id, name: 'c1' });
    const c2 = makeSpan({ traceId, parentSpanId: root.id, name: 'c2' });
    const gc = makeSpan({ traceId, parentSpanId: c1.id, name: 'gc' });
    await startSpan(exporter, root);
    await startSpan(exporter, c1);
    await startSpan(exporter, c2);
    await startSpan(exporter, gc);
    await endSpan(exporter, gc);
    await endSpan(exporter, c1);
    await endSpan(exporter, c2);
    const gcAttrs = getAttrs(capture, 0);
    assert.deepEqual(gcAttrs['traceroot.span.path'], ['root', 'c1', 'gc']);
    assert.deepEqual(gcAttrs['traceroot.span.ids_path'], [root.id, c1.id]);
  });
});

// ---------------------------------------------------------------------------
// 1b. ids_path normalization parity with OTLP spanIds
// ---------------------------------------------------------------------------
// The backend joins a child's traceroot.span.ids_path entries against the
// parent's exported OTLP spanId. convertToOtelSpan normalizes the exported
// spanId via normalizeHex(_, 16), so ids_path entries must use the same
// normalization or ancestry joins break for any non-canonical upstream ID.

describe('ids_path normalization parity with OTLP spanIds', () => {
  it('non-canonical parent id: ids_path entry matches normalized OTLP spanId', async () => {
    const { exporter, capture } = makeRig();
    const traceId = uid();
    // Mixed case + 0x prefix + short — exercises every transformation normalizeHex applies.
    const root = makeSpan({ traceId, id: '0xABcD', name: 'root' });
    const child = makeSpan({ traceId, parentSpanId: root.id, name: 'child' });

    await startSpan(exporter, root);
    await startSpan(exporter, child);
    await endSpan(exporter, child);
    await endSpan(exporter, root);

    const findByPath = (path: string[]) =>
      capture.spans.findIndex(
        (s) =>
          (s.attributes as Record<string, unknown>)['traceroot.span.path']?.toString() ===
          path.toString(),
      );
    const childIdx = findByPath(['root', 'child']);
    const rootIdx = findByPath(['root']);
    assert.notEqual(childIdx, -1, 'child span exported');
    assert.notEqual(rootIdx, -1, 'root span exported');

    const exportedRootSpanId = capture.spans[rootIdx].spanContext().spanId;
    const childIdsPath = (capture.spans[childIdx].attributes as Record<string, unknown>)[
      'traceroot.span.ids_path'
    ] as string[];

    assert.match(exportedRootSpanId, /^[0-9a-f]{16}$/, 'exported spanId is canonical');
    assert.deepEqual(
      childIdsPath,
      [exportedRootSpanId],
      'child ids_path[0] joins against parent exported spanId',
    );
  });

  it('canonical parent id: ids_path unchanged (no regression)', async () => {
    const { exporter, capture } = makeRig();
    const traceId = uid();
    const root = makeSpan({ traceId, name: 'root' }); // uid() already canonical
    const child = makeSpan({ traceId, parentSpanId: root.id, name: 'child' });

    await startSpan(exporter, root);
    await startSpan(exporter, child);
    await endSpan(exporter, child);

    const attrs = getAttrs(capture);
    assert.deepEqual(attrs['traceroot.span.ids_path'], [root.id]);
  });
});

// ---------------------------------------------------------------------------
// 2. isEvent guard: SPAN_STARTED for event spans is a no-op
// ---------------------------------------------------------------------------

describe('isEvent guard', () => {
  it('event span started: not added to path maps', async () => {
    const { exporter } = makeRig();
    const traceId = uid();
    const evt = makeSpan({ traceId, name: 'chunk', isEvent: true });
    await startSpan(exporter, evt);
    const maps = pathMaps(exporter);
    assert.equal(maps.name.has(evt.id), false);
    assert.equal(maps.ids.has(evt.id), false);
  });

  it('event span ended: no path attributes on exported span', async () => {
    const { exporter, capture } = makeRig();
    const traceId = uid();
    const evt = makeSpan({ traceId, name: 'chunk', isEvent: true });
    await startSpan(exporter, evt);
    await endSpan(exporter, evt);
    const attrs = getAttrs(capture);
    assert.equal(attrs['traceroot.span.path'], undefined);
    assert.equal(attrs['traceroot.span.ids_path'], undefined);
  });

  it('event child under tracked parent does not pollute parent path map', async () => {
    const { exporter } = makeRig();
    const traceId = uid();
    const root = makeSpan({ traceId, name: 'agent-run' });
    const evt = makeSpan({ traceId, parentSpanId: root.id, name: 'chunk', isEvent: true });
    await startSpan(exporter, root);
    await startSpan(exporter, evt);
    const maps = pathMaps(exporter);
    assert.equal(maps.name.has(evt.id), false);
    assert.deepEqual(maps.name.get(root.id), ['agent-run']);
  });
});

// ---------------------------------------------------------------------------
// 3. Orphaned parentSpanId: unknown parent → treat span as root
// ---------------------------------------------------------------------------

describe('orphaned parentSpanId guard', () => {
  it('span referencing unknown parent gets root-level path', async () => {
    const { exporter, capture } = makeRig();
    const traceId = uid();
    const orphan = makeSpan({ traceId, parentSpanId: uid(), name: 'orphan' });
    await runSpan(exporter, orphan);
    const attrs = getAttrs(capture);
    assert.deepEqual(attrs['traceroot.span.path'], ['orphan']);
    assert.deepEqual(attrs['traceroot.span.ids_path'], []);
  });

  it('child of orphan inherits orphan as parent correctly', async () => {
    const { exporter, capture } = makeRig();
    const traceId = uid();
    const orphan = makeSpan({ traceId, parentSpanId: uid(), name: 'orphan' });
    const child = makeSpan({ traceId, parentSpanId: orphan.id, name: 'child-of-orphan' });
    await startSpan(exporter, orphan);
    await startSpan(exporter, child);
    await endSpan(exporter, child);
    const attrs = getAttrs(capture);
    assert.deepEqual(attrs['traceroot.span.path'], ['orphan', 'child-of-orphan']);
    assert.deepEqual(attrs['traceroot.span.ids_path'], [orphan.id]);
  });
});

// ---------------------------------------------------------------------------
// 4. Memory cleanup
// ---------------------------------------------------------------------------

describe('memory cleanup', () => {
  it('both maps are empty after span ends', async () => {
    const { exporter } = makeRig();
    const traceId = uid();
    const root = makeSpan({ traceId, name: 'root' });
    await runSpan(exporter, root);
    const maps = pathMaps(exporter);
    assert.equal(maps.name.size, 0);
    assert.equal(maps.ids.size, 0);
  });

  it('shutdown clears maps even for in-flight spans', async () => {
    const { exporter } = makeRig();
    const traceId = uid();
    const root = makeSpan({ traceId, name: 'root' });
    await startSpan(exporter, root); // started, not ended
    const maps = pathMaps(exporter);
    assert.equal(maps.name.size, 1, 'map should have in-flight entry');
    await exporter.shutdown();
    assert.equal(maps.name.size, 0);
    assert.equal(maps.ids.size, 0);
  });

  it('maps stay at size 0 after 20 sequential traces', async () => {
    const { exporter } = makeRig();
    for (let i = 0; i < 20; i++) {
      const traceId = uid();
      const root = makeSpan({ traceId, name: `root-${i}` });
      const child = makeSpan({ traceId, parentSpanId: root.id, name: `child-${i}` });
      await startSpan(exporter, root);
      await runSpan(exporter, child);
      await endSpan(exporter, root);
    }
    const maps = pathMaps(exporter);
    assert.equal(maps.name.size, 0);
    assert.equal(maps.ids.size, 0);
  });
});

// ---------------------------------------------------------------------------
// 5. Concurrent trace isolation
// ---------------------------------------------------------------------------

describe('concurrent trace isolation', () => {
  it('two interleaved traces do not share path state', async () => {
    const { exporter, capture } = makeRig();
    const traceA = uid();
    const traceB = uid();
    const rootA = makeSpan({ traceId: traceA, name: 'root-a' });
    const rootB = makeSpan({ traceId: traceB, name: 'root-b' });
    const childA = makeSpan({ traceId: traceA, parentSpanId: rootA.id, name: 'child-a' });
    const childB = makeSpan({ traceId: traceB, parentSpanId: rootB.id, name: 'child-b' });
    await startSpan(exporter, rootA);
    await startSpan(exporter, rootB);
    await startSpan(exporter, childA);
    await startSpan(exporter, childB);
    await endSpan(exporter, childA);
    await endSpan(exporter, childB);
    const attrsA = getAttrs(capture, 0);
    const attrsB = getAttrs(capture, 1);
    assert.deepEqual(attrsA['traceroot.span.path'], ['root-a', 'child-a']);
    assert.deepEqual(attrsB['traceroot.span.path'], ['root-b', 'child-b']);
    assert.deepEqual(attrsA['traceroot.span.ids_path'], [rootA.id]);
    assert.deepEqual(attrsB['traceroot.span.ids_path'], [rootB.id]);
  });

  it('three concurrent traces remain isolated', async () => {
    const { exporter, capture } = makeRig();
    const traces = [uid(), uid(), uid()];
    const roots = traces.map((traceId, i) => makeSpan({ traceId, name: `root-${i}` }));
    const children = roots.map((root, i) =>
      makeSpan({ traceId: traces[i], parentSpanId: root.id, name: `child-${i}` }),
    );
    for (const root of roots) await startSpan(exporter, root);
    for (const child of children) await startSpan(exporter, child);
    for (const child of children) await endSpan(exporter, child);
    for (let i = 0; i < 3; i++) {
      const attrs = getAttrs(capture, i);
      assert.deepEqual(attrs['traceroot.span.path'], [`root-${i}`, `child-${i}`]);
      assert.deepEqual(attrs['traceroot.span.ids_path'], [roots[i].id]);
    }
  });
});

// ---------------------------------------------------------------------------
// 6. Deep nesting
// ---------------------------------------------------------------------------

describe('deep nesting', () => {
  it('10-level hierarchy: leaf gets full path and ids', async () => {
    const { exporter, capture } = makeRig();
    const traceId = uid();
    const spans: AnyExportedSpan[] = [];
    for (let i = 0; i < 10; i++) {
      spans.push(makeSpan({ traceId, parentSpanId: spans[i - 1]?.id, name: `level-${i}` }));
    }
    for (const span of spans) await startSpan(exporter, span);
    await endSpan(exporter, spans[9]);
    const attrs = getAttrs(capture);
    assert.deepEqual(
      attrs['traceroot.span.path'],
      spans.map((s) => s.name),
    );
    assert.deepEqual(
      attrs['traceroot.span.ids_path'],
      spans.slice(0, -1).map((s) => s.id),
    );
  });

  it('10-level hierarchy: each intermediate span has correct partial path', async () => {
    const { exporter, capture } = makeRig();
    const traceId = uid();
    const spans: AnyExportedSpan[] = [];
    for (let i = 0; i < 10; i++) {
      spans.push(makeSpan({ traceId, parentSpanId: spans[i - 1]?.id, name: `L${i}` }));
    }
    for (const span of spans) await startSpan(exporter, span);
    // End in reverse order (deepest first); capture index i corresponds to spans[9-i]
    for (let i = 9; i >= 0; i--) await endSpan(exporter, spans[i]);
    for (let captureIdx = 0; captureIdx < 10; captureIdx++) {
      const spanDepth = 10 - captureIdx; // deepest span ended first → longest path first
      const attrs = getAttrs(capture, captureIdx);
      assert.deepEqual(
        attrs['traceroot.span.path'],
        spans.slice(0, spanDepth).map((s) => s.name),
      );
    }
  });
});

// ---------------------------------------------------------------------------
// 7. Real Mastra span tree: AGENT_RUN → MODEL_GENERATION → MODEL_STEP
// ---------------------------------------------------------------------------

describe('Mastra span tree (AGENT_RUN → MODEL_GENERATION → MODEL_STEP)', () => {
  it('backend anchor: path[0] is always the root agent name', async () => {
    const { exporter, capture } = makeRig();
    const traceId = uid();
    const agentRun = makeSpan({ traceId, name: 'myAgent', type: SpanType.AGENT_RUN });
    const modelGen = makeSpan({
      traceId,
      parentSpanId: agentRun.id,
      name: 'model-generation',
      type: SpanType.MODEL_GENERATION,
    });
    const modelStep = makeSpan({
      traceId,
      parentSpanId: modelGen.id,
      name: 'model-step',
      type: SpanType.MODEL_STEP,
    });
    await startSpan(exporter, agentRun);
    await startSpan(exporter, modelGen);
    await startSpan(exporter, modelStep);
    await endSpan(exporter, modelStep);
    const attrs = getAttrs(capture);
    const path = attrs['traceroot.span.path'] as string[];
    assert.equal(path[0], 'myAgent', 'path[0] must be root span name (backend streaming anchor)');
    assert.deepEqual(path, ['myAgent', 'model-generation', 'model-step']);
    assert.deepEqual(attrs['traceroot.span.ids_path'], [agentRun.id, modelGen.id]);
  });

  it('MODEL_GENERATION emitted while AGENT_RUN still active: path is correct', async () => {
    const { exporter, capture } = makeRig();
    const traceId = uid();
    const agentRun = makeSpan({ traceId, name: 'myAgent', type: SpanType.AGENT_RUN });
    const modelGen = makeSpan({
      traceId,
      parentSpanId: agentRun.id,
      name: 'model-gen',
      type: SpanType.MODEL_GENERATION,
    });
    await startSpan(exporter, agentRun);
    await runSpan(exporter, modelGen); // ends before agentRun ends
    const attrs = getAttrs(capture, 0);
    assert.deepEqual(attrs['traceroot.span.path'], ['myAgent', 'model-gen']);
    assert.deepEqual(attrs['traceroot.span.ids_path'], [agentRun.id]);
  });

  it('multi-step agent: each MODEL_GENERATION child gets correct per-step path', async () => {
    const { exporter, capture } = makeRig();
    const traceId = uid();
    const agentRun = makeSpan({ traceId, name: 'myAgent', type: SpanType.AGENT_RUN });
    const gen1 = makeSpan({
      traceId,
      parentSpanId: agentRun.id,
      name: 'gen-1',
      type: SpanType.MODEL_GENERATION,
    });
    const gen2 = makeSpan({
      traceId,
      parentSpanId: agentRun.id,
      name: 'gen-2',
      type: SpanType.MODEL_GENERATION,
    });
    await startSpan(exporter, agentRun);
    await runSpan(exporter, gen1);
    await runSpan(exporter, gen2);
    assert.deepEqual(getAttrs(capture, 0)['traceroot.span.path'], ['myAgent', 'gen-1']);
    assert.deepEqual(getAttrs(capture, 1)['traceroot.span.path'], ['myAgent', 'gen-2']);
  });
});

// ---------------------------------------------------------------------------
// 8. Attribute co-existence: path attrs do not overwrite other attrs
// ---------------------------------------------------------------------------

describe('attribute co-existence', () => {
  it('openinference.span.kind is present alongside path attrs', async () => {
    const { exporter, capture } = makeRig();
    const traceId = uid();
    const root = makeSpan({ traceId, name: 'myAgent', type: SpanType.AGENT_RUN });
    await runSpan(exporter, root);
    const attrs = getAttrs(capture);
    assert.equal(attrs['openinference.span.kind'], 'AGENT');
    assert.ok(Array.isArray(attrs['traceroot.span.path']));
    assert.ok(Array.isArray(attrs['traceroot.span.ids_path']));
  });

  it('traceroot.sdk.name is present alongside path attrs', async () => {
    const { exporter, capture } = makeRig();
    const traceId = uid();
    const root = makeSpan({ traceId, name: 'root' });
    await runSpan(exporter, root);
    const attrs = getAttrs(capture);
    assert.equal(attrs['traceroot.sdk.name'], 'traceroot-mastra');
    assert.ok(Array.isArray(attrs['traceroot.span.path']));
    assert.ok(Array.isArray(attrs['traceroot.span.ids_path']));
  });

  it('LLM span: gen_ai attrs and path attrs co-exist', async () => {
    const { exporter, capture } = makeRig();
    const traceId = uid();
    const root = makeSpan({ traceId, name: 'agent-run', type: SpanType.AGENT_RUN });
    const llm = {
      ...makeSpan({
        traceId,
        parentSpanId: root.id,
        name: 'llm-call',
        type: SpanType.MODEL_GENERATION,
      }),
      attributes: {
        provider: 'openai.chat',
        model: 'gpt-4o-mini',
        usage: { inputTokens: 10, outputTokens: 5 },
      },
    } as unknown as AnyExportedSpan;
    await startSpan(exporter, root);
    await runSpan(exporter, llm);
    const attrs = getAttrs(capture, 0);
    assert.equal(attrs['gen_ai.system'], 'openai');
    assert.equal(attrs['gen_ai.request.model'], 'gpt-4o-mini');
    assert.equal(attrs['gen_ai.usage.input_tokens'], 10);
    assert.equal(attrs['gen_ai.usage.output_tokens'], 5);
    assert.deepEqual(attrs['traceroot.span.path'], ['agent-run', 'llm-call']);
    assert.deepEqual(attrs['traceroot.span.ids_path'], [root.id]);
  });

  it('span with metadata: traceroot.metadata.* and path attrs co-exist', async () => {
    const { exporter, capture } = makeRig();
    const traceId = uid();
    const root = {
      ...makeSpan({ traceId, name: 'root' }),
      metadata: { sessionId: 'sess-1', userId: 'user-1', env: 'prod' },
    } as unknown as AnyExportedSpan;
    await runSpan(exporter, root);
    const attrs = getAttrs(capture);
    assert.equal(attrs['session.id'], 'sess-1');
    assert.equal(attrs['user.id'], 'user-1');
    assert.equal(attrs['traceroot.metadata.env'], 'prod');
    assert.deepEqual(attrs['traceroot.span.path'], ['root']);
    assert.deepEqual(attrs['traceroot.span.ids_path'], []);
  });
});
