/**
 * Adversarial tests — edge cases, robustness, and regression guards.
 * Each test is designed to reveal a specific failure mode in path tracking.
 */
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { SpanType, TracingEventType } from '@mastra/core/observability';
import type { AnyExportedSpan } from '@mastra/core/observability';
import type { SpanExporter, ReadableSpan } from '@opentelemetry/sdk-trace-base';
import { TraceRootExporter } from '../src/exporter';

// ---------------------------------------------------------------------------
// Shared test infrastructure (copied locally — no shared state across files)
// ---------------------------------------------------------------------------

class CapturingExporter implements SpanExporter {
  readonly spans: ReadableSpan[] = [];
  export(spans: ReadableSpan[], cb: (result: { code: number }) => void): void {
    this.spans.push(...spans);
    cb({ code: 0 });
  }
  async shutdown(): Promise<void> {}
}

let _seq = 1000; // distinct range from the other test file
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
  exporter.init({ config: { serviceName: 'adversarial' } } as Parameters<typeof exporter.init>[0]);
  return { exporter, capture };
}

function makeSpan(opts: {
  id?: string;
  traceId?: string;
  parentSpanId?: string;
  name?: string;
  type?: SpanType;
  isEvent?: boolean;
  errorInfo?: { message: string; details?: { stack?: string } };
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
    errorInfo: opts.errorInfo,
    attributes: undefined,
    requestContext: undefined,
  } as unknown as AnyExportedSpan;
}

type Exporter = TraceRootExporter;

async function emit(
  exporter: Exporter,
  type: TracingEventType,
  span: AnyExportedSpan,
): Promise<void> {
  await (
    exporter as unknown as { _exportTracingEvent(e: unknown): Promise<void> }
  )._exportTracingEvent({ type, exportedSpan: span });
}

const start = (e: Exporter, s: AnyExportedSpan) => emit(e, TracingEventType.SPAN_STARTED, s);
const end = (e: Exporter, s: AnyExportedSpan) => emit(e, TracingEventType.SPAN_ENDED, s);
const update = (e: Exporter, s: AnyExportedSpan) => emit(e, TracingEventType.SPAN_UPDATED, s);
const run = async (e: Exporter, s: AnyExportedSpan) => {
  await start(e, s);
  await end(e, s);
};

function getAttrs(capture: CapturingExporter, index?: number): Record<string, unknown> {
  const i = index ?? capture.spans.length - 1;
  return (capture.spans[i]?.attributes ?? {}) as Record<string, unknown>;
}

function pathMaps(exporter: TraceRootExporter) {
  const e = exporter as unknown as {
    _namePathBySpanId: Map<string, string[]>;
    _idsPathBySpanId: Map<string, string[]>;
    traceMap: Map<string, { activeSpanIds: Set<string> }>;
  };
  return { name: e._namePathBySpanId, ids: e._idsPathBySpanId, trace: e.traceMap };
}

// ---------------------------------------------------------------------------
// 1. Out-of-order ending: parent ends BEFORE child ends
// ---------------------------------------------------------------------------

describe('out-of-order ending (parent ends before child)', () => {
  it('child path is still correct when parent ends first', async () => {
    const { exporter, capture } = makeRig();
    const traceId = uid();
    const parent = makeSpan({ traceId, name: 'parent' });
    const child = makeSpan({ traceId, parentSpanId: parent.id, name: 'child' });

    await start(exporter, parent);
    await start(exporter, child);
    await end(exporter, parent); // <-- parent ends first
    await end(exporter, child); // <-- child ends after parent

    // parent exported first
    assert.deepEqual(getAttrs(capture, 0)['traceroot.span.path'], ['parent']);
    assert.deepEqual(getAttrs(capture, 0)['traceroot.span.ids_path'], []);
    // child path was computed at start time — must survive parent end
    assert.deepEqual(getAttrs(capture, 1)['traceroot.span.path'], ['parent', 'child']);
    assert.deepEqual(getAttrs(capture, 1)['traceroot.span.ids_path'], [parent.id]);
  });

  it('grandchild path survives both parent and grandparent ending first', async () => {
    const { exporter, capture } = makeRig();
    const traceId = uid();
    const root = makeSpan({ traceId, name: 'root' });
    const mid = makeSpan({ traceId, parentSpanId: root.id, name: 'mid' });
    const leaf = makeSpan({ traceId, parentSpanId: mid.id, name: 'leaf' });

    await start(exporter, root);
    await start(exporter, mid);
    await start(exporter, leaf);
    await end(exporter, root); // root ends first
    await end(exporter, mid); // mid ends second
    await end(exporter, leaf); // leaf ends last

    const leafAttrs = getAttrs(capture, 2);
    assert.deepEqual(leafAttrs['traceroot.span.path'], ['root', 'mid', 'leaf']);
    assert.deepEqual(leafAttrs['traceroot.span.ids_path'], [root.id, mid.id]);
  });

  it('traceMap is cleaned up correctly when child outlives parent', async () => {
    const { exporter } = makeRig();
    const traceId = uid();
    const parent = makeSpan({ traceId, name: 'parent' });
    const child = makeSpan({ traceId, parentSpanId: parent.id, name: 'child' });

    await start(exporter, parent);
    await start(exporter, child);
    await end(exporter, parent);

    // traceMap should still have an entry (child is still active)
    const maps = pathMaps(exporter);
    assert.equal(maps.trace.has(traceId), true, 'traceMap should persist while child is active');

    await end(exporter, child);

    // Now traceMap should be cleaned
    assert.equal(maps.trace.has(traceId), false, 'traceMap should be cleared after all spans end');
    assert.equal(maps.name.size, 0);
    assert.equal(maps.ids.size, 0);
  });
});

// ---------------------------------------------------------------------------
// 2. SPAN_ENDED without prior SPAN_STARTED (orphaned end)
// ---------------------------------------------------------------------------

describe('SPAN_ENDED without SPAN_STARTED', () => {
  it('exports span without path attrs — no crash', async () => {
    const { exporter, capture } = makeRig();
    const span = makeSpan({ name: 'unstarted' });
    await end(exporter, span); // no prior start
    assert.equal(capture.spans.length, 1, 'span should still be exported');
    const attrs = getAttrs(capture);
    assert.equal(attrs['traceroot.span.path'], undefined);
    assert.equal(attrs['traceroot.span.ids_path'], undefined);
  });

  it('no memory leak: maps empty after orphaned end', async () => {
    const { exporter } = makeRig();
    const span = makeSpan({ name: 'unstarted' });
    await end(exporter, span);
    const maps = pathMaps(exporter);
    assert.equal(maps.name.size, 0);
    assert.equal(maps.ids.size, 0);
    assert.equal(maps.trace.size, 0);
  });

  it('orphaned end does not corrupt same-trace tracked spans', async () => {
    const { exporter, capture } = makeRig();
    const traceId = uid();
    const tracked = makeSpan({ traceId, name: 'tracked' });
    const unstarted = makeSpan({ traceId, name: 'unstarted' });

    await start(exporter, tracked);
    await end(exporter, unstarted); // orphan end in same trace
    await end(exporter, tracked); // tracked span ends

    const trackedAttrs = getAttrs(capture, 1); // tracked is second exported
    assert.deepEqual(trackedAttrs['traceroot.span.path'], ['tracked']);
    assert.deepEqual(trackedAttrs['traceroot.span.ids_path'], []);
  });
});

// ---------------------------------------------------------------------------
// 3. SPAN_UPDATED is a no-op (must not export and must not alter maps)
// ---------------------------------------------------------------------------

describe('SPAN_UPDATED no-op', () => {
  it('SPAN_UPDATED does not export the span', async () => {
    const { exporter, capture } = makeRig();
    const span = makeSpan({ name: 'updating-span' });
    await start(exporter, span);
    await update(exporter, span); // should be ignored
    assert.equal(capture.spans.length, 0, 'update must not trigger export');
  });

  it('SPAN_UPDATED does not alter path maps', async () => {
    const { exporter } = makeRig();
    const traceId = uid();
    const parent = makeSpan({ traceId, name: 'parent' });
    const child = makeSpan({ traceId, parentSpanId: parent.id, name: 'child' });

    await start(exporter, parent);
    await start(exporter, child);
    const mapsBefore = pathMaps(exporter).name.size;
    await update(exporter, parent); // should be no-op
    assert.equal(pathMaps(exporter).name.size, mapsBefore, 'update must not change map size');
  });

  it('SPAN_UPDATED for untracked span does not create map entries', async () => {
    const { exporter } = makeRig();
    const span = makeSpan({ name: 'ghost' });
    await update(exporter, span);
    const maps = pathMaps(exporter);
    assert.equal(maps.name.has(span.id), false);
    assert.equal(maps.trace.size, 0);
  });
});

// ---------------------------------------------------------------------------
// 4. Double SPAN_ENDED — second end must not crash and must not have path attrs
// ---------------------------------------------------------------------------

describe('double SPAN_ENDED', () => {
  it('second end of same span exports without path attrs', async () => {
    const { exporter, capture } = makeRig();
    const traceId = uid();
    const span = makeSpan({ traceId, name: 'double-end' });

    await start(exporter, span);
    await end(exporter, span); // first end — has path attrs
    await end(exporter, span); // second end — maps already cleared

    assert.equal(capture.spans.length, 2, 'both ends should produce an export');
    assert.deepEqual(getAttrs(capture, 0)['traceroot.span.path'], ['double-end']);
    assert.equal(getAttrs(capture, 1)['traceroot.span.path'], undefined, 'second end has no path');
  });

  it('double-end does not leave stale map entries', async () => {
    const { exporter } = makeRig();
    const span = makeSpan({ name: 'double-end' });
    await run(exporter, span);
    await end(exporter, span); // second end
    const maps = pathMaps(exporter);
    assert.equal(maps.name.size, 0);
    assert.equal(maps.ids.size, 0);
  });
});

// ---------------------------------------------------------------------------
// 5. Self-referential parentSpanId (span.parentSpanId === span.id)
// ---------------------------------------------------------------------------

describe('self-referential parentSpanId', () => {
  it('span whose parentSpanId equals its own id is treated as root', async () => {
    const { exporter, capture } = makeRig();
    const id = uid();
    const traceId = uid();
    const selfRef = makeSpan({ id, traceId, parentSpanId: id, name: 'self-ref' });
    await run(exporter, selfRef);
    const attrs = getAttrs(capture);
    // At start time, the span is not yet in the map, so parent lookup returns undefined → root
    assert.deepEqual(attrs['traceroot.span.path'], ['self-ref']);
    assert.deepEqual(attrs['traceroot.span.ids_path'], []);
  });
});

// ---------------------------------------------------------------------------
// 6. Stress: 50 sibling spans in the same trace
// ---------------------------------------------------------------------------

describe('stress: 50 siblings in same trace', () => {
  it('all 50 siblings get correct path=[root, sibling-N]', async () => {
    const { exporter, capture } = makeRig();
    const traceId = uid();
    const root = makeSpan({ traceId, name: 'root' });
    const siblings: AnyExportedSpan[] = Array.from({ length: 50 }, (_, i) =>
      makeSpan({ traceId, parentSpanId: root.id, name: `sibling-${i}` }),
    );

    await start(exporter, root);
    for (const s of siblings) await start(exporter, s);
    for (const s of siblings) await end(exporter, s);

    assert.equal(capture.spans.length, 50);
    for (let i = 0; i < 50; i++) {
      const attrs = getAttrs(capture, i);
      assert.deepEqual(attrs['traceroot.span.path'], ['root', `sibling-${i}`]);
      assert.deepEqual(attrs['traceroot.span.ids_path'], [root.id]);
    }
  });

  it('all 50 sibling map entries are cleaned after they end', async () => {
    const { exporter } = makeRig();
    const traceId = uid();
    const root = makeSpan({ traceId, name: 'root' });
    const siblings = Array.from({ length: 50 }, (_, i) =>
      makeSpan({ traceId, parentSpanId: root.id, name: `s-${i}` }),
    );

    await start(exporter, root);
    for (const s of siblings) await start(exporter, s);
    // root + 50 siblings in maps → 51 entries
    assert.equal(pathMaps(exporter).name.size, 51);
    for (const s of siblings) await end(exporter, s);
    // siblings cleaned, root still active
    assert.equal(pathMaps(exporter).name.size, 1);
    assert.ok(pathMaps(exporter).name.has(root.id));
    await end(exporter, root);
    assert.equal(pathMaps(exporter).name.size, 0);
  });
});

// ---------------------------------------------------------------------------
// 7. Error spans: path attrs co-exist with exception event
// ---------------------------------------------------------------------------

describe('error spans: path attrs survive error info', () => {
  it('root error span has correct path and exception event', async () => {
    const { exporter, capture } = makeRig();
    const traceId = uid();
    const errorSpan = makeSpan({
      traceId,
      name: 'agent-run',
      errorInfo: { message: 'something went wrong', details: { stack: 'Error: ...\n  at foo:1' } },
    });
    await run(exporter, errorSpan);

    const otelSpan = capture.spans[0];
    assert.ok(otelSpan, 'span should be exported');
    const attrs = getAttrs(capture);
    assert.deepEqual(attrs['traceroot.span.path'], ['agent-run']);
    assert.deepEqual(attrs['traceroot.span.ids_path'], []);
    // Exception event should also be present
    assert.ok(otelSpan.events.length > 0, 'error span must have an exception event');
    assert.equal(otelSpan.events[0].name, 'exception');
    assert.equal(otelSpan.events[0].attributes?.['exception.message'], 'something went wrong');
  });

  it('child error span has correct path', async () => {
    const { exporter, capture } = makeRig();
    const traceId = uid();
    const root = makeSpan({ traceId, name: 'agent-run' });
    const errorChild = makeSpan({
      traceId,
      parentSpanId: root.id,
      name: 'failing-tool',
      type: SpanType.TOOL_CALL,
      errorInfo: { message: 'tool failed' },
    });
    await start(exporter, root);
    await run(exporter, errorChild);
    const attrs = getAttrs(capture, 0);
    assert.deepEqual(attrs['traceroot.span.path'], ['agent-run', 'failing-tool']);
    assert.deepEqual(attrs['traceroot.span.ids_path'], [root.id]);
    assert.equal(capture.spans[0].events[0]?.name, 'exception');
  });
});

// ---------------------------------------------------------------------------
// 8. isEvent edge cases
// ---------------------------------------------------------------------------

describe('isEvent edge cases', () => {
  it('regular span and event span in same trace: regular gets path, event does not', async () => {
    const { exporter, capture } = makeRig();
    const traceId = uid();
    const regular = makeSpan({ traceId, name: 'agent-run' });
    const evt = makeSpan({ traceId, parentSpanId: regular.id, name: 'token-chunk', isEvent: true });

    await start(exporter, regular);
    await start(exporter, evt); // no-op
    await end(exporter, evt); // exports but without path
    await end(exporter, regular); // exports with path

    // event span exported first (ended first)
    assert.equal(getAttrs(capture, 0)['traceroot.span.path'], undefined, 'event span has no path');
    // regular span exported second
    assert.deepEqual(getAttrs(capture, 1)['traceroot.span.path'], ['agent-run']);
  });

  it('multiple event spans in same trace do not pollute maps', async () => {
    const { exporter } = makeRig();
    const traceId = uid();
    const root = makeSpan({ traceId, name: 'root' });
    const events = Array.from({ length: 10 }, (_, i) =>
      makeSpan({ traceId, parentSpanId: root.id, name: `chunk-${i}`, isEvent: true }),
    );

    await start(exporter, root);
    for (const e of events) {
      await start(exporter, e); // no-op for each event
      await end(exporter, e); // exports without path
    }

    // Only root should be in the name map
    const maps = pathMaps(exporter);
    assert.equal(maps.name.size, 1);
    assert.deepEqual(maps.name.get(root.id), ['root']);
  });
});

// ---------------------------------------------------------------------------
// 9. Unicode and special character span names
// ---------------------------------------------------------------------------

describe('span names with special characters', () => {
  it('unicode span names are preserved in path', async () => {
    const { exporter, capture } = makeRig();
    const traceId = uid();
    const root = makeSpan({ traceId, name: '智能体运行' }); // 智能体运行
    const child = makeSpan({ traceId, parentSpanId: root.id, name: '模型生成' }); // 模型生成
    await start(exporter, root);
    await run(exporter, child);
    assert.deepEqual(getAttrs(capture, 0)['traceroot.span.path'], ['智能体运行', '模型生成']);
  });

  it('span names with dots and slashes are preserved', async () => {
    const { exporter, capture } = makeRig();
    const traceId = uid();
    const root = makeSpan({ traceId, name: 'my-agent/v2.0' });
    const child = makeSpan({ traceId, parentSpanId: root.id, name: 'openai.gpt-4o-mini' });
    await start(exporter, root);
    await run(exporter, child);
    assert.deepEqual(getAttrs(capture, 0)['traceroot.span.path'], [
      'my-agent/v2.0',
      'openai.gpt-4o-mini',
    ]);
  });

  it('empty span name is preserved (edge case)', async () => {
    const { exporter, capture } = makeRig();
    const traceId = uid();
    const root = makeSpan({ traceId, name: '' });
    await run(exporter, root);
    assert.deepEqual(getAttrs(capture)['traceroot.span.path'], ['']);
    assert.deepEqual(getAttrs(capture)['traceroot.span.ids_path'], []);
  });
});

// ---------------------------------------------------------------------------
// 10. Late-joining span: parent was already ended when child starts
// ---------------------------------------------------------------------------

describe('late-joining span (parent ended before child starts)', () => {
  it('late child with unknown parent is treated as root (no stale path)', async () => {
    const { exporter, capture } = makeRig();
    const traceId = uid();
    const parent = makeSpan({ traceId, name: 'parent' });
    const lateChild = makeSpan({ traceId, parentSpanId: parent.id, name: 'late-child' });

    await run(exporter, parent); // parent fully ends before child starts
    await run(exporter, lateChild); // child starts after parent map entry is deleted

    // parent ends: map entry deleted; late child: parentNamePath = undefined → root path
    assert.deepEqual(getAttrs(capture, 1)['traceroot.span.path'], ['late-child']);
    assert.deepEqual(getAttrs(capture, 1)['traceroot.span.ids_path'], []);
  });
});

// ---------------------------------------------------------------------------
// 11. OTel span metadata correctness (span kind, status, attributes)
// ---------------------------------------------------------------------------

describe('OTel span metadata correctness', () => {
  it('span kind is INTERNAL for AGENT_RUN', async () => {
    const { exporter, capture } = makeRig();
    const span = makeSpan({ name: 'agent', type: SpanType.AGENT_RUN });
    await run(exporter, span);
    const { SpanKind } = await import('@opentelemetry/api');
    assert.equal(capture.spans[0].kind, SpanKind.INTERNAL);
  });

  it('span kind is CLIENT for MODEL_GENERATION', async () => {
    const { exporter, capture } = makeRig();
    const span = makeSpan({ name: 'model-gen', type: SpanType.MODEL_GENERATION });
    await run(exporter, span);
    const { SpanKind } = await import('@opentelemetry/api');
    assert.equal(capture.spans[0].kind, SpanKind.CLIENT);
  });

  it('span kind is CLIENT for MCP_TOOL_CALL', async () => {
    const { exporter, capture } = makeRig();
    const span = makeSpan({ name: 'mcp-call', type: SpanType.MCP_TOOL_CALL });
    await run(exporter, span);
    const { SpanKind } = await import('@opentelemetry/api');
    assert.equal(capture.spans[0].kind, SpanKind.CLIENT);
  });

  it('OTel span is marked ended=true', async () => {
    const { exporter, capture } = makeRig();
    const span = makeSpan({ name: 'span' });
    await run(exporter, span);
    assert.equal(capture.spans[0].ended, true);
  });

  it('OTel traceId is padded to 32 hex chars', async () => {
    const { exporter, capture } = makeRig();
    // Use a short traceId that needs padding
    const span = makeSpan({ traceId: 'abc', name: 'short-trace' });
    await run(exporter, span);
    const traceId = capture.spans[0].spanContext().traceId;
    assert.equal(traceId.length, 32);
    assert.match(traceId, /^[0-9a-f]{32}$/);
  });

  it('OTel spanId is padded to 16 hex chars', async () => {
    const { exporter, capture } = makeRig();
    const span = makeSpan({ id: 'ab', name: 'short-span' });
    await run(exporter, span);
    const spanId = capture.spans[0].spanContext().spanId;
    assert.equal(spanId.length, 16);
    assert.match(spanId, /^[0-9a-f]{16}$/);
  });
});

// ---------------------------------------------------------------------------
// 12. Flush and shutdown do not throw
// ---------------------------------------------------------------------------

describe('flush and shutdown robustness', () => {
  it('flush on empty exporter does not throw', async () => {
    const { exporter } = makeRig();
    await assert.doesNotReject(() => exporter.flush());
  });

  it('shutdown on exporter with in-flight spans does not throw', async () => {
    const { exporter } = makeRig();
    const span = makeSpan({ name: 'in-flight' });
    await start(exporter, span); // started but not ended
    await assert.doesNotReject(() => exporter.shutdown());
  });

  it('double shutdown does not throw', async () => {
    const { exporter } = makeRig();
    await exporter.shutdown();
    await assert.doesNotReject(() => exporter.shutdown());
  });
});
