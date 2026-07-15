/**
 * Lens: shared-mode-instrumentation.
 *
 * End-to-end version of provider-shared-mode-detection.test.ts's checks,
 * driven through instrumentPiCodingAgent() and a real AgentSession.prompt()
 * call rather than createTracing() directly -- confirms the shared-mode path
 * produces correctly-shaped spans through the full patch/event-handling
 * pipeline, not just that createTracing() picks the right branch, and that
 * the beforeExit flush hook is gated on which mode actually ran.
 *
 * Merged with the former shared-mode-no-apikey.test.ts: shared mode requires
 * no apiKey. Regression for the config-plumbing P0: when a real global
 * TracerProvider is already registered (shared mode), instrumentPiCodingAgent()
 * never builds an OTLP exporter, so it needs no apiKey at all. The
 * missing-apiKey safety net must therefore gate only the PRIVATE-provider path
 * that actually constructs an OTLP exporter, never the shared path. Before the
 * gate was moved after the shared-vs-private decision, a shared-mode call with
 * no apiKey bailed early and left AgentSession.prototype.prompt unpatched --
 * silently producing zero spans even though a fully valid export pipeline was
 * available. This is the exact shape TraceRoot.initialize() hands through: it
 * registers its own provider first, then wires pi in shared mode, and a host
 * that only passed apiKey programmatically (never via TRACEROOT_API_KEY) would
 * otherwise see nothing traced.
 *
 * Merged with the former shared-mode-tracer-reresolution.test.ts: the
 * shared-mode tracer must survive a global-provider swap. Regression for the
 * P0 where pi's shared-mode tracer was resolved ONCE at wrap time and closed
 * over forever. TraceRoot.shutdown() (and anything else that calls
 * trace.disable()) swaps the OTel API's internal ProxyTracerProvider for a
 * brand-new instance rather than mutating the old one, so a tracer captured
 * before the swap stays bound to the old, now-detached provider -- every
 * subsequent span goes dark, permanently, after a shutdown()/initialize()
 * cycle, with no recovery path. Those tests reproduce the plan's exact repro:
 * register provider A, obtain a shared-mode tracer, export a span (lands in
 * A); run shutdown()'s exact disable sequence; register provider B; export a
 * span on the SAME captured handle -- it must now route to B.
 *
 * Also folds in the former confirmed-bugfix-regressions.test.ts's
 * beforeExit forceFlush() test (its Bug 1): the beforeExit hook itself is
 * private-mode-only machinery, but the fix it guards lives in provider.ts's
 * flush/shutdown split, so it belongs alongside this file's other beforeExit
 * coverage rather than in a no-coherent-subject grab-bag file.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { context, propagation, trace } from '@opentelemetry/api';
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import { InMemorySpanExporter, SimpleSpanProcessor } from '@opentelemetry/sdk-trace-base';
import { createTracing } from '../src/provider';
import { resolveConfig } from '../src/config';
import { instrumentPiCodingAgent } from '../src/instrumentation';
import {
  assistantMessage,
  attrs,
  CapturingExporter,
  makeFakeSessionClass,
  makeRig,
} from './test-helpers';

test('a full prompt() run in shared mode exports its span tree through the pre-registered global provider', async () => {
  const exporter = new InMemorySpanExporter();
  const provider = new NodeTracerProvider();
  provider.addSpanProcessor(new SimpleSpanProcessor(exporter));
  provider.register();
  try {
    const Session = makeFakeSessionClass();
    const sdk = { AgentSession: Session };
    instrumentPiCodingAgent(sdk, { apiKey: 'test-key' }); // no _spanExporter -> shared mode
    const session = new Session();

    // AgentEvent sequence spliced from instrumentation.test.ts's
    // "full turn with one tool call" fixture (its lines 29-64).
    await session.prompt('list files in /tmp');
    session.emit({ type: 'agent_start' });
    session.emit({ type: 'turn_start' });
    session.emit({ type: 'message_start', message: assistantMessage() });
    session.emit({
      type: 'message_end',
      message: assistantMessage({
        content: [
          { type: 'text', text: "I'll list the files now." },
          { type: 'toolCall', id: 't1', name: 'bash' },
        ],
      }),
    });
    session.emit({
      type: 'tool_execution_start',
      toolCallId: 't1',
      toolName: 'bash',
      args: { command: 'ls /tmp' },
    });
    session.emit({
      type: 'tool_execution_end',
      toolCallId: 't1',
      toolName: 'bash',
      result: { content: [{ type: 'text', text: 'a.txt' }] },
      isError: false,
    });
    session.emit({ type: 'turn_end', message: assistantMessage(), toolResults: [] });
    session.emit({
      type: 'agent_end',
      messages: [assistantMessage({ content: [{ type: 'text', text: 'listed the files' }] })],
      willRetry: false,
    });

    const spans = exporter.getFinishedSpans();
    const kind = (s: (typeof spans)[number]) =>
      (s.attributes as Record<string, unknown>)['openinference.span.kind'];
    const rootSpan = spans.find((s) => kind(s) === 'AGENT');
    const llmSpan = spans.find((s) => kind(s) === 'LLM');
    const toolSpan = spans.find((s) => kind(s) === 'TOOL');

    assert.ok(rootSpan, 'the AGENT root span must be exported through the shared global provider');
    assert.equal(rootSpan!.name, 'AgentSession.prompt');
    assert.equal((rootSpan!.attributes as Record<string, unknown>)['session.id'], 'sess-1');
    assert.equal(
      (rootSpan!.attributes as Record<string, unknown>)['input.value'],
      'list files in /tmp',
    );
    assert.equal(
      (rootSpan!.attributes as Record<string, unknown>)['output.value'],
      'listed the files',
    );
    assert.equal(
      (rootSpan!.attributes as Record<string, unknown>)['traceroot.sdk.name'],
      'traceroot-pi',
    );
    assert.ok(llmSpan, 'the LLM span must also route through the shared provider');
    assert.ok(toolSpan, 'the TOOL span must also route through the shared provider');
    assert.equal(
      llmSpan!.parentSpanId,
      rootSpan!.spanContext().spanId,
      'LLM span must be a child of the AGENT root',
    );
  } finally {
    trace.disable();
  }
});

// Confirms config flags behave identically whether the pipeline is shared or
// private -- mirrors instrumentation.test.ts's captureContent: false
// coverage, but routed through the global provider instead of a private one.
test('captureContent: false suppresses input/output.value in shared mode exactly like in private mode', async () => {
  const exporter = new InMemorySpanExporter();
  const provider = new NodeTracerProvider();
  provider.addSpanProcessor(new SimpleSpanProcessor(exporter));
  provider.register();
  try {
    const Session = makeFakeSessionClass();
    const sdk = { AgentSession: Session };
    instrumentPiCodingAgent(sdk, { apiKey: 'test-key', captureContent: false }); // shared mode
    const session = new Session();

    await session.prompt('a prompt that must not be captured');
    session.emit({ type: 'agent_start' });
    session.emit({
      type: 'agent_end',
      messages: [assistantMessage({ content: [{ type: 'text', text: 'a reply' }] })],
      willRetry: false,
    });

    const spans = exporter.getFinishedSpans();
    const rootSpan = spans.find(
      (s) => (s.attributes as Record<string, unknown>)['openinference.span.kind'] === 'AGENT',
    );
    assert.ok(rootSpan, 'the AGENT root span must still be exported through the shared provider');
    assert.equal((rootSpan!.attributes as Record<string, unknown>)['session.id'], 'sess-1');
    assert.equal((rootSpan!.attributes as Record<string, unknown>)['input.value'], undefined);
    assert.equal((rootSpan!.attributes as Record<string, unknown>)['output.value'], undefined);
  } finally {
    trace.disable();
  }
});

// Positive assertion of the non-retroactive-rebind guarantee named in this
// phase's commit message ("a tracer already bound to a private provider
// never retroactively rebinds if a real provider registers later") --
// instrumentPiCodingAgent() builds and closes over a tracer exactly once,
// inside createTracing(), so a REAL global provider that registers later in
// the same process must never receive spans from a session instrumented
// before that registration. Uses _spanExporter to force private mode with a
// capturable exporter (see provider-shared-mode-detection.test.ts's
// "_spanExporter override always forces private-provider mode" test) so which
// pipeline the spans land in is pinned down precisely -- private mode here
// could equally have been chosen by plain auto-discovery (nothing registered
// yet), since the underlying mechanism under test (the tracer is captured once
// and never re-evaluated per event) is identical either way.
test('DOCUMENTED CAVEAT: a session already bound to a private provider never retroactively rebinds onto a global provider registered later', async () => {
  const { capture, Session } = makeRig(); // _spanExporter -> private provider, bound now

  const lateExporter = new InMemorySpanExporter();
  const lateProvider = new NodeTracerProvider();
  lateProvider.addSpanProcessor(new SimpleSpanProcessor(lateExporter));
  lateProvider.register(); // registers AFTER instrumentPiCodingAgent() already committed
  try {
    const session = new Session();
    await session.prompt('does this rebind onto the late-registered provider?');
    session.emit({ type: 'agent_start' });
    session.emit({
      type: 'agent_end',
      messages: [assistantMessage({ content: [{ type: 'text', text: 'no rebind' }] })],
      willRetry: false,
    });

    const rootSpan = capture.spans.find((s) => attrs(s)['openinference.span.kind'] === 'AGENT');
    assert.ok(
      rootSpan,
      "the AGENT span must still land in this sdk's own private pipeline, bound at instrumentPiCodingAgent() time",
    );
    assert.deepEqual(
      lateExporter.getFinishedSpans(),
      [],
      'a provider registered AFTER instrumentation must never receive spans from an already-private-bound session',
    );
  } finally {
    trace.disable();
  }
});

// The only real test of the one-call-site change in instrumentation.ts: the
// `if (ownsProvider)` branch must actually gate the beforeExit hook. Uses a
// before/after set difference (rather than a raw listenerCount delta) so the
// leaked listener is also cleaned up, since 'beforeExit' listeners added via
// process.once() would otherwise accumulate across tests in this file.
test('shared mode does not register a beforeExit flush hook (the shared provider owns flush)', () => {
  const provider = new NodeTracerProvider();
  provider.register();
  try {
    const before = new Set(process.listeners('beforeExit'));
    const Session = makeFakeSessionClass();
    instrumentPiCodingAgent({ AgentSession: Session }, { apiKey: 'test-key' }); // shared
    const added = process.listeners('beforeExit').filter((l) => !before.has(l));
    try {
      assert.equal(added.length, 0);
    } finally {
      for (const l of added) process.removeListener('beforeExit', l);
    }
  } finally {
    trace.disable();
  }
});

// Positive control for the test above: proves the shared-mode assertion
// isn't passing simply because instrumentation stopped adding beforeExit
// hooks entirely. Nothing registered here -- private mode must still add
// exactly one hook, since it owns its own provider's flush.
//
// Kept distinct from the "beforeExit hook calls forceFlush(), not shutdown()"
// test below: that test asserts the hook's BEHAVIOR when it fires (forceFlush,
// not shutdown, so a later run still exports); this one asserts the hook's
// REGISTRATION COUNT (exactly one is added, as the positive control for the
// "shared mode adds zero" test above). Neither subsumes the other, so both
// are kept.
test('private mode registers exactly one beforeExit flush hook (it owns the provider)', () => {
  const before = new Set(process.listeners('beforeExit'));
  const Session = makeFakeSessionClass();
  instrumentPiCodingAgent({ AgentSession: Session }, { apiKey: 'test-key' }); // private
  const added = process.listeners('beforeExit').filter((l) => !before.has(l));
  try {
    assert.equal(added.length, 1);
  } finally {
    for (const l of added) process.removeListener('beforeExit', l);
  }
});

// Regression for the P0 where the beforeExit hook called shutdown() instead
// of forceFlush() -- shutdown() permanently disables the export pipeline, so
// the FIRST drain of the event loop in any long-lived host process would
// silently kill every subsequent run's export. Complements the registration
// tests above (which only assert a hook is added) by capturing the actual
// listener and firing it, then proving export still works afterward.
test('beforeExit hook calls forceFlush(), not shutdown() — a later run still exports after it fires', async () => {
  const capture = new CapturingExporter();
  const Session = makeFakeSessionClass();
  const sdk = { AgentSession: Session };

  const originalOnce = process.once.bind(process);
  let beforeExitListener: (() => void) | undefined;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (process as any).once = (event: string, listener: (...a: unknown[]) => void) => {
    if (event === 'beforeExit') beforeExitListener = listener as () => void;
    return originalOnce(event as never, listener as never);
  };
  try {
    instrumentPiCodingAgent(sdk, { apiKey: 'k', _spanExporter: capture });
  } finally {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (process as any).once = originalOnce;
  }

  assert.ok(beforeExitListener, 'instrumentPiCodingAgent() must register a beforeExit hook');

  const session = new Session();
  await session.prompt('first run');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
  assert.equal(capture.spans.length, 1, 'run 1 exports normally');

  // Simulate the event loop draining once, as it would in a real long-lived
  // host process after the first prompt's work settles.
  assert.doesNotThrow(() => beforeExitListener!());
  await new Promise((resolve) => setTimeout(resolve, 10));

  // Run 2, well after the beforeExit hook already fired. If the hook had
  // called shutdown() (the bug) instead of forceFlush(), the pipeline would
  // be permanently disabled and this span would never export — exactly the
  // "first drain of the event loop kills all future export" failure mode.
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });

  assert.equal(
    capture.spans.length,
    2,
    'a second run after beforeExit fired must still export — forceFlush() must not permanently ' +
      'disable the pipeline the way shutdown() would',
  );
});

// Extends the shared FakeAgentSession with a steer() entry point (which the
// base fixture omits) so steer()-as-first-interaction can be exercised in
// shared mode, matching the real SDK's standalone steer() method. Fresh per
// call, like makeFakeSessionClass itself, so prototype patches never stack.
function makeSteerableSessionClass() {
  const Base = makeFakeSessionClass();
  return class SteerableAgentSession extends Base {
    async steer(_text: string, _images?: unknown[]): Promise<void> {}
  };
}

// The private-mode equivalents (session-dispose.test.ts's "dispose() mid-run
// ... force-closes ..." and steer-followup-instrumentation.test.ts's
// "calling steer() as the FIRST interaction ...") only ever validate the
// private-mode tracer. Shared mode re-resolves its tracer per span through the
// global provider (see provider.ts's createReresolvingSharedTracer), a
// materially different code path, so the same dispose-sweep and
// steer-first-subscribe behaviors are re-asserted here end-to-end through a
// real, pre-registered global provider.

test('dispose() mid-run in shared mode force-closes the open AGENT/LLM/TOOL spans through the shared provider, identically to private mode', () => {
  const exporter = new InMemorySpanExporter();
  const provider = new NodeTracerProvider();
  provider.addSpanProcessor(new SimpleSpanProcessor(exporter));
  provider.register();
  try {
    const Session = makeFakeSessionClass();
    const sdk = { AgentSession: Session };
    instrumentPiCodingAgent(sdk, { apiKey: 'test-key' }); // no _spanExporter -> shared mode
    const session = new Session();

    // Open a run and leave it mid-flight (root + LLM + tool all open), then
    // dispose() before agent_end -- the exact private-mode scenario, now in
    // shared mode.
    void session.prompt('long-running task in shared mode');
    session.emit({ type: 'agent_start' });
    session.emit({ type: 'message_start', message: assistantMessage({ model: 'mid-run-model' }) });
    session.emit({
      type: 'tool_execution_start',
      toolCallId: 'call-1',
      toolName: 'read_file',
      args: { path: '/tmp/x' },
    });
    assert.equal(
      exporter.getFinishedSpans().length,
      0,
      'no span should export before dispose() while still open',
    );

    assert.doesNotThrow(() => session.dispose());
    assert.equal(session.disposed, true, 'the real dispose() must still run and mark the session');

    const spans = exporter.getFinishedSpans();
    const kind = (s: (typeof spans)[number]) => attrs(s)['openinference.span.kind'];
    const rootSpan = spans.find((s) => kind(s) === 'AGENT');
    const llmSpan = spans.find((s) => attrs(s)['gen_ai.request.model'] === 'mid-run-model');
    const toolSpan = spans.find((s) => attrs(s)['gen_ai.tool.call.id'] === 'call-1');

    assert.equal(
      spans.length,
      3,
      'the open AGENT/LLM/TOOL spans must all be force-closed and exported through the shared provider',
    );
    assert.ok(rootSpan, 'the AGENT root span must export through the shared provider');
    assert.ok(llmSpan, 'the LLM span must export through the shared provider');
    assert.ok(toolSpan, 'the TOOL span must export through the shared provider');
    assert.equal(
      attrs(rootSpan!)['traceroot.pi.force_closed'],
      true,
      'the root span must be marked force_closed in shared mode, exactly as in private mode',
    );
    assert.equal(attrs(llmSpan!)['traceroot.pi.force_closed'], true);
    assert.equal(attrs(toolSpan!)['traceroot.pi.force_closed'], true);
  } finally {
    trace.disable();
  }
});

test('calling steer() as the FIRST interaction in shared mode still attaches tracing through the shared provider', async () => {
  const exporter = new InMemorySpanExporter();
  const provider = new NodeTracerProvider();
  provider.addSpanProcessor(new SimpleSpanProcessor(exporter));
  provider.register();
  try {
    const Session = makeSteerableSessionClass();
    const sdk = { AgentSession: Session };
    instrumentPiCodingAgent(sdk, { apiKey: 'test-key' }); // no _spanExporter -> shared mode
    const session = new Session();

    // No prompt() anywhere -- steer() is the only entry point this host uses,
    // and it must attach the span listener itself even in shared mode.
    await session.steer('do X instead');
    session.emit({ type: 'agent_start' });
    session.emit({
      type: 'agent_end',
      messages: [assistantMessage({ content: [{ type: 'text', text: 'steered reply' }] })],
      willRetry: false,
    });

    const spans = exporter.getFinishedSpans();
    const rootSpan = spans.find((s) => attrs(s)['openinference.span.kind'] === 'AGENT');
    assert.ok(
      rootSpan,
      'steer() as the first interaction must attach the listener and export its AGENT span through ' +
        'the shared provider, not rely on prompt() having been called first',
    );
    assert.equal(attrs(rootSpan!)['session.id'], 'sess-1');
    assert.equal(
      attrs(rootSpan!)['output.value'],
      'steered reply',
      'the shared-mode run driven by a steer()-first session must still capture its output',
    );
  } finally {
    trace.disable();
  }
});

// The whole point of this regression is a genuinely-absent apiKey, so pin
// TRACEROOT_API_KEY to unset for the duration regardless of the ambient env.
function withoutApiKeyEnv(fn: () => Promise<void> | void): Promise<void> | void {
  const saved = process.env.TRACEROOT_API_KEY;
  delete process.env.TRACEROOT_API_KEY;
  try {
    return fn();
  } finally {
    if (saved === undefined) delete process.env.TRACEROOT_API_KEY;
    else process.env.TRACEROOT_API_KEY = saved;
  }
}

test('shared mode instruments prompt() and exports its span tree with NO apiKey at all', async () => {
  const exporter = new InMemorySpanExporter();
  const provider = new NodeTracerProvider();
  provider.addSpanProcessor(new SimpleSpanProcessor(exporter));
  provider.register();
  try {
    await withoutApiKeyEnv(async () => {
      const Session = makeFakeSessionClass();
      const sdk = { AgentSession: Session };
      const originalPrompt = Session.prototype.prompt;

      // No apiKey, no _spanExporter — exactly what initialize() forwards in
      // shared mode when neither the option nor TRACEROOT_API_KEY is set.
      instrumentPiCodingAgent(sdk, {});

      assert.notEqual(
        Session.prototype.prompt,
        originalPrompt,
        'prompt() MUST be patched in shared mode even without an apiKey — a real global provider owns export',
      );

      const session = new Session();
      await session.prompt('trace me with no api key');
      session.emit({ type: 'agent_start' });
      session.emit({
        type: 'agent_end',
        messages: [assistantMessage({ content: [{ type: 'text', text: 'traced anyway' }] })],
        willRetry: false,
      });

      const rootSpan = exporter
        .getFinishedSpans()
        .find((s) => attrs(s)['openinference.span.kind'] === 'AGENT');
      assert.ok(
        rootSpan,
        'the AGENT root span must export through the shared provider with no apiKey configured',
      );
      assert.equal(attrs(rootSpan!)['input.value'], 'trace me with no api key');
      assert.equal(attrs(rootSpan!)['output.value'], 'traced anyway');
    });
  } finally {
    trace.disable();
  }
});

// Companion negative control: the PRIVATE path (no global provider registered)
// with no apiKey and no exporter override still cannot build an export
// pipeline, so it must keep degrading to an untouched no-op rather than
// pretending to instrument. Confirms the gate was scoped, not deleted.
test('private mode with no apiKey and no exporter override still degrades to a no-op', () => {
  withoutApiKeyEnv(() => {
    const Session = makeFakeSessionClass();
    const originalPrompt = Session.prototype.prompt;
    const sdk = { AgentSession: Session };

    const result = instrumentPiCodingAgent(sdk, {}); // nothing registered globally -> private

    assert.equal(result, sdk);
    assert.equal(
      Session.prototype.prompt,
      originalPrompt,
      'prompt must stay untouched when no pipeline (shared provider or apiKey) is available',
    );
  });
});

function registerProvider(): { provider: NodeTracerProvider; exporter: InMemorySpanExporter } {
  const exporter = new InMemorySpanExporter();
  const provider = new NodeTracerProvider();
  provider.addSpanProcessor(new SimpleSpanProcessor(exporter));
  provider.register();
  return { provider, exporter };
}

// Exactly the three process-wide resets TraceRoot.shutdown() performs, after
// tearing down its own provider.
async function runShutdownSequence(provider: NodeTracerProvider): Promise<void> {
  await provider.shutdown();
  trace.disable();
  context.disable();
  propagation.disable();
}

test('a shared-mode tracer handle re-resolves onto a provider registered after a shutdown()/disable() cycle', async () => {
  const a = registerProvider();
  let b: { provider: NodeTracerProvider; exporter: InMemorySpanExporter } | undefined;
  try {
    // Shared mode: no _spanExporter override, a real global provider (A) live.
    const handle = createTracing(resolveConfig({ apiKey: 'test-key' }));
    assert.equal(handle.ownsProvider, false, 'sanity: provider A must put us in shared mode');

    const first = handle.tracer.startSpan('span-before-cycle');
    first.end();
    assert.deepEqual(
      a.exporter.getFinishedSpans().map((s) => s.name),
      ['span-before-cycle'],
      'sanity: the pre-cycle span must land in provider A',
    );

    // TraceRoot.shutdown()'s exact sequence: tear down A, then reset the OTel
    // globals (which swaps the internal ProxyTracerProvider wholesale).
    await runShutdownSequence(a.provider);

    // A fresh provider takes over the global slot, exactly as a subsequent
    // TraceRoot.initialize() would install.
    b = registerProvider();

    // The SAME captured handle from before the cycle must now route to B.
    const second = handle.tracer.startSpan('span-after-cycle');
    second.end();

    assert.deepEqual(
      b.exporter.getFinishedSpans().map((s) => s.name),
      ['span-after-cycle'],
      'the captured shared-mode handle must re-resolve onto provider B after the disable() cycle',
    );
  } finally {
    trace.disable();
  }
});

test('an already-instrumented AgentSession keeps exporting across a shutdown()/initialize() cycle in shared mode', async () => {
  const a = registerProvider();
  let b: { provider: NodeTracerProvider; exporter: InMemorySpanExporter } | undefined;
  try {
    const Session = makeFakeSessionClass();
    const sdk = { AgentSession: Session };
    instrumentPiCodingAgent(sdk, { apiKey: 'test-key' }); // shared mode, wraps the prototype once

    // First run lands in A.
    const s1 = new Session();
    await s1.prompt('first run');
    s1.emit({ type: 'agent_start' });
    s1.emit({
      type: 'agent_end',
      messages: [assistantMessage({ content: [{ type: 'text', text: 'done A' }] })],
      willRetry: false,
    });
    assert.ok(
      a.exporter.getFinishedSpans().some((s) => attrs(s)['openinference.span.kind'] === 'AGENT'),
      'sanity: the first run must export through provider A',
    );

    // Simulate TraceRoot.shutdown() -> TraceRoot.initialize(): the prototype is
    // still wrapped (Symbol.for() guard blocks re-instrumentation), so recovery
    // depends entirely on the captured tracer re-resolving, not re-wrapping.
    await runShutdownSequence(a.provider);
    b = registerProvider();

    const s2 = new Session();
    await s2.prompt('second run');
    s2.emit({ type: 'agent_start' });
    s2.emit({
      type: 'agent_end',
      messages: [assistantMessage({ content: [{ type: 'text', text: 'done B' }] })],
      willRetry: false,
    });

    const rootB = b.exporter
      .getFinishedSpans()
      .find((s) => attrs(s)['openinference.span.kind'] === 'AGENT');
    assert.ok(
      rootB,
      'the second run must export through provider B after the cycle — no silent black hole',
    );
    assert.equal(attrs(rootB!)['input.value'], 'second run');
  } finally {
    trace.disable();
  }
});
