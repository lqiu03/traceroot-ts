/**
 * Lens: shared-mode-instrumentation.
 *
 * End-to-end version of shared-provider-detection.test.ts's checks, driven
 * through instrumentPiCodingAgent() and a real AgentSession.prompt() call
 * rather than createTracing() directly -- confirms the shared-mode path
 * produces correctly-shaped spans through the full patch/event-handling
 * pipeline, not just that createTracing() picks the right branch, and that
 * the beforeExit flush hook is gated on which mode actually ran.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { trace } from '@opentelemetry/api';
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import { InMemorySpanExporter, SimpleSpanProcessor } from '@opentelemetry/sdk-trace-base';
import { instrumentPiCodingAgent } from '../src/instrumentation';
import { assistantMessage, attrs, makeFakeSessionClass, makeRig } from './test-helpers';

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
// capturable exporter (see shared-provider-detection.test.ts's "_spanExporter
// override always forces private-provider mode" test) so which pipeline the
// spans land in is pinned down precisely -- private mode here could equally
// have been chosen by plain auto-discovery (nothing registered yet), since
// the underlying mechanism under test (the tracer is captured once and never
// re-evaluated per event) is identical either way.
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
