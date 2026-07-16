/**
 * Regression guard for a P3 code-review finding on
 * packages/pi/src/instrumentation.ts (safeCloseDanglingSpan): a span
 * implementation whose force-close throws on EVERY attempt would otherwise
 * emit one console.warn per dangling span, on every sweep, indefinitely —
 * flooding the host's logs and burying the first, actually-useful warning.
 *
 * This is a behavioral test, driven entirely through the public API: it opens
 * a run with many still-open spans, poisons every span's force-close so each
 * one throws, disposes the session mid-run (triggering the sweep), and asserts
 * that the resulting warnings are rate-limited — the first failure is still
 * surfaced, but the host is not warned once per failure, and exactly one
 * "further warnings suppressed" notice is emitted once the cap is crossed.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { trace } from '@opentelemetry/api';
import { Span, SimpleSpanProcessor } from '@opentelemetry/sdk-trace-base';
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import { instrumentPiCodingAgent } from '../src/pi/instrumentation';
import { assistantMessage, CapturingExporter, makeFakeSessionClass } from './pi-test-helpers';

// Registers a real, freshly-registered global TracerProvider wired to
// `capture`, replacing the deleted private-exporter (`_spanExporter`)
// injection path — see pi-test-helpers.ts's makeRig() for the full
// isolation rationale behind calling trace.disable() first.
function registerCapturingProvider(capture: CapturingExporter): void {
  trace.disable();
  const provider = new NodeTracerProvider();
  provider.addSpanProcessor(new SimpleSpanProcessor(capture));
  provider.register();
}

// Number of still-open TOOL spans to leave dangling — plus the LLM and root
// spans, the sweep tries to force-close TOOL_FAILURES + 2 spans in total.
// Chosen comfortably above any reasonable warning cap so the rate limiter must
// engage and leave a wide margin between "failures" and "warnings emitted".
const TOOL_FAILURES = 40;

test('a systematically-failing span force-close is rate-limited, not warned once per span', () => {
  const capture = new CapturingExporter();
  const Session = makeFakeSessionClass();
  const sdk = { AgentSession: Session };
  registerCapturingProvider(capture);
  instrumentPiCodingAgent(sdk, {});
  const session = new Session();

  // Open one run and leave it mid-flight with the root span, an LLM span, and
  // many TOOL spans all still open — none of them will see a normal close.
  void session.prompt('systematically failing run');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'message_start', message: assistantMessage({ model: 'rate-limit-llm' }) });
  for (let i = 0; i < TOOL_FAILURES; i++) {
    session.emit({
      type: 'tool_execution_start',
      toolCallId: `call-${i}`,
      toolName: 'read_file',
      args: { path: `/tmp/${i}` },
    });
  }
  const totalFailures = TOOL_FAILURES + 2; // every TOOL span, plus LLM + root

  // Poison the force-close of EVERY span (not just one): setAttribute throws on
  // the force_closed marker for all spans, so each span's force-close throws
  // out of closeDanglingSpan() and into safeCloseDanglingSpan()'s catch.
  type SetAttributeFn = typeof Span.prototype.setAttribute;
  const originalSetAttribute: SetAttributeFn = Span.prototype.setAttribute;
  Span.prototype.setAttribute = function (
    this: Span,
    key: string,
    value?: Parameters<SetAttributeFn>[1],
  ): Span {
    if (key === 'traceroot.pi.force_closed') {
      throw new Error('injected systematic force-close failure');
    }
    return originalSetAttribute.call(this, key, value);
  } as SetAttributeFn;

  const warnings: string[] = [];
  const originalWarn = console.warn;
  console.warn = (first?: unknown): void => {
    warnings.push(typeof first === 'string' ? first : String(first));
  };

  try {
    // One dispose() sweeps and force-closes every open span — all totalFailures
    // of them fail identically.
    assert.doesNotThrow(() => session.dispose());
  } finally {
    Span.prototype.setAttribute = originalSetAttribute;
    console.warn = originalWarn;
  }

  const perSpanWarnings = warnings.filter((m) => m.includes('failed to force-close a dangling'));
  const suppressionNotices = warnings.filter(
    (m) => m.includes('further') && m.includes('suppressed'),
  );

  // The first, genuinely-useful failure must still be surfaced — rate limiting
  // must not silence failures from the very start.
  assert.ok(
    perSpanWarnings.length >= 1,
    'the first dangling-span close failure must still be warned about, not suppressed from the start',
  );
  // ...but the host must NOT be warned once per failure.
  assert.ok(
    perSpanWarnings.length < totalFailures,
    `expected the per-span close-failure warnings to be capped below the ${totalFailures} ` +
      `failures, got ${perSpanWarnings.length} — the warning is not rate-limited`,
  );
  // Exactly one suppression notice tells the host why the flood stopped.
  assert.equal(
    suppressionNotices.length,
    1,
    'expected exactly one "further warnings suppressed" notice once the cap is crossed, got ' +
      String(suppressionNotices.length),
  );
});
