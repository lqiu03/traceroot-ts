/**
 * Tests for the shared test fixtures in ./test-helpers.ts.
 *
 * This file exists to give the extraction (CapturingExporter, attrs(),
 * assistantMessage(), makeFakeSessionClass()/makeRig() — previously
 * copy-pasted across ~14 files in this directory, see the code-review
 * finding this resolves) its own RED/GREEN coverage, independent of any one
 * consuming test file.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { ExportResultCode } from '@opentelemetry/core';
import type { ReadableSpan } from '@opentelemetry/sdk-trace-base';
import { instrumentPiCodingAgent } from '../src/instrumentation';
import {
  assistantMessage,
  attrs,
  CapturingExporter,
  makeFakeSessionClass,
  makeRig,
} from './test-helpers';

test('CapturingExporter captures exported spans and reports SUCCESS', () => {
  const capture = new CapturingExporter();
  const span = { name: 'fake-span' } as unknown as ReadableSpan;
  let reportedCode: number | undefined;

  capture.export([span], (result) => {
    reportedCode = result.code;
  });

  assert.deepEqual(capture.spans, [span]);
  assert.equal(reportedCode, ExportResultCode.SUCCESS);
});

test('assistantMessage() returns the minimal default fixture and applies shallow overrides', () => {
  const message = assistantMessage();

  assert.equal(message.role, 'assistant');
  assert.equal(message.stopReason, 'stop');
  assert.deepEqual(message.usage, {
    input: 1,
    output: 1,
    cacheRead: 0,
    cacheWrite: 0,
    totalTokens: 2,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
  });

  const overridden = assistantMessage({ stopReason: 'error', errorMessage: 'boom' });
  assert.equal(overridden.stopReason, 'error');
  assert.equal(overridden.errorMessage, 'boom');
  // Overrides must not mutate the base fixture used by other tests.
  assert.equal(assistantMessage().stopReason, 'stop');
});

test("attrs() reads a span's attributes as a plain record", () => {
  const span = { attributes: { foo: 'bar' } } as unknown as ReadableSpan;
  assert.deepEqual(attrs(span), { foo: 'bar' });
});

test('makeFakeSessionClass() returns a fresh class each call, and prompt() resolves by default', async () => {
  const SessionA = makeFakeSessionClass();
  const SessionB = makeFakeSessionClass();
  assert.notEqual(SessionA, SessionB, 'each call must return an independent class');

  const session = new SessionA();
  await assert.doesNotReject(() => session.prompt('anything'));
});

test('makeFakeSessionClass(shouldReject) rejects prompt() only for matching text', async () => {
  const Session = makeFakeSessionClass((text) => text === 'bad');
  const session = new Session();

  await assert.rejects(() => session.prompt('bad'));
  await assert.doesNotReject(() => session.prompt('good'));
});

test('makeRig() wires a fresh CapturingExporter and FakeAgentSession through instrumentPiCodingAgent', async () => {
  const { capture, Session } = makeRig();
  const session = new Session();

  await session.prompt('hello');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });

  assert.equal(capture.spans.length, 1);
  assert.equal(attrs(capture.spans[0]!)['openinference.span.kind'], 'AGENT');
});

test('makeRig(config) forwards captureContent/captureToolIo through to instrumentation', async () => {
  const { capture, Session } = makeRig({ captureContent: false });
  const session = new Session();

  await session.prompt('sensitive text');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });

  assert.equal(attrs(capture.spans[0]!)['input.value'], undefined);
});

// Regression test for the P2 finding this file resolves: instrumentPiCodingAgent()
// registers a process.once('beforeExit', flushOnExit) hook on every
// successful private-mode call (see instrumentation.ts) and never removes it
// on a normal call -- only a mid-setup failure rolls it back. Before
// makeRig() cleaned up after itself, calling it once per test across files
// with many tests (e.g. spans-truncation.test.ts's 11) reliably accumulated
// enough live listeners to trip Node's MaxListenersExceededWarning by the
// end of the file -- real noise that also risked masking a genuine future
// regression (e.g. a change that starts registering two hooks per call, or
// stops cleaning up at all) behind that same warning threshold. Calling
// makeRig() many more times than Node's default max (10) here proves the
// cleanup actually holds: the live listener count must stay flat, not merely
// "under the warning threshold by luck".
test('repeated makeRig() calls do not accumulate beforeExit listeners', () => {
  const baselineCount = process.listenerCount('beforeExit');

  const CALL_COUNT = 25; // deliberately > Node's default max of 10
  for (let i = 0; i < CALL_COUNT; i++) {
    makeRig();
  }

  assert.equal(
    process.listenerCount('beforeExit'),
    baselineCount,
    'makeRig() must remove the beforeExit listener it installs before returning, so repeated ' +
      'calls across a file with many tests never accumulate live listeners',
  );
});

// Complements the rig-hygiene test above by asserting the underlying
// production behavior directly, bypassing makeRig()'s own cleanup: each
// independent instrumentPiCodingAgent() call in private mode must register
// AT MOST ONE beforeExit listener, and that per-call registration must stay
// linear and bounded (exactly N listeners after N calls on N distinct SDK
// module objects) -- never 0 (silently dropped, so a private pipeline would
// never flush before a short-lived process exits) and never more than 1 per
// call (a double-registration bug that would export every span twice or
// worse). This is the "N independent private-mode instrument calls register
// at most one beforeExit listener per call" bounded-growth assertion the
// leak finding calls for; see instrumentation.ts's own comment at the
// registration site for why cleanup on a per-session dispose() is NOT
// attempted here (registration is intentionally bounded per distinct SDK
// object, not per session, and there is currently no public API to tear a
// single install back down after it succeeds).
test('N independent private-mode instrumentPiCodingAgent() calls register exactly N beforeExit listeners, never more', () => {
  const CALL_COUNT = 5;
  const addedListeners: Array<() => void> = [];

  try {
    for (let i = 0; i < CALL_COUNT; i++) {
      const before = new Set(process.listeners('beforeExit'));
      const Session = makeFakeSessionClass();
      instrumentPiCodingAgent({ AgentSession: Session }, { apiKey: 'test-key' }); // private mode
      const added = process.listeners('beforeExit').filter((listener) => !before.has(listener));

      assert.equal(
        added.length,
        1,
        `call #${i + 1} must register exactly one beforeExit listener (registered ${added.length})`,
      );
      addedListeners.push(...(added as Array<() => void>));
    }

    assert.equal(
      addedListeners.length,
      CALL_COUNT,
      'total beforeExit listeners registered must grow exactly linearly with the number of ' +
        'independently-instrumented SDK objects, not faster (double-registration) or slower ' +
        '(silently dropped)',
    );
  } finally {
    // Manual cleanup mirrors makeRig()'s own diff-and-remove idiom (and
    // provider-shared-mode-behavior.test.ts's) since instrumentPiCodingAgent()
    // itself exposes no way to uninstall a successful private-mode install.
    for (const listener of addedListeners) {
      process.removeListener('beforeExit', listener);
    }
  }
});
