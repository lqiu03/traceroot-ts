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
import {
  assistantMessage,
  attrs,
  CapturingExporter,
  makeFakeSessionClass,
  makeRig,
} from './pi-test-helpers';

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

test('makeFakeSessionClass() returns a fresh class each call, and prompt() resolves once its final agent_end fires', async () => {
  const SessionA = makeFakeSessionClass();
  const SessionB = makeFakeSessionClass();
  assert.notEqual(SessionA, SessionB, 'each call must return an independent class');

  // prompt() no longer resolves merely by being called (see this file's
  // module header): it stays pending until an agent_end with willRetry !==
  // true fires, mirroring the real SDK's prompt() awaiting its whole
  // internal loop.
  const session = new SessionA();
  const done = session.prompt('anything');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'agent_end', messages: [], willRetry: false });
  await assert.doesNotReject(() => done);
});

test("makeFakeSessionClass()'s resolvePrompt() settles a pending call with no agent_end at all (the early-return mirror)", async () => {
  const Session = makeFakeSessionClass();
  const session = new Session();

  const done = session.prompt('a handled slash command');
  session.resolvePrompt();
  await assert.doesNotReject(() => done);
});

test("makeFakeSessionClass()'s rejectPrompt() rejects a pending call (the async-path failure mirror)", async () => {
  const Session = makeFakeSessionClass();
  const session = new Session();

  const done = session.prompt('a run that fails internally');
  session.emit({ type: 'agent_start' });
  session.rejectPrompt(new Error('internal failure'));
  await assert.rejects(() => done, /internal failure/);
});

test('makeFakeSessionClass(shouldReject) rejects prompt() SYNCHRONOUSLY only for matching text', async () => {
  const Session = makeFakeSessionClass((text) => text === 'bad');
  const session = new Session();

  await assert.rejects(() => session.prompt('bad'));
  const done = session.prompt('good');
  session.resolvePrompt();
  await assert.doesNotReject(() => done);
});

test('makeRig() wires a fresh CapturingExporter and FakeAgentSession through instrumentPiCodingAgent', async () => {
  const { capture, Session } = makeRig();
  const session = new Session();

  const done = session.prompt('hello');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
  await done;

  assert.equal(capture.spans.length, 1);
  assert.equal(attrs(capture.spans[0]!)['openinference.span.kind'], 'AGENT');
});

test('makeRig(config) forwards captureContent/captureToolIo through to instrumentation', async () => {
  const { capture, Session } = makeRig({ captureContent: false });
  const session = new Session();

  const done = session.prompt('sensitive text');
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'agent_end', messages: [assistantMessage()], willRetry: false });
  await done;

  assert.equal(attrs(capture.spans[0]!)['input.value'], undefined);
});

// Two beforeExit-listener-accounting tests previously lived here
// ("repeated makeRig() calls do not accumulate beforeExit listeners" and "N
// independent private-mode instrumentPiCodingAgent() calls register exactly
// N beforeExit listeners"). Both are deleted: this in-tree integration never
// builds or owns a TracerProvider of its own, so it has no
// process.once('beforeExit', flushOnExit) hook to register or roll back —
// process-exit flushing is core's responsibility now (see
// packages/traceroot/src/pi/instrumentation.ts's own comment on this, and
// packages/traceroot/src/traceroot.ts's process.once('beforeExit', ...)
// convention). There is nothing left of that machinery for these tests to
// guard.
