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
