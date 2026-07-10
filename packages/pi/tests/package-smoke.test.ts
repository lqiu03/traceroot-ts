import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, readdirSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath, pathToFileURL } from 'node:url';
import type { ExportResult } from '@opentelemetry/core';
import { ExportResultCode } from '@opentelemetry/core';
import type { ReadableSpan, SpanExporter } from '@opentelemetry/sdk-trace-base';

const rootDir = fileURLToPath(new URL('..', import.meta.url));

class CapturingExporter implements SpanExporter {
  readonly spans: ReadableSpan[] = [];
  export(spans: ReadableSpan[], resultCallback: (result: ExportResult) => void): void {
    this.spans.push(...spans);
    resultCallback({ code: ExportResultCode.SUCCESS });
  }
  async shutdown(): Promise<void> {}
}

test('the packed tarball exports instrumentPiCodingAgent and reports its own packed version', async () => {
  // Rebuild first so this test is correct standalone, not just after CI's build step.
  execFileSync('pnpm', ['build'], { cwd: rootDir, stdio: 'pipe' });

  const tempRoot = mkdtempSync(join(tmpdir(), 'traceroot-pi-pack-'));
  execFileSync('pnpm', ['pack', '--pack-destination', tempRoot], { cwd: rootDir, stdio: 'pipe' });

  const packedFile = readdirSync(tempRoot).find((entry) => entry.endsWith('.tgz'));
  assert.ok(packedFile, 'pnpm pack must create a tarball');
  const tarball = join(tempRoot, packedFile!);

  const entries = execFileSync('tar', ['-tf', tarball], { encoding: 'utf8' }).split('\n');
  assert.ok(
    entries.includes('package/package.json'),
    'package.json must be included in the tarball',
  );
  assert.ok(
    entries.includes('package/dist/index.js'),
    'built dist/index.js must be included in the tarball',
  );
  assert.ok(
    !entries.some((e) => e.startsWith('package/src/')),
    'raw TS source must NOT be published',
  );

  const packageDir = join(tempRoot, 'node_modules', '@traceroot-ai', 'pi');
  mkdirSync(packageDir, { recursive: true });
  execFileSync('tar', ['-xzf', tarball, '-C', packageDir, '--strip-components=1']);

  const packedPackageJson = JSON.parse(readFileSync(join(packageDir, 'package.json'), 'utf8')) as {
    version: string;
  };

  const mod = (await import(pathToFileURL(join(packageDir, 'dist', 'index.js')).href)) as {
    instrumentPiCodingAgent: (sdk: unknown, config?: unknown) => unknown;
  };
  assert.equal(typeof mod.instrumentPiCodingAgent, 'function');

  const capture = new CapturingExporter();

  class FakeAgentSession {
    sessionId = 'sess-1';
    private listeners: Array<(event: unknown) => void> = [];
    async prompt(): Promise<void> {}
    subscribe(listener: (event: unknown) => void): () => void {
      this.listeners.push(listener);
      return () => {
        this.listeners = this.listeners.filter((l) => l !== listener);
      };
    }
    emit(event: unknown): void {
      for (const listener of this.listeners) listener(event);
    }
  }

  const fakeSdk = { AgentSession: FakeAgentSession };
  mod.instrumentPiCodingAgent(fakeSdk, { apiKey: 'k', _spanExporter: capture });

  const session = new FakeAgentSession();
  await session.prompt();
  session.emit({ type: 'agent_start' });
  session.emit({ type: 'agent_end', messages: [], willRetry: false });

  assert.equal(capture.spans.length, 1);
  const rootSpan = capture.spans[0]!;
  assert.equal(
    (rootSpan.attributes as Record<string, unknown>)['traceroot.sdk.version'],
    packedPackageJson.version,
    'the packed, installed artifact must read its own version from the packed package.json, not the dev one',
  );
});
