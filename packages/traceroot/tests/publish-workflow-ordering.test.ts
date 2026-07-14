/**
 * Regression test for .github/workflows/npm-publish.yml's publish-traceroot
 * job. traceroot references @traceroot-ai/pi via `workspace:^`, which pnpm
 * rewrites to a concrete `^<version>` at publish time. Nothing forces the pi-*
 * release to be published to npm before the traceroot release, so a traceroot
 * publish can ship a reference to a pi version that does not exist on the
 * registry yet — an unresolvable peer for anyone who opts into the pi
 * integration. The publish-traceroot job must therefore run a pre-flight check
 * that the pinned pi version is already on npm, BEFORE it publishes traceroot.
 */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const workflowPath = path.join(
  path.dirname(fileURLToPath(import.meta.url)),
  '..',
  '..',
  '..',
  '.github',
  'workflows',
  'npm-publish.yml',
);

// Normalize CRLF so byte-for-byte line matching works on a Windows checkout
// (core.autocrlf=true) as well as on the Linux CI runners.
function readWorkflow(): string {
  return readFileSync(workflowPath, 'utf8').replace(/\r\n/g, '\n');
}

function extractJobBlock(workflowText: string, jobName: string): string {
  const jobHeaderRegex = new RegExp(`^  ${jobName}:\\n`, 'm');
  const jobHeaderMatch = jobHeaderRegex.exec(workflowText);
  assert.ok(jobHeaderMatch, `job "${jobName}" not found in workflow`);

  const jobStart = jobHeaderMatch.index + jobHeaderMatch[0].length;
  const remainder = workflowText.slice(jobStart);

  // The next top-level job key starts a line indented exactly two spaces.
  const nextJobRegex = /^ {2}[A-Za-z0-9_-]+:\n/m;
  const nextJobMatch = nextJobRegex.exec(remainder);
  return nextJobMatch ? remainder.slice(0, nextJobMatch.index) : remainder;
}

test('publish-traceroot verifies the pinned pi version is on npm before publishing', () => {
  const jobBlock = extractJobBlock(readWorkflow(), 'publish-traceroot');

  // The guard must query the registry for the exact pi version this release
  // references...
  const checkIndex = jobBlock.search(/npm view\s+["']?@traceroot-ai\/pi/);
  assert.notEqual(
    checkIndex,
    -1,
    'publish-traceroot must run `npm view @traceroot-ai/pi@<version>` to confirm the pinned pi ' +
      'version is published before publishing traceroot',
  );

  // ...read from the in-repo manifest, not a hardcoded literal, so it tracks
  // the version that pnpm will actually rewrite `workspace:^` to.
  assert.match(
    jobBlock,
    /packages\/pi\/package\.json/,
    'the pi-version pre-flight check should read the version from packages/pi/package.json',
  );

  // ...and it must run BEFORE the traceroot publish step, or it guards nothing.
  const publishIndex = jobBlock.search(/pnpm --filter @traceroot-ai\/traceroot publish/);
  assert.notEqual(
    publishIndex,
    -1,
    'expected a `pnpm --filter @traceroot-ai/traceroot publish` step',
  );
  assert.ok(
    checkIndex < publishIndex,
    'the pi-published pre-flight check must run before the traceroot publish step',
  );
});
