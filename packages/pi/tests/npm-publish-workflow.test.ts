/**
 * Regression test for .github/workflows/npm-publish.yml's publish-pi-test
 * job structure. `pnpm --filter @traceroot-ai/pi build` is plain `tsc`,
 * which already performs a full program type-check against the same
 * tsconfig.json used by `tsc --noEmit` and fails the job on any diagnostic —
 * a follow-up "Typecheck" step can only ever re-verify a program Build has
 * already proven clean, so it's pure redundant cost on every pi-* release.
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

function extractStepNames(jobBlock: string): string[] {
  const stepNameRegex = /-\s*name:\s*(.+)/g;
  const stepNames: string[] = [];
  let match: RegExpExecArray | null;
  while ((match = stepNameRegex.exec(jobBlock)) !== null) {
    stepNames.push(match[1].trim());
  }
  return stepNames;
}

test('publish-pi-test job does not run a redundant separate Typecheck step after Build', () => {
  const workflowText = readFileSync(workflowPath, 'utf8');
  const jobBlock = extractJobBlock(workflowText, 'publish-pi-test');
  const stepNames = extractStepNames(jobBlock);

  assert.ok(stepNames.includes('Build'), 'expected a "Build" step in the publish-pi-test job');
  assert.ok(
    !stepNames.includes('Typecheck'),
    'publish-pi-test should not run a separate "Typecheck" step: Build (tsc) ' +
      'already performs a full type-check of the identical source/tsconfig ' +
      'and fails the job on any diagnostic, so a follow-up `tsc --noEmit` ' +
      'step is pure redundant re-work',
  );
});
