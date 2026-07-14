/**
 * Regression test for the pi/traceroot Node `engines` mismatch.
 *
 * @traceroot-ai/pi requires a strictly newer Node than @traceroot-ai/traceroot
 * promises to support (pi tracks the Node floor of the SDK it instruments,
 * @earendil-works/pi-coding-agent, which is >=22.19.0). If traceroot listed pi
 * as a hard `dependency`, every traceroot consumer — including one running on
 * traceroot's own advertised minimum Node — would be transitively forced to
 * install a package that cannot run there (an npm EBADENGINE warning, or a hard
 * failure under --engine-strict), even a consumer that never touches the pi
 * integration at all. CI only ever runs one Node version, so this never turns
 * red there; it has to be asserted structurally.
 *
 * The integration is already lazy-`require()`d and optional in code
 * (see wirePiCodingAgentInstrumentation() in src/instrumentation.ts), exactly
 * like the other optional peers (openai / @openai/agents / @anthropic-ai/sdk),
 * so pi belongs in `peerDependencies` (optional), not `dependencies`.
 */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const testDir = path.dirname(fileURLToPath(import.meta.url));

interface PackageManifest {
  engines?: { node?: string };
  dependencies?: Record<string, string>;
  peerDependencies?: Record<string, string>;
  peerDependenciesMeta?: Record<string, { optional?: boolean }>;
}

function readManifest(relativePath: string): PackageManifest {
  return JSON.parse(readFileSync(path.join(testDir, relativePath), 'utf8')) as PackageManifest;
}

type Version = [number, number, number];

function parseNodeFloor(engines: PackageManifest['engines']): Version {
  const raw = engines?.node ?? '';
  const match = raw.match(/(\d+)(?:\.(\d+))?(?:\.(\d+))?/);
  assert.ok(match, `could not parse a Node version floor from engines.node="${raw}"`);
  return [Number(match[1]), Number(match[2] ?? 0), Number(match[3] ?? 0)];
}

function isStrictlyGreater(a: Version, b: Version): boolean {
  for (let i = 0; i < 3; i++) {
    if (a[i] !== b[i]) return a[i] > b[i];
  }
  return false;
}

test('traceroot does not force a stricter-engine pi onto consumers that meet its own Node floor', () => {
  const traceroot = readManifest('../package.json');
  const pi = readManifest('../../pi/package.json');

  const trFloor = parseNodeFloor(traceroot.engines);
  const piFloor = parseNodeFloor(pi.engines);

  // Only meaningful while pi's floor is strictly newer than traceroot's; if a
  // future change brings the two into line this guard simply passes trivially.
  if (!isStrictlyGreater(piFloor, trFloor)) return;

  assert.ok(
    !(traceroot.dependencies && '@traceroot-ai/pi' in traceroot.dependencies),
    'pi has a stricter Node engines floor than traceroot but is listed as a hard dependency; ' +
      "that transitively forces every traceroot consumer (even on traceroot's own minimum Node, " +
      'even those not using pi) to install a package they cannot run. Move it to an optional ' +
      'peerDependency instead.',
  );
  assert.ok(
    traceroot.peerDependencies?.['@traceroot-ai/pi'],
    'pi should be declared as a peerDependency of traceroot',
  );
  assert.equal(
    traceroot.peerDependenciesMeta?.['@traceroot-ai/pi']?.optional,
    true,
    'the pi peerDependency must be marked optional so consumers who do not use it are not ' +
      'required to install it',
  );
});
