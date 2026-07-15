/**
 * Lens: MODULE_NOT_FOUND from a BROKEN transitive dep is not mislabeled "not
 * installed".
 *
 * wirePiCodingAgentInstrumentation() require()s the optional @traceroot-ai/pi
 * peer and, on failure, must distinguish "the peer isn't installed" (benign)
 * from "the peer IS installed but broken" (a real defect whose diagnostic must
 * surface). The old code keyed that decision solely on err.code ===
 * 'MODULE_NOT_FOUND' — but Node throws the identical code when @traceroot-ai/pi
 * is present and one of ITS OWN require() calls fails to resolve a transitive
 * dependency. That case was wrongly reported as "not installed", burying the
 * real error.
 *
 * This drives a GENUINELY broken package, not a hand-fabricated error object:
 * it writes a real @traceroot-ai/pi whose index.js require()s a module that
 * does not exist, so Node's own loader produces the exact MODULE_NOT_FOUND
 * (with the transitive specifier in the message and a populated requireStack)
 * that a real broken install throws. The fix must route this through the
 * "installed but broken" branch — proving the distinguishing signal is the
 * missing SPECIFIER, not the mere presence of a MODULE_NOT_FOUND code (both
 * cases carry the code, and both carry a non-empty requireStack).
 */
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

function runScript(script: string) {
  const pkgDir = process.cwd();
  const tsxLoader = pathToFileURL(
    path.join(pkgDir, 'node_modules', 'tsx', 'dist', 'loader.mjs'),
  ).href;
  return spawnSync(process.execPath, ['--import', tsxLoader, '-e', script], {
    cwd: pkgDir,
    encoding: 'utf8',
  });
}

describe('wireInstrumentations() piCodingAgent broken-transitive-dep diagnosis', () => {
  it('reports "installed but broken" (not "not installed") when pi is present but a transitive require fails', () => {
    const result = runScript(`
      const fs = require('node:fs');
      const os = require('node:os');
      const path = require('node:path');
      const Module = require('node:module');

      // A REAL, installed-but-broken @traceroot-ai/pi: its entry point requires
      // a module that does not exist, so Node's own loader throws a genuine
      // MODULE_NOT_FOUND whose message names the TRANSITIVE specifier (not
      // @traceroot-ai/pi) and whose requireStack lists the pi file. This is the
      // exact shape a broken install produces — nothing about the error is
      // fabricated by the test.
      const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'pi-broken-transitive-'));
      const entry = path.join(dir, 'index.js');
      fs.writeFileSync(entry, "require('__traceroot_missing_transitive_dep__');\\n");

      const orig = Module._load;
      Module._load = function(request, ...rest) {
        if (request === '@traceroot-ai/pi') {
          // Delegate to the REAL loader against the broken package's real path,
          // so the MODULE_NOT_FOUND (and its requireStack) is produced by Node,
          // exactly as it would be for a genuine broken transitive dependency.
          return orig.apply(this, [entry, ...rest]);
        }
        return orig.apply(this, [request, ...rest]);
      };

      const warnings = [];
      console.warn = (...args) => { warnings.push(args.join(' ')); };
      const { wireInstrumentations } = require('./src/instrumentation.ts');
      wireInstrumentations({ piCodingAgent: { AgentSession: class {} } });
      console.log(JSON.stringify(warnings));
    `);
    assert.equal(result.status, 0, result.stderr || result.stdout);
    const warnings = JSON.parse(result.stdout.trim().split('\n').pop() ?? '[]') as string[];

    // A broken-but-present install (MODULE_NOT_FOUND for a TRANSITIVE dep) must
    // NOT be mislabeled "not installed"...
    assert.ok(
      !warnings.some((w) => w.includes('is not installed')),
      'a broken transitive dependency must not be reported as "not installed"',
    );
    // ...it must take the "installed but broken" branch...
    assert.ok(
      warnings.some((w) => w.includes('failed to load')),
      'expected a "failed to load" diagnostic for a broken transitive dependency',
    );
    // ...and the real underlying error (the missing transitive specifier) must
    // be surfaced, not discarded.
    assert.ok(
      warnings.some((w) => w.includes('__traceroot_missing_transitive_dep__')),
      'expected the original transitive-resolution error to be surfaced',
    );
  });
});
