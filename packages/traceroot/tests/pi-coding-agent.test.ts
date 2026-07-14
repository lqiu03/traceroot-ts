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

describe('wireInstrumentations() piCodingAgent lazy loading', () => {
  it('does nothing when key not set', () => {
    const result = runScript(`
      const Module = require('node:module');
      const orig = Module._load;
      Module._load = function(request, ...rest) {
        if (request === '@traceroot-ai/pi') throw new Error('should not load pi package');
        return orig.apply(this, [request, ...rest]);
      };
      const { wireInstrumentations } = require('./src/instrumentation.ts');
      wireInstrumentations({});
      console.log('OK');
    `);
    assert.equal(result.status, 0, result.stderr || result.stdout);
    assert.ok(result.stdout.includes('OK'));
  });

  it('does not load on the undefined/auto path', () => {
    const result = runScript(`
      const Module = require('node:module');
      const orig = Module._load;
      Module._load = function(request, ...rest) {
        if (request === '@traceroot-ai/pi') throw new Error('should not load pi package');
        return orig.apply(this, [request, ...rest]);
      };
      const { wireInstrumentations } = require('./src/instrumentation.ts');
      wireInstrumentations(undefined);
      console.log('OK');
    `);
    assert.equal(result.status, 0, result.stderr || result.stdout);
    assert.ok(result.stdout.includes('OK'));
  });

  it('falsy-but-present value results in no load', () => {
    const result = runScript(`
      const Module = require('node:module');
      const orig = Module._load;
      Module._load = function(request, ...rest) {
        if (request === '@traceroot-ai/pi') throw new Error('should not load pi package');
        return orig.apply(this, [request, ...rest]);
      };
      const { wireInstrumentations } = require('./src/instrumentation.ts');
      wireInstrumentations({ piCodingAgent: null });
      console.log('OK');
    `);
    assert.equal(result.status, 0, result.stderr || result.stdout);
    assert.ok(result.stdout.includes('OK'));
  });

  it('warns when @traceroot-ai/pi is not installed', () => {
    const result = runScript(`
      const Module = require('node:module');
      const orig = Module._load;
      Module._load = function(request, ...rest) {
        if (request === '@traceroot-ai/pi') {
          // A genuinely-absent module: Node's CJS loader throws an Error whose
          // code is MODULE_NOT_FOUND, which is how "not installed" is told apart
          // from "installed but broken" (see the broken-install test below).
          const err = new Error("Cannot find module '@traceroot-ai/pi'");
          err.code = 'MODULE_NOT_FOUND';
          throw err;
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
    assert.ok(warnings.some((w) => w.includes('@traceroot-ai/pi is not installed')));
  });

  it('surfaces the real error (not "not installed") when @traceroot-ai/pi is present but fails to load', () => {
    const result = runScript(`
      const Module = require('node:module');
      const orig = Module._load;
      Module._load = function(request, ...rest) {
        if (request === '@traceroot-ai/pi') {
          // A package that IS installed but throws on load (bad build, syntax
          // error, or a failing top-level side effect) — NOT a MODULE_NOT_FOUND.
          throw new Error('boom: pi package is broken');
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
    // A broken-but-present install must NOT be mislabeled "not installed"...
    assert.ok(
      !warnings.some((w) => w.includes('is not installed')),
      'a broken install should not be reported as "not installed"',
    );
    // ...and the real diagnostic must be surfaced, not silently discarded.
    assert.ok(
      warnings.some((w) => w.includes('failed to load')),
      'expected a "failed to load" diagnostic for a broken install',
    );
    assert.ok(
      warnings.some((w) => w.includes('boom: pi package is broken')),
      'expected the original load error to be surfaced',
    );
  });

  it('still reports "not installed" for a genuine MODULE_NOT_FOUND', () => {
    const result = runScript(`
      const Module = require('node:module');
      const orig = Module._load;
      Module._load = function(request, ...rest) {
        if (request === '@traceroot-ai/pi') {
          const err = new Error("Cannot find module '@traceroot-ai/pi'");
          err.code = 'MODULE_NOT_FOUND';
          throw err;
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
    assert.ok(warnings.some((w) => w.includes('@traceroot-ai/pi is not installed')));
  });

  it('passes the bare module ref as arg 0 and a plumbed config object as arg 1', () => {
    const result = runScript(`
      const Module = require('node:module');
      const orig = Module._load;
      Module._load = function(request, ...rest) {
        if (request === '@traceroot-ai/pi') {
          return {
            instrumentPiCodingAgent: (...args) => {
              global.__callArgs = args;
            },
          };
        }
        return orig.apply(this, [request, ...rest]);
      };
      const { wireInstrumentations } = require('./src/instrumentation.ts');
      const fakeMod = { AgentSession: class {} };
      wireInstrumentations({ piCodingAgent: fakeMod });
      console.log(JSON.stringify({
        length: global.__callArgs.length,
        sameRef: global.__callArgs[0] === fakeMod,
        configIsObject: typeof global.__callArgs[1] === 'object' && global.__callArgs[1] !== null,
      }));
    `);
    assert.equal(result.status, 0, result.stderr || result.stdout);
    const parsed = JSON.parse(result.stdout.trim().split('\n').pop() ?? '{}') as {
      length: number;
      sameRef: boolean;
      configIsObject: boolean;
    };
    // The bare module is forwarded by reference as the first argument, and a
    // config object is ALWAYS threaded as the second argument (empty of
    // apiKey/baseUrl here, since wireInstrumentations() was called with no
    // initialize()-resolved defaults) so pi is never left to fall back to
    // TRACEROOT_API_KEY alone.
    assert.equal(parsed.length, 2);
    assert.equal(parsed.sameRef, true);
    assert.equal(parsed.configIsObject, true);
  });

  it('threads initialize()-resolved apiKey/baseUrl through to instrumentPiCodingAgent()', () => {
    const result = runScript(`
      const Module = require('node:module');
      const orig = Module._load;
      Module._load = function(request, ...rest) {
        if (request === '@traceroot-ai/pi') {
          return { instrumentPiCodingAgent: (...args) => { global.__piArgs = args; } };
        }
        return orig.apply(this, [request, ...rest]);
      };
      delete process.env.TRACEROOT_API_KEY;
      const { TraceRoot } = require('./src/traceroot.ts');
      const fakeMod = { AgentSession: class {} };
      TraceRoot.initialize({
        apiKey: 'trk_real_key',
        baseUrl: 'https://custom.example.com',
        disableBatch: true,
        instrumentModules: { piCodingAgent: fakeMod },
      });
      console.log(JSON.stringify({
        moduleIsFirst: global.__piArgs[0] === fakeMod,
        apiKey: global.__piArgs[1] && global.__piArgs[1].apiKey,
        baseUrl: global.__piArgs[1] && global.__piArgs[1].baseUrl,
      }));
      process.exit(0);
    `);
    assert.equal(result.status, 0, result.stderr || result.stdout);
    const parsed = JSON.parse(result.stdout.trim().split('\n').pop() ?? '{}') as {
      moduleIsFirst: boolean;
      apiKey: string;
      baseUrl: string;
    };
    // Reproduces the exact plumbing bug: initialize({ apiKey }) with
    // TRACEROOT_API_KEY unset must reach pi as config.apiKey, not vanish.
    assert.equal(parsed.moduleIsFirst, true);
    assert.equal(parsed.apiKey, 'trk_real_key');
    assert.equal(parsed.baseUrl, 'https://custom.example.com');
  });

  it('threads an explicit { module, config } wrapper (captureContent/captureToolIo) through to pi', () => {
    const result = runScript(`
      const Module = require('node:module');
      const orig = Module._load;
      Module._load = function(request, ...rest) {
        if (request === '@traceroot-ai/pi') {
          return { instrumentPiCodingAgent: (...args) => { global.__piArgs = args; } };
        }
        return orig.apply(this, [request, ...rest]);
      };
      delete process.env.TRACEROOT_API_KEY;
      const { TraceRoot } = require('./src/traceroot.ts');
      const fakeMod = { AgentSession: class {} };
      TraceRoot.initialize({
        apiKey: 'trk_real_key',
        disableBatch: true,
        instrumentModules: {
          piCodingAgent: { module: fakeMod, config: { captureContent: false, captureToolIo: false } },
        },
      });
      console.log(JSON.stringify({
        moduleUnwrapped: global.__piArgs[0] === fakeMod,
        captureContent: global.__piArgs[1] && global.__piArgs[1].captureContent,
        captureToolIo: global.__piArgs[1] && global.__piArgs[1].captureToolIo,
        apiKey: global.__piArgs[1] && global.__piArgs[1].apiKey,
      }));
      process.exit(0);
    `);
    assert.equal(result.status, 0, result.stderr || result.stdout);
    const parsed = JSON.parse(result.stdout.trim().split('\n').pop() ?? '{}') as {
      moduleUnwrapped: boolean;
      captureContent: boolean;
      captureToolIo: boolean;
      apiKey: string;
    };
    // The wrapper's inner module is unwrapped to arg 0; its PII-control flags
    // reach pi verbatim; and initialize()'s own apiKey is still merged in as a
    // default the wrapper didn't override.
    assert.equal(parsed.moduleUnwrapped, true);
    assert.equal(parsed.captureContent, false);
    assert.equal(parsed.captureToolIo, false);
    assert.equal(parsed.apiKey, 'trk_real_key');
  });

  it('warns when @traceroot-ai/pi does not export instrumentPiCodingAgent', () => {
    const result = runScript(`
      const Module = require('node:module');
      const orig = Module._load;
      Module._load = function(request, ...rest) {
        if (request === '@traceroot-ai/pi') return {};
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
    assert.ok(warnings.some((w) => w.includes('does not export instrumentPiCodingAgent')));
  });

  it('warns when instrumentPiCodingAgent export is present but not a function', () => {
    const result = runScript(`
      const Module = require('node:module');
      const orig = Module._load;
      Module._load = function(request, ...rest) {
        if (request === '@traceroot-ai/pi') return { instrumentPiCodingAgent: 'not-a-function' };
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
    assert.ok(warnings.some((w) => w.includes('does not export instrumentPiCodingAgent')));
  });

  it('warns without throwing when require resolves to null', () => {
    const result = runScript(`
      const Module = require('node:module');
      const orig = Module._load;
      Module._load = function(request, ...rest) {
        if (request === '@traceroot-ai/pi') return null;
        return orig.apply(this, [request, ...rest]);
      };
      const warnings = [];
      console.warn = (...args) => { warnings.push(args.join(' ')); };
      const { wireInstrumentations } = require('./src/instrumentation.ts');
      wireInstrumentations({ piCodingAgent: { AgentSession: class {} } });
      console.log(JSON.stringify(warnings));
    `);
    // Accessing a property on the null returned from require() throws a
    // TypeError inside the try block. That TypeError carries no
    // code === 'MODULE_NOT_FOUND', so it is correctly classified as an
    // "installed but broken" load failure (the module DID resolve, it's just
    // malformed) rather than "not installed" — and the real error is surfaced.
    // Either way: no crash, and the operator gets a clear, actionable warning.
    assert.equal(result.status, 0, result.stderr || result.stdout);
    const warnings = JSON.parse(result.stdout.trim().split('\n').pop() ?? '[]') as string[];
    assert.ok(warnings.length > 0);
  });

  it('propagates a throw from the delegated instrumentPiCodingAgent() call', () => {
    const result = runScript(`
      const Module = require('node:module');
      const orig = Module._load;
      Module._load = function(request, ...rest) {
        if (request === '@traceroot-ai/pi') {
          return {
            instrumentPiCodingAgent: () => { throw new Error('boom from pi'); },
          };
        }
        return orig.apply(this, [request, ...rest]);
      };
      const { wireInstrumentations } = require('./src/instrumentation.ts');
      wireInstrumentations({ piCodingAgent: { AgentSession: class {} } });
      console.log('SHOULD_NOT_REACH_HERE');
    `);
    assert.notEqual(result.status, 0);
    assert.ok(!result.stdout.includes('SHOULD_NOT_REACH_HERE'));
    assert.ok(result.stderr.includes('boom from pi'));
  });

  it('wires alongside another instrumentModules entry without interference', () => {
    const result = runScript(`
      const Module = require('node:module');
      const orig = Module._load;
      Module._load = function(request, ...rest) {
        if (request === '@arizeai/openinference-instrumentation-anthropic') {
          return {
            AnthropicInstrumentation: class {
              manuallyInstrument() { global.__anthCalled = true; }
            },
          };
        }
        if (request === '@traceroot-ai/pi') {
          return {
            instrumentPiCodingAgent: (...args) => { global.__piArgs = args; },
          };
        }
        return orig.apply(this, [request, ...rest]);
      };
      const { wireInstrumentations } = require('./src/instrumentation.ts');
      const fakeMod = { AgentSession: class {} };
      wireInstrumentations({ anthropic: { name: 'mock' }, piCodingAgent: fakeMod });
      console.log(JSON.stringify({
        anthCalled: global.__anthCalled === true,
        piArgsLength: global.__piArgs.length,
        piSameRef: global.__piArgs[0] === fakeMod,
      }));
    `);
    assert.equal(result.status, 0, result.stderr || result.stdout);
    const parsed = JSON.parse(result.stdout.trim().split('\n').pop() ?? '{}') as {
      anthCalled: boolean;
      piArgsLength: number;
      piSameRef: boolean;
    };
    assert.equal(parsed.anthCalled, true);
    assert.equal(parsed.piArgsLength, 2);
    assert.equal(parsed.piSameRef, true);
  });

  it('delegates on every call -- core does not dedup', () => {
    const result = runScript(`
      const Module = require('node:module');
      const orig = Module._load;
      Module._load = function(request, ...rest) {
        if (request === '@traceroot-ai/pi') {
          return {
            instrumentPiCodingAgent: () => { global.__callCount = (global.__callCount ?? 0) + 1; },
          };
        }
        return orig.apply(this, [request, ...rest]);
      };
      const { wireInstrumentations } = require('./src/instrumentation.ts');
      const fakeMod = { AgentSession: class {} };
      wireInstrumentations({ piCodingAgent: fakeMod });
      wireInstrumentations({ piCodingAgent: fakeMod });
      console.log(JSON.stringify({ callCount: global.__callCount }));
    `);
    assert.equal(result.status, 0, result.stderr || result.stdout);
    const parsed = JSON.parse(result.stdout.trim().split('\n').pop() ?? '{}') as {
      callCount: number;
    };
    assert.equal(parsed.callCount, 2);
  });

  it('initialize(): missing pi warns but leaves isInitialized() true', () => {
    const result = runScript(`
      const Module = require('node:module');
      const orig = Module._load;
      Module._load = function(request, ...rest) {
        if (request === '@traceroot-ai/pi') {
          const err = new Error("Cannot find module '@traceroot-ai/pi'");
          err.code = 'MODULE_NOT_FOUND';
          throw err;
        }
        return orig.apply(this, [request, ...rest]);
      };
      const warnings = [];
      console.warn = (...args) => { warnings.push(args.join(' ')); };
      const { TraceRoot } = require('./src/traceroot.ts');
      TraceRoot.initialize({
        apiKey: 'test-key',
        disableBatch: true,
        instrumentModules: { piCodingAgent: { AgentSession: class {} } },
      });
      console.log(JSON.stringify({
        initialized: TraceRoot.isInitialized(),
        warned: warnings.some((w) => w.includes('is not installed')),
      }));
      process.exit(0);
    `);
    assert.equal(result.status, 0, result.stderr || result.stdout);
    const parsed = JSON.parse(result.stdout.trim().split('\n').pop() ?? '{}') as {
      initialized: boolean;
      warned: boolean;
    };
    assert.equal(parsed.initialized, true);
    assert.equal(parsed.warned, true);
  });
});
