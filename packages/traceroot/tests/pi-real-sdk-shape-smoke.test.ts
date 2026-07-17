/**
 * Lens: the private @earendil-works/pi-coding-agent internals that
 * src/instrumentation.ts and src/types.ts document by hand.
 *
 * Those files cite specific fields and methods of the real, installed SDK --
 * the private `_eventListeners` listener array and the standalone
 * steer()/followUp()/dispose() entry points -- all read straight out of the
 * installed source, none of them part of a stable public contract. The peer
 * range (>=0.79.0 <1) permits a patch bump that could silently rename or drop
 * any of them, invalidating those comments (and the behavior that depends on
 * them) with zero failures anywhere else in the suite, because every other
 * test mocks a hand-rolled FakeAgentSession.
 *
 * The `isStreaming`/`extensionRunner` getters, `hasExtensionHandlers()`, and
 * ExtensionRunner's `getCommand()` were previously verified here too --
 * proto.prompt's now-deleted prompt-queue "should this call skip the FIFO"
 * heuristic (shouldSkipQueue, prompt-queue.ts) was the only reader of any of
 * them. Now that the root span is anchored on prompt()'s own promise window
 * instead of a per-session input-attribution queue (see instrumentation.ts's
 * module header), instrumentation.ts no longer reads any of those four, so
 * asserting their shape here no longer protects anything this package
 * depends on.
 *
 * This is the ONLY test that imports the REAL package, so a future SDK bump
 * that changes any of these shapes fails loudly here instead of silently
 * shipping broken instrumentation. It asserts shape only -- it never
 * constructs a live AgentSession (the real constructor needs a full
 * agent/session/settings runtime) or performs any agent work.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';

// @earendil-works/pi-coding-agent is ESM-only (its package.json exports only an
// `import` condition), so it is loaded via a dynamic import() -- which always
// goes through Node's ESM resolver and matches that condition -- rather than a
// static import that this CommonJS-compiled test file would turn into a
// require() the package deliberately does not support. Cached so the (heavy)
// module graph is only evaluated once across the tests below. Accessing exports
// off a loose record means a REMOVED export surfaces as a clear assertion
// failure below, not an opaque module-link error.
type RealSdk = {
  AgentSession?: { prototype: Record<string, unknown> };
};
let cachedSdk: Promise<RealSdk> | undefined;
function loadRealSdk(): Promise<RealSdk> {
  if (!cachedSdk) {
    cachedSdk = import('@earendil-works/pi-coding-agent') as unknown as Promise<RealSdk>;
  }
  return cachedSdk;
}

test('the real AgentSession still exposes every prototype method instrumentPiCodingAgent patches or reads', async () => {
  const sdk = await loadRealSdk();
  const AgentSession = sdk.AgentSession;
  assert.equal(typeof AgentSession, 'function', 'AgentSession must still be an exported class');
  const proto = AgentSession!.prototype;

  // prompt + subscribe are REQUIRED by instrumentation.ts's install guard --
  // it disables itself entirely if either is missing.
  assert.equal(
    typeof proto.prompt,
    'function',
    'AgentSession.prototype.prompt must exist -- the whole patch layer keys off it',
  );
  assert.equal(
    typeof proto.subscribe,
    'function',
    'AgentSession.prototype.subscribe must exist -- the entire span tree is built from one subscribe() listener',
  );

  // steer/followUp/dispose are optional-guarded in the patch layer, but
  // instrumentation.ts's comments assert they ARE present on the real SDK; if
  // that stops being true the comments (and the behavior they justify) are
  // stale.
  assert.equal(
    typeof proto.steer,
    'function',
    'AgentSession.prototype.steer must exist -- patched as a standalone first-interaction entry point',
  );
  assert.equal(
    typeof proto.followUp,
    'function',
    'AgentSession.prototype.followUp must exist -- patched as a standalone first-interaction entry point',
  );
  assert.equal(
    typeof proto.dispose,
    'function',
    'AgentSession.prototype.dispose must exist -- patched to force-close in-flight spans on teardown',
  );
});

test('the real AgentSession.subscribe() still pushes onto a private _eventListeners array its unsubscribe closure splices back out', async () => {
  const sdk = await loadRealSdk();
  // instrumentation.ts's module header and its "no cleanup needed" design rest
  // entirely on this: subscribe(listener) pushes onto a private array field
  // literally named _eventListeners, and dispose() reassigns that same field to
  // [] to stop delivery -- with no per-listener unsubscribe() call from our
  // side. A bare prototype instance (the real constructor needs a full runtime,
  // so it is deliberately bypassed) with the field pre-seeded lets us drive the
  // REAL subscribe()/unsubscribe() and confirm they still operate on a field of
  // exactly that name; a rename leaves this seeded array untouched and fails.
  const proto = sdk.AgentSession!.prototype as unknown as { subscribe(l: unknown): () => void };
  const instance = Object.create(proto) as {
    _eventListeners: unknown[];
    subscribe(l: unknown): () => void;
  };
  instance._eventListeners = [];
  const listener = (): void => {};

  const unsubscribe = instance.subscribe(listener);
  assert.equal(typeof unsubscribe, 'function', 'subscribe() must return an unsubscribe closure');
  assert.ok(
    instance._eventListeners.includes(listener),
    "subscribe() must push the listener onto a private array field named _eventListeners -- instrumentation.ts's no-cleanup design depends on this exact field name",
  );

  unsubscribe();
  assert.ok(
    !instance._eventListeners.includes(listener),
    'the unsubscribe closure returned by subscribe() must splice the listener back out of _eventListeners',
  );
});
