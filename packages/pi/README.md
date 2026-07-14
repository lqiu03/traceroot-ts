# @traceroot-ai/pi

[![Y Combinator][y-combinator-image]][y-combinator-url]
[![License][license-image]][license-url]
[![npm][npm-image]][npm-url]
[![X (Twitter)][twitter-image]][twitter-url]
[![Discord][discord-image]][discord-url]
[![Documentation][docs-image]][docs-url]

TraceRoot observability instrumentation for the [Pi coding agent](https://pi.dev) SDK. Sends traces from programmatically-embedded Pi agent sessions (`AgentSession.prompt()`) to TraceRoot with full span-type semantics (agent, LLM, tool).

This package instruments Pi's SDK for developers embedding `@earendil-works/pi-coding-agent` in their own Node/TypeScript app. If you're tracing the interactive `pi` CLI tool itself, see [`@traceroot-ai/pi-extension`](https://github.com/traceroot-ai/traceroot-pi-extension) instead.

`@earendil-works/pi-coding-agent` is ESM-only (no CommonJS `require()` export) and requires Node >=22.19. Your consuming code must use `import`.

## Installation

```bash
npm install @traceroot-ai/pi @earendil-works/pi-coding-agent
```

## Usage

```ts
import * as pi from "@earendil-works/pi-coding-agent";
import { AuthStorage, createAgentSession, ModelRegistry } from "@earendil-works/pi-coding-agent";
import { instrumentPiCodingAgent } from "@traceroot-ai/pi";

// Instrument BEFORE creating any session.
instrumentPiCodingAgent(pi, {
  apiKey: process.env.TRACEROOT_API_KEY,
});

const authStorage = AuthStorage.create();
const modelRegistry = ModelRegistry.create(authStorage);
const [model] = await modelRegistry.getAvailable();

const { session } = await createAgentSession({ model, authStorage, modelRegistry });

try {
  await session.prompt("Fix the failing test in src/math.ts");
} finally {
  session.dispose();
}
```

Use `createAgentSession()` — Pi's own documented SDK entry point — rather than constructing `AgentSession` directly; its constructor requires assembling several internal objects (`Agent`, `SessionManager`, `SettingsManager`, `ResourceLoader`) that `createAgentSession()` builds for you. `instrumentPiCodingAgent()` patches `AgentSession.prototype`, so it instruments sessions built either way.

If you also use TraceRoot's core SDK, `instrumentPiCodingAgent()` must run after `TraceRoot.initialize()` (or any other global OpenTelemetry provider registration) in the same process to attach to that shared pipeline; if it runs first, it commits to its own private export pipeline for the life of the process and will not pick up a provider registered later.

## Configuration

| Option | Default | Description |
|---|---|---|
| `apiKey` | `TRACEROOT_API_KEY` env | TraceRoot API key |
| `baseUrl` | `https://app.traceroot.ai` | TraceRoot backend URL (`TRACEROOT_HOST_URL` env) |
| `captureContent` | `true` | Capture prompt/response text as `input.value`/`output.value` on AGENT and LLM spans |
| `captureToolIo` | `true` | Capture tool call args/results as `input.value`/`output.value` on TOOL spans |

## Documentation

See the [TraceRoot Docs](https://traceroot.ai/docs/tracing/get-started) for details.

<!-- Links -->

[discord-image]: https://img.shields.io/discord/1395844148568920114?logo=discord&labelColor=%235462eb&logoColor=%23f5f5f5&color=%235462eb
[discord-url]: https://discord.gg/tPyffEZvvJ
[docs-image]: https://img.shields.io/badge/docs-traceroot.ai-0dbf43
[docs-url]: https://traceroot.ai/docs/tracing/get-started
[license-image]: https://img.shields.io/badge/License-Apache%202.0-blue.svg
[license-url]: https://opensource.org/licenses/Apache-2.0
[npm-image]: https://img.shields.io/npm/v/%40traceroot-ai%2Fpi?label=%40traceroot-ai%2Fpi&labelColor=CB3837&color=555555
[npm-url]: https://www.npmjs.com/package/@traceroot-ai/pi
[twitter-image]: https://img.shields.io/twitter/follow/TracerootAI
[twitter-url]: https://x.com/TracerootAI
[y-combinator-image]: https://img.shields.io/badge/Combinator-S25-orange?logo=ycombinator&labelColor=white
[y-combinator-url]: https://www.ycombinator.com/companies/traceroot-ai
