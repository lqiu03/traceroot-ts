/**
 * Privacy-safe tool span naming, ported from traceroot-pi-extension
 * (the Pi CLI extension). Never emits a full file path — basename only,
 * handling both / and \ separators — and never emits more than
 * MAX_BASH_NAME chars of a bash command OR of a path-like argument's
 * basename, truncated without splitting a UTF-16 surrogate pair (which
 * would corrupt the UTF-8 an OTLP/proto collector requires).
 *
 * The basename cap matters because basename() only strips path separators:
 * a path-like argument with none at all (or whose final segment is itself
 * huge — e.g. a hallucinated or adversarial tool-call argument) would
 * otherwise pass through completely unbounded, the same OTLP-payload-bloat
 * and leak risk the bash-command cap below exists to guard against.
 *
 * Truncating a bash command (or an unseparated "path") to 60 chars can
 * still leak the start of a pasted secret (e.g. an Authorization header)
 * even when captureToolIo is off — that tradeoff is deliberate: the
 * alternative (no name at all) makes traces far less useful, and the full
 * value is only captured as a span attribute when captureToolIo is
 * explicitly enabled.
 *
 * A non-empty path-like argument can still basename() down to an EMPTY
 * string — a root or drive-only reference such as "/", "\\", "///", "C:\\",
 * or "C:/" has no filename component to keep. That is an ordinary,
 * non-adversarial tool call (e.g. listing a root directory), so it falls
 * through to the bare tool name (or the bash-command branch, if the same
 * args object also carries a command) instead of emitting a dangling
 * "toolName: " with nothing after the colon.
 */
import { win32 } from 'node:path';
import { sliceSurrogateSafe } from './surrogate-safe';

const basename = win32.basename;

const MAX_BASH_NAME = 60;

const TOOL_PATH_ARGUMENT_KEYS = [
  'path',
  'file',
  'filePath',
  'file_path',
  'filename',
  'target',
] as const;

function firstPathArgument(args: Record<string, unknown>): string | undefined {
  for (const key of TOOL_PATH_ARGUMENT_KEYS) {
    const value = args[key];
    if (typeof value === 'string' && value) return value;
  }
  return undefined;
}

// Thin wrapper around the shared ./surrogate-safe boundary cut: appends the
// ellipsis marker this file's span names use, but only when truncation
// actually happened (an untruncated name gets no trailing "…").
function truncateSurrogateSafe(text: string, maxLen: number): string {
  if (text.length <= maxLen) return text;
  return `${sliceSurrogateSafe(text, maxLen)}…`;
}

export function describeToolCallSpan(toolName: string, args: unknown): string {
  if (args && typeof args === 'object') {
    const a = args as Record<string, unknown>;
    // Bash calls are checked first and exclusively by their command: tool
    // schemas can attach an incidental path/file/target-like argument
    // alongside `command` (e.g. a cwd or target field), and the generic
    // path-like check below has no bash-specific exemption. Resolving the
    // command here, before that check, ensures such an argument never
    // shadows the actual command being run.
    if (toolName === 'bash' && typeof a.command === 'string' && a.command) {
      const cmd = a.command.replace(/\s+/g, ' ').trim();
      if (cmd) return `bash: ${truncateSurrogateSafe(cmd, MAX_BASH_NAME)}`;
    }
    const pathLike = firstPathArgument(a);
    if (pathLike) {
      // basename() only strips path separators — a value with none at all
      // (or whose final segment is itself huge) passes through completely
      // unchanged. Truncate it the same way the bash branch above truncates
      // a command, so an untrusted/hallucinated "path"-like argument can't
      // inflate the span NAME without bound the same way a raw bash command
      // could without MAX_BASH_NAME.
      const base = basename(pathLike);
      // A non-empty, truthy path-like value can still basename() down to an
      // EMPTY string — e.g. "/", "\\", "///", "C:\\", "C:/": a root or
      // drive-only reference with no filename component to keep. That is a
      // perfectly ordinary, non-adversarial tool call (listing/reading a
      // root directory), not an edge case worth degrading gracelessly for:
      // falling through here (instead of returning) avoids emitting a
      // dangling "toolName: " with nothing after the colon, matching the
      // args.path === '' behavior a few lines below in firstPathArgument.
      if (base) {
        return `${toolName}: ${truncateSurrogateSafe(base, MAX_BASH_NAME)}`;
      }
    }
  }
  return toolName;
}
