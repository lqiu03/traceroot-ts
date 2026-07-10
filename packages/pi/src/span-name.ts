/**
 * Privacy-safe tool span naming, ported from traceroot-pi-extension
 * (the Pi CLI extension). Never emits a full file path — basename only,
 * handling both / and \ separators — and never emits more than
 * MAX_BASH_NAME chars of a bash command, truncated without splitting a
 * UTF-16 surrogate pair (which would corrupt the UTF-8 an OTLP/proto
 * collector requires).
 *
 * Truncating a bash command to 60 chars can still leak the start of a
 * pasted secret (e.g. an Authorization header) even when captureToolIo is
 * off — that tradeoff is deliberate: the alternative (no name at all)
 * makes traces far less useful, and the full command is only captured
 * as a span attribute when captureToolIo is explicitly enabled.
 */
import { win32 } from 'node:path';

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

function truncateSurrogateSafe(text: string, maxLen: number): string {
  if (text.length <= maxLen) return text;
  let cut = maxLen;
  const code = text.charCodeAt(cut - 1);
  if (code >= 0xd800 && code <= 0xdbff) {
    // High surrogate sitting right at the cut boundary — back off one so we
    // never emit a lone surrogate, which would corrupt UTF-8 on export.
    cut -= 1;
  }
  return `${text.slice(0, cut)}…`;
}

export function describeToolCallSpan(toolName: string, args: unknown): string {
  if (args && typeof args === 'object') {
    const a = args as Record<string, unknown>;
    const pathLike = firstPathArgument(a);
    if (pathLike) return `${toolName}: ${basename(pathLike)}`;
    if (toolName === 'bash' && typeof a.command === 'string' && a.command) {
      const cmd = a.command.replace(/\s+/g, ' ').trim();
      if (cmd) return `bash: ${truncateSurrogateSafe(cmd, MAX_BASH_NAME)}`;
    }
  }
  return toolName;
}
