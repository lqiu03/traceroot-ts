/**
 * Shared UTF-16 surrogate-pair-safe truncation boundary check.
 *
 * Both spans.ts (tool I/O JSON payloads) and span-name.ts (privacy-safe tool
 * span names) need to cap a string at a UTF-16 code-unit length without ever
 * splitting a surrogate pair — doing so would leave a lone high surrogate in
 * the output and corrupt the UTF-8 an OTLP/proto collector requires. This is
 * the single implementation of that boundary math; each caller appends
 * whatever suffix fits its context (an ellipsis, a "…[truncated]" marker, or
 * nothing at all) on top of the raw sliced text this returns.
 */
export function sliceSurrogateSafe(text: string, maxLen: number): string {
  if (text.length <= maxLen) return text;
  let cut = maxLen;
  const code = text.charCodeAt(cut - 1);
  if (code >= 0xd800 && code <= 0xdbff) {
    // High surrogate sitting right at the cut boundary — back off one so we
    // never emit a lone surrogate.
    cut -= 1;
  }
  return text.slice(0, cut);
}
