/**
 * Shared UTF-16 surrogate-pair-safe truncation boundary check.
 *
 * Both of spans.ts's callers — the tool I/O JSON payloads and the
 * privacy-safe tool span names (describeToolCallSpan) — need to cap a string
 * at a UTF-16 code-unit length without ever
 * splitting a surrogate pair — doing so would leave a lone high surrogate in
 * the output and corrupt the UTF-8 an OTLP/proto collector requires. This is
 * the single implementation of that boundary math; each caller appends
 * whatever suffix fits its context (an ellipsis, a "…[truncated]" marker, or
 * nothing at all) on top of the raw sliced text this returns.
 *
 * Contract for maxLen <= 0: treated as "cap to nothing" and returns ''.
 * Without this guard, a negative maxLen would fall through to
 * `text.slice(0, cut)` with a negative `cut`, which slices from the END of
 * the string (e.g. 'abcdefghij'.slice(0, -3) === 'abcdefg') — the opposite
 * of capping. Both current call sites pass hardcoded positive literals, but
 * this is an exported, reusable primitive, so a future caller computing
 * maxLen dynamically (e.g. a remaining-budget calculation that can underflow
 * below 0) must get a clear, safe cap rather than silently oversized output.
 */
export function sliceSurrogateSafe(text: string, maxLen: number): string {
  if (maxLen <= 0) return '';
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
