#!/bin/bash
# Post-edit format hook: auto-formats TS/JS files with prettier.
# eslint is intentionally NOT run here — it's slow with typescript-eslint and
# already runs via lint-staged on commit and in CI. Use a slash command for
# on-demand eslint feedback.

INPUT=$(cat)

# Guard: jq is required to parse hook input
if ! command -v jq &>/dev/null; then
  exit 0
fi

FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

if [ -z "$FILE_PATH" ] || [ ! -f "$FILE_PATH" ]; then
  exit 0
fi

PROJECT_ROOT="${CLAUDE_PROJECT_DIR:-$(pwd)}"

case "$FILE_PATH" in
  *.ts|*.tsx|*.js|*.jsx|*.json|*.md|*.yml|*.yaml) ;;
  *) exit 0 ;;
esac

# Skip generated/vendored paths
if [[ "$FILE_PATH" == */node_modules/* || "$FILE_PATH" == */dist/* ]]; then
  exit 0
fi

PRETTIER="$PROJECT_ROOT/node_modules/.bin/prettier"
if [ ! -x "$PRETTIER" ]; then
  exit 0
fi

(cd "$PROJECT_ROOT" && "$PRETTIER" --write "$FILE_PATH" >/dev/null 2>&1)
exit 0
