#!/bin/bash
# Tracks file edits for downstream hooks (build checker, benchmark reminder).
# Receives JSON on stdin with tool_name and tool_input.file_path.

LOGFILE="$CLAUDE_PROJECT_DIR/.claude/hooks/.edit-log"

INPUT=$(cat)

# Extract file path — use jq if available, fall back to grep/sed
if command -v jq >/dev/null 2>&1; then
    FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')
else
    FILE_PATH=$(echo "$INPUT" | grep -o '"file_path"[[:space:]]*:[[:space:]]*"[^"]*"' | head -1 | sed 's/.*"file_path"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/')
fi

if [ -n "$FILE_PATH" ]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $FILE_PATH" >> "$LOGFILE"
fi
