#!/bin/bash
# Tracks file edits for downstream hooks (build checker, test reminder)
# Receives JSON on stdin with tool_name and tool_input.file_path

LOGFILE="$CLAUDE_PROJECT_DIR/.claude/hooks/.edit-log"

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')
TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name // empty')

if [ -n "$FILE_PATH" ]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $TOOL_NAME $FILE_PATH" >> "$LOGFILE"
fi
