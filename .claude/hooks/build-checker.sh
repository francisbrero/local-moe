#!/bin/bash
# Runs build validation on Stop if C/Metal/Python files were edited
# Checks .edit-log for relevant file types

LOGFILE="$CLAUDE_PROJECT_DIR/.claude/hooks/.edit-log"

if [ ! -f "$LOGFILE" ]; then
    exit 0
fi

# Check if any source files were edited
HAS_C=$(grep -c '\.c$\|\.h$\|\.metal$' "$LOGFILE" 2>/dev/null || echo 0)
HAS_PY=$(grep -c '\.py$' "$LOGFILE" 2>/dev/null || echo 0)

if [ "$HAS_C" -gt 0 ] && [ -f "$CLAUDE_PROJECT_DIR/Makefile" ]; then
    echo "C/Metal files were edited. Run 'make build' to verify compilation."
fi

if [ "$HAS_PY" -gt 0 ]; then
    echo "Python files were edited. Run 'uv run python -m pytest' if tests exist."
fi

# Clear log after check
rm -f "$LOGFILE"
