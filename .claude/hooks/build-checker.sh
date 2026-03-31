#!/bin/bash
# Runs build validation and benchmark reminders on Stop if relevant files were edited.
# Reads .edit-log (written by file-edit-tracker.sh) and clears it after processing.

LOGFILE="$CLAUDE_PROJECT_DIR/.claude/hooks/.edit-log"

if [ ! -f "$LOGFILE" ]; then
    exit 0
fi

# Snapshot and clear the log atomically to avoid races
SNAPSHOT=$(cat "$LOGFILE" 2>/dev/null)
rm -f "$LOGFILE"

if [ -z "$SNAPSHOT" ]; then
    exit 0
fi

# Check for source file edits
HAS_C=$(echo "$SNAPSHOT" | grep -c '\.c$\|\.h$\|\.metal$' || echo 0)
HAS_PY=$(echo "$SNAPSHOT" | grep -c '\.py$' || echo 0)
HAS_INFERENCE=$(echo "$SNAPSHOT" | grep -c 'src/\|scripts/\|\.metal$' || echo 0)

if [ "$HAS_C" -gt 0 ] && [ -f "$CLAUDE_PROJECT_DIR/Makefile" ] || [ -f "$CLAUDE_PROJECT_DIR/CMakeLists.txt" ]; then
    echo "C/Metal files were edited. Run 'make build' to verify compilation."
fi

if [ "$HAS_PY" -gt 0 ]; then
    echo "Python files were edited. Run 'make test' if tests exist."
fi

if [ "$HAS_INFERENCE" -gt 0 ]; then
    echo "Inference-related files were modified. Consider running benchmarks (make bench) to measure impact."
fi
