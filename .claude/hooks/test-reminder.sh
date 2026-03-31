#!/bin/bash
# Reminds to run benchmarks if inference-related files were edited

LOGFILE="$CLAUDE_PROJECT_DIR/.claude/hooks/.edit-log"

if [ ! -f "$LOGFILE" ]; then
    exit 0
fi

HAS_INFERENCE=$(grep -c 'src/\|scripts/\|*.metal' "$LOGFILE" 2>/dev/null || echo 0)

if [ "$HAS_INFERENCE" -gt 0 ]; then
    echo "Inference-related files were modified. Consider running benchmarks to measure impact."
fi
