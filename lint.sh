#!/usr/bin/env bash

set -e

# Default values
FIX="--fix"
FORMAT_CHECK=""
RUN_PY=true

OVERALL_EXIT_CODE=0

### ---------- Python Linting ----------

if [ "$RUN_PY" = true ]; then
    PY_SRC_DIRS=(src tests)

    echo "Running ruff format in:" "${PY_SRC_DIRS[@]}"
    uv run ruff format $FORMAT_CHECK "${PY_SRC_DIRS[@]}" || OVERALL_EXIT_CODE=1

    echo "Running ruff check in:" "${PY_SRC_DIRS[@]}"
    uv run ruff check $FIX "${PY_SRC_DIRS[@]}" || OVERALL_EXIT_CODE=1

    CODESPELL_FLAG="--write-changes"
    if [ "$CI_MODE" = true ]; then
        CODESPELL_FLAG=""
    fi

    for DIR in "${PY_SRC_DIRS[@]}"; do
        echo "Running codespell in: " "${DIR}"
        uv run codespell $CODESPELL_FLAG "${DIR}/**/*.py" || OVERALL_EXIT_CODE=1
    done

    echo "Running mypy in:" "${PY_SRC_DIRS[@]}"
    uv run mypy "${PY_SRC_DIRS[@]}" || OVERALL_EXIT_CODE=1

    echo "Running vulture in: src"
    uv run vulture --min-confidence 90 \
        --exclude "src/crystal/calculators/gpaw/external_runtime/*" \
        src || OVERALL_EXIT_CODE=1
fi

### ---------- Done ----------

if [ "$OVERALL_EXIT_CODE" == 0 ]; then
    echo "✅ All checks passed!"
else
    echo "❌ Linting failed."
fi

exit $OVERALL_EXIT_CODE
