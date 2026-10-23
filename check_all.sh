#!/usr/bin/env bash
# Run compile.py on every open problem through submit.py (compile + ptxas + GPU run + compare)
# and print a pass/fail summary.
#
#   ./check_all.sh                  # all 15 problems
#   ./check_all.sh r01_array_sum    # a subset
#   ./check_all.sh --dir tests/problems   # every problem folder under another directory
#   PYTHON=python3.12 ./check_all.sh
#
# Exit code 0 only when every selected problem passes.

set -u
cd "$(dirname "$0")"

if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x .venv/bin/python ]]; then PYTHON=.venv/bin/python; else PYTHON=python3; fi
fi

root=problems
if [[ ${1:-} == "--dir" ]]; then
    root=${2:?--dir needs a directory}
    shift 2
fi

problems=()
if [[ $# -gt 0 ]]; then
    for p in "$@"; do problems+=("$root/$p"); done
else
    for d in "$root"/*/; do
        [[ -f "$d/meta.json" ]] && problems+=("${d%/}")
    done
fi

pass=0
fail=0
failed=()
logdir=$(mktemp -d)

for dir in "${problems[@]}"; do
    p=$(basename "$dir")
    log="$logdir/$p.log"
    start=$(date +%s)
    if "$PYTHON" submit.py "$dir" --compiler compile.py >"$log" 2>&1; then
        status="PASS"
        pass=$((pass + 1))
    else
        status="FAIL"
        fail=$((fail + 1))
        failed+=("$p")
    fi
    secs=$(( $(date +%s) - start ))
    detail=$(grep -E "PASS|FAIL|rejected|failed|error" "$log" | head -1 | sed 's/^\[[^]]*\] //')
    printf "%-20s %s  (%ss)  %s\n" "$p" "$status" "$secs" "$detail"
    if [[ $status == "FAIL" ]]; then
        sed 's/^/    | /' "$log" | head -40
    fi
done

echo "----------------------------------------"
echo "passed $pass / $((pass + fail))"
if [[ $fail -gt 0 ]]; then
    echo "failed: ${failed[*]}"
    echo "logs: $logdir"
    exit 1
fi
rm -rf "$logdir"
exit 0
