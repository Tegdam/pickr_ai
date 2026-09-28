#!/usr/bin/env bash
# Run the session anchor: one unchanged reference run, date-stamped so each
# session gets its own sweep_id, then print the whole anchor series so an
# anomalous session is visible immediately rather than in analysis months later.
#
# Usage:  bash bench/scripts/anchor.sh
# Run it FIRST in any session that will produce data worth trusting.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

STAMP=$(date -u +%Y%m%dT%H%MZ)
SRC=bench/configs/anchor.yaml
GEN=$(mktemp /tmp/anchor-XXXXXX.yaml)

# The template's sweep_id is a placeholder; the date stamp makes each session's
# anchor its own sweep (the runner refuses to reuse an existing sweep dir).
sed "s|^sweep_id: anchor$|sweep_id: anchor-${STAMP}|" "$SRC" > "$GEN"

echo "=== anchor ${STAMP} ==="
env/bin/python -m bench.runner check-env --sweep "$GEN" >/dev/null 2>&1
echo "check-env exit=$?"
env/bin/python -m bench.runner run "$GEN" --results bench/results 2>&1 \
  | grep -viE 'pytorch was not found|is not valid: failed to fetch metadata|No blkio throttle'
RC=${PIPESTATUS[0]}
echo "run exit=$RC"
rm -f "$GEN"

echo
env/bin/python -m bench.analysis.anchor_series 2>&1 | grep -viE 'pytorch was not found'
exit "$RC"
