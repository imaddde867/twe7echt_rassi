#!/usr/bin/env bash
# Check that runs with reproduction OFF are byte-identical to a reference git ref (default: origin/main).
#
#   scripts/check_off_mode.sh [REF]
#
# Features such as births are meant to be invisible when switched off: outcomes.csv, events.csv, deaths.csv and
# snapshots.csv must match the reference exactly for the same seed. The third run is 50 years long on purpose,
# because founders are at most 40 and only a long run reaches retirement (pensions). Exit status 0 = identical.
set -uo pipefail

REF="${1:-origin/main}"
ROOT="$(git rev-parse --show-toplevel)"
TMP="$(mktemp -d)"
cleanup() { git -C "$ROOT" worktree remove --force "$TMP/ref" >/dev/null 2>&1; rm -rf "$TMP"; }
trap cleanup EXIT

git -C "$ROOT" worktree add -q "$TMP/ref" "$REF" || { echo "cannot check out $REF"; exit 2; }

status=0
for spec in "rule:25:12:4" "utility:25:12:4" "rule:20:50:2"; do
  IFS=: read -r pol agents years seed <<<"$spec"
  tag="${pol}_${years}y"
  args=(--policy "$pol" --agents "$agents" --years "$years" --seed "$seed")
  (cd "$TMP/ref" && PYTHONPATH="$TMP/ref" python -m lifesim.run "${args[@]}" --out "$TMP/ref_out/$tag" >/dev/null 2>&1) \
    || { echo "reference run failed: $tag"; status=2; continue; }
  (cd "$ROOT" && PYTHONPATH="$ROOT" python -m lifesim.run "${args[@]}" --out "$TMP/new_out/$tag" >/dev/null 2>&1) \
    || { echo "run failed: $tag"; status=2; continue; }
  for f in outcomes events deaths snapshots; do
    if cmp -s "$TMP/ref_out/$tag/$f.csv" "$TMP/new_out/$tag/$f.csv"; then
      echo "identical  $tag/$f.csv ($(wc -c <"$TMP/new_out/$tag/$f.csv") bytes)"
    else
      echo "DIFFERENT  $tag/$f.csv"; status=1
    fi
  done
done

[ "$status" -eq 0 ] && echo "OK: reproduction-off outputs are byte-identical to $REF" || echo "FAILED (status $status)"
exit "$status"
