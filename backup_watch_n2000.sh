#!/usr/bin/env bash
# Watch for N=2000 result files; back each up AND copy into the selected-experiment folders
# the moment it appears, until the run completes.
cd "/home1/haghim/code 1.1"
BK="results_n2000_backup"; mkdir -p "$BK"
SOWG="selected experiment/exp_owgap"; SNM="selected experiment/exp_nmcap"
mkdir -p "$SOWG" "$SNM"
copy_new(){
  for f in assets/exp_owgap/owgap_results_n2000_ce*.json; do
    [ -f "$f" ] || continue
    b="$BK/$(basename "$f")"
    if [ ! -f "$b" ] || [ "$f" -nt "$b" ]; then
      cp -f "$f" "$b"; cp -f "$f" "$SOWG/"
      echo "$(date +%H:%M:%S) saved $(basename "$f") -> backup + $SOWG ($(wc -c <"$f") bytes)"
    fi
  done
  for f in assets/exp_nmcap/nmcap_results_n2000_ce*.json; do
    [ -f "$f" ] || continue
    b="$BK/$(basename "$f")"
    if [ ! -f "$b" ] || [ "$f" -nt "$b" ]; then
      cp -f "$f" "$b"; cp -f "$f" "$SNM/"
      echo "$(date +%H:%M:%S) saved $(basename "$f") -> backup + $SNM ($(wc -c <"$f") bytes)"
    fi
  done
}
while ! grep -q "ALL_N2000_DONE" run_n2000_5seed.log 2>/dev/null; do copy_new; sleep 30; done
copy_new
echo "BACKUP_COMPLETE: $(ls -1 $BK | wc -l) files in $BK; copies in '$SOWG' and '$SNM'"
ls -la "$BK"
