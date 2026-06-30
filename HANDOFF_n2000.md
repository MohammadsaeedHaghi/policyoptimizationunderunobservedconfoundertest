# Handoff — N=2000, 5-seed reruns of the selected experiments

**Why this file exists:** the work was paused to restart inside a larger SLURM allocation
(the previous 32 GB allocation OOM-killed the N=2000 run). A new SLURM session is a fresh
Claude with no memory of the prior chat — this file is the pickup point.

## Context
- Project: Confounding-Robust Policy Optimization (`/home1/haghim/code 1.1`).
- Two "selected" experiments live in `selected experiment/` (folder) and `assets/exp_owgap`, `assets/exp_nmcap`.
- Goal: rerun the value sweeps for BOTH experiments at **N_train=2000, 5 seeds** (was owgap N=600/5-seed, nmcap N=500/15-seed).
- The interactive write-up is `index.html` ("Ninth experiment" = exp_owgap); the standalone
  bundle is `selected experiment/report.html` (self-contained, base64 plots, 4 tabs, Uncapped/Capped sub-tabs).

## Resource issue (root cause)
- Prior allocation: `SLURM_MEM_PER_NODE=32768` (32 GB), 16 CPUs. N=2000 with 8 Gurobi workers
  exceeded 32 GB → cgroup OOM-killed (empty log, vanished procs).
- Fix chosen by user: restart with more RAM+CPU, e.g.:
  `salloc --mem=128G --cpus-per-task=32 --time=12:00:00` then launch claude.
- Gurobi WLS license caps concurrent sessions at **8** → keep `WORKERS=8`; use extra cores via
  `THREADS` (8×4 = 32 cores).

## Step 1 — run it  (MUST be daemonized — a plain bash/tracked-task run DIES on Claude session restart)
```bash
cd "/home1/haghim/code 1.1"
setsid nohup env THREADS=3 bash run_n2000_5seed.sh < /dev/null > run_n2000_launch.out 2>&1 &
# verify detached: `ps -o ppid= -p $(pgrep -f run_n2000_5seed.sh)` should be 1 (reparented to init)
# config: WORKERS=10 x THREADS=3 (32-core/64GB SLURM box). ~2+ hrs PER epsilon at N=2000 -> ~6-8h total.
# marker "ALL_N2000_DONE" in run_n2000_5seed.log when finished. Results write per-epsilon (incremental).
```
LESSON LEARNED: the first attempts ran as Claude-session-tied background tasks and were KILLED on
session teardown (lost ~2h, no results — each epsilon only writes when all 10 of its jobs finish).
Always launch with `setsid` so the run lives in its own session (PPID=1) and survives restarts.
Note: still dies if the SLURM allocation itself ends (job id changes).
Outputs:
- `assets/exp_owgap/owgap_results_n2000_ce{1.0,1.5,2.0}.json`
- `assets/exp_nmcap/nmcap_results_n2000_ce1.0.json`
(These new files do NOT overwrite the existing 5-seed/15-seed results.)

## Step 2 — regenerate plots WITH SD bands (the user specifically wants the SD shading visible)
The value PNGs must show mean ± SD over the 5 seeds (the results JSON has `regimes[reg]['mean']`
and `['sd']`). A working band-plot generator was used before — replicate it:
- For each owgap result file → `val_{uncap,cap}_ce{1,1.5,2}.png` in `assets/exp_owgap/` AND
  `selected experiment/exp_owgap/`.
- For nmcap → `nm_value_{uncap,cap}.png` in `assets/exp_nmcap/` AND `selected experiment/exp_nmcap/`.
- Style convention: colour = estimator family
  (IPW #1f77b4, DoublyRobust #2ca02c, Hajek #9467bd, Direct #ff7f0e, Oracle #444),
  shape = uncertainty set (O-W circle, O-X square, X-X dashed, Oracle dotted).
  Method order: IPW-X-X, DoublyRobust-X-X, Direct-X-X, IPW-O-X, DoublyRobust-O-X, Hajek-O-X,
  IPW-O-W, DoublyRobust-O-W, Oracle. Use `fill_between(mean-sd, mean+sd, alpha=0.13)`.

## Step 3 — rebuild the standalone report
```bash
cd "/home1/haghim/code 1.1/selected experiment"
python3 build_report.py        # embeds plots as base64; 4 tabs, Uncapped/Capped sub-sub-tabs
```
Update captions/lead in `build_report.py` from "5-seed / 15-seed" + "N=600 / N=500" to
"5-seed, **N=2000**" for both experiments. Also refresh the restab tables (they read the chart JSON;
point them at the new n2000 means, or regenerate owgap_charts.json / combined_charts.json).

## Step 4 — (optional) update index.html "Ninth experiment"
The interactive owgap chart there uses `OWGAP_CHARTS` (embedded `assets/exp_owgap/owgap_charts.json`).
Rebuild that JSON from the n2000 results if you want the interactive panel to match.

## Serve to view (cluster has no display; use the tunnel)
```bash
cd "/home1/haghim/code 1.1"; python3 -m http.server 8731 --bind 127.0.0.1 &
# open http://localhost:8731/index.html  and
#      http://localhost:8731/selected%20experiment/report.html
```

## Open follow-ups the user may still want
- Push SD bands into the index.html interactive owgap chart (currently band-less).
- Add ±SD into the result-TABLE cells (currently mean-only).
