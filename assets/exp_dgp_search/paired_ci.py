"""Paired confidence intervals from a saved run (run_experiment_parallel output).

The runner saves per-seed POLICIES (policy_by_seed); since the realised value V(pi)=exact_value(pi) is a
deterministic function of the policy, we recompute the per-seed value for every method, then form PAIRED
per-seed differences (method - baseline) and bootstrap a 95% CI on the mean difference. Paired = same seed =
same data draw, so it removes the across-seed variance and is the right test for "does O-W beat naive."

Usage:
  python3 assets/exp_dgp_search/paired_ci.py --results <run.json> --dgp <dgp.py> --regime uncap \
      --baseline DoublyRobust-X-X --methods IPW-O-W,DoublyRobust-O-W --gammas 2,3 [--out paired.json]
"""
import sys, json, argparse, importlib.util
from pathlib import Path
import numpy as np

def load_dgp(path):
    sp = importlib.util.spec_from_file_location("d", path); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
def gk(g): g = float(g); return str(int(g)) if g == int(g) else str(g)

def boot_ci(diffs, B=10000, seed=0):
    diffs = np.asarray(diffs, float); n = len(diffs)
    rng = np.random.default_rng(seed)
    means = diffs[rng.integers(0, n, size=(B, n))].mean(axis=1)
    return float(np.mean(diffs)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True); ap.add_argument("--dgp", required=True)
    ap.add_argument("--regime", default="uncap"); ap.add_argument("--baseline", default="DoublyRobust-X-X")
    ap.add_argument("--methods", default="IPW-O-W,DoublyRobust-O-W")
    ap.add_argument("--gammas", default=""); ap.add_argument("--out", default="")
    a = ap.parse_args()
    d = load_dgp(a.dgp); R = json.loads(Path(a.results).read_text())
    G = R["gammas"]; gammas = [float(x) for x in a.gammas.split(",")] if a.gammas else G
    seeds = [str(s) for s in R["seeds"]]; PB = R["regimes"][a.regime]["policy_by_seed"]
    methods = a.methods.split(","); base = a.baseline
    # per-seed value of a method at a gamma
    def val_seed(m, s, g):
        pol = PB[m][s][gk(g)]; return d.exact_value(pol)
    out = {"results": a.results, "regime": a.regime, "baseline": base, "n_seeds": len(seeds), "oracle": R["oracle"], "by_gamma": {}}
    print("PAIRED CIs (95%% bootstrap), regime=%s, baseline=%s, %d seeds\n" % (a.regime, base, len(seeds)))
    for g in gammas:
        row = {}
        bvals = np.array([val_seed(base, s, g) for s in seeds])
        print("Γ=%-4s  %s = %.3f ± %.3f" % (gk(g), base, bvals.mean(), bvals.std()))
        for m in methods:
            mvals = np.array([val_seed(m, s, g) for s in seeds])
            diffs = mvals - bvals
            md, lo, hi = boot_ci(diffs)
            wins = int(np.sum(diffs > 0)); sig = "SIG" if lo > 0 else ("---" if hi < 0 else "ns")
            row[m] = {"mean": round(float(mvals.mean()), 4), "sd": round(float(mvals.std()), 4),
                      "paired_diff": round(md, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                      "win_seeds": wins, "n": len(seeds), "significant": bool(lo > 0)}
            print("        %-18s %.3f ± %.3f | Δ=%+.3f [%.3f, %.3f] %s  (%d/%d seeds win)"
                  % (m, mvals.mean(), mvals.std(), md, lo, hi, sig, wins, len(seeds)))
        out["by_gamma"][gk(g)] = row
        print()
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=2)); print("saved", a.out)

if __name__ == "__main__":
    main()
