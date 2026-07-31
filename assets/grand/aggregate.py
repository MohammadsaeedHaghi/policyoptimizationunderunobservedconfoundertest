#!/usr/bin/env python3
"""Aggregate the grand campaign's per-seed grid files into per-campaign surfaces.

Each job wrote one seed's full Gamma x L x c_eps x cap grid. This averages them over seeds,
finds each method's best cell WITH its (Gamma, L, c_eps) coordinates, and flags any optimum
sitting on a grid EDGE -- an edge optimum means the sweep is truncated, not converged, which is
exactly the mistake this campaign was launched to fix.

Importable (agg / best_cells) and runnable (prints the tables).
Usage: python3 assets/grand/aggregate.py [gs|km|kz]
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
LABELS = {"gs": "gstar (ours)", "km": "Kallus-Mao-Zhou", "kz": "Kallus-Zhou"}


def load(camp):
    """-> (surface[cell][method][G][L] = mean over seeds, refs[cell], meta, n_seeds)"""
    files = sorted(HERE.glob("%s_s*.json" % camp))
    if not files: return None
    ds = [json.load(open(f)) for f in files]
    d0 = ds[0]
    cells = sorted(set.intersection(*[set(d["cells"]) for d in ds]))
    surf, refs = {}, {}
    for ck in cells:
        surf[ck] = {}
        for m in d0["methods"]:
            surf[ck][m] = {}
            for g in d0["gammas"]:
                surf[ck][m][g] = {}
                for l in d0["Lgrid"]:
                    vals = [d["cells"][ck][m][g][l] for d in ds
                            if d["cells"][ck][m][g][l] == d["cells"][ck][m][g][l]]
                    surf[ck][m][g][l] = float(np.mean(vals)) if vals else float("nan")
        refs[ck] = {k: float(np.mean([d["refs"][ck][k] for d in ds]))
                    for k in ("oracle", "never_treat", "all_treat", "naive_dr")}
    meta = {"gammas": d0["gammas"], "Lgrid": d0["Lgrid"], "methods": d0["methods"],
            "N_train": d0["N_train"], "seeds": [d["seed"] for d in ds], "dgp": d0["dgp"]}
    return surf, refs, meta, len(ds)


def best_cells(surf, meta, regime):
    """Best (value, Gamma, L, c_eps) per method, restricted to cells of one cap regime."""
    keys = [ck for ck in surf if ck.endswith("_" + regime)]
    out = {}
    for m in meta["methods"]:
        cand = [(surf[ck][m][g][l], g, l, ck.split("_")[0][2:])
                for ck in keys for g in meta["gammas"] for l in meta["Lgrid"]
                if surf[ck][m][g][l] == surf[ck][m][g][l]]
        if cand: out[m] = max(cand, key=lambda t: t[0])
    return out


def main():
    camps = sys.argv[1:] or ["gs", "km", "kz"]
    for camp in camps:
        got = load(camp)
        if not got:
            print("== %s: no per-seed files yet" % camp); continue
        surf, refs, meta, ns = got
        regs = sorted({ck.split("_")[1] for ck in surf},
                      key=lambda r: (r != "uncap", r))
        print("\n=== %s  (%d seeds, n=%d, Gamma grid %s..%s, %d L, eps %s)"
              % (LABELS.get(camp, camp), ns, meta["N_train"], meta["gammas"][0],
                 meta["gammas"][-1], len(meta["Lgrid"]),
                 sorted({ck.split("_")[0][2:] for ck in surf})))
        for reg in regs:
            b = best_cells(surf, meta, reg)
            r0 = refs["ce1_" + reg]
            print("  -- %s   oracle %.3f  naive %.3f  never %.3f"
                  % (reg, r0["oracle"], r0["naive_dr"], r0["never_treat"]))
            for m, (v, g, l, ce) in sorted(b.items(), key=lambda kv: -kv[1][0]):
                edge = []
                if g in (meta["gammas"][0], meta["gammas"][-1]): edge.append("G-edge")
                if l in (meta["Lgrid"][0], meta["Lgrid"][-1]): edge.append("L-edge")
                if ce in ("0.5", "2"): edge.append("eps-edge")
                print("     %-18s %8.3f   at Gamma=%-5s L=%-4s c_eps=%-4s %s"
                      % (m, v, g, l, ce, ("[" + ",".join(edge) + "]") if edge else ""))


if __name__ == "__main__":
    main()
