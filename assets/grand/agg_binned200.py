#!/usr/bin/env python3
"""Aggregate the binned-policy-class test: does sharp's 15-bin class close O-W's deficit?

Per campaign x regime, reports each arm at the matched Gamma and at its best cell, alongside
Sharp-O-X and both naives, plus the PAIRED per-seed difference (binned - free-pi) so the effect
is not swamped by seed noise.

Usage: python3 assets/grand/agg_binned200.py
"""
import json, glob, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
MATCH = {"gs": "5", "km": "4.4817", "kz": "4.4817"}
SHARP_ARM = {"gs": "L=5 (main)", "km": "e^1.5 (main)", "kz": "e^1.5 (main)"}
NAME = {"gs": "gstar (ours)", "km": "Kallus-Mao-Zhou", "kz": "Kallus-Zhou"}
OW = ["IPW-O-W", "DoublyRobust-O-W"]


def main():
    SH = json.load(open(HERE / "sharp_all.json"))
    RF = json.load(open(HERE / "capped_refs.json"))["refs"]
    for camp in ("gs", "km", "kz"):
        fs = sorted(glob.glob(str(HERE.parent / "binned200" / ("%s_s*.json" % camp))))
        if not fs:
            print("== %s: nothing landed yet" % camp); continue
        ds = [json.load(open(f)) for f in fs]
        G0 = MATCH[camp]; arms = ds[0]["arms"]; gam = ds[0]["gammas"]; meths = ds[0]["methods"]
        print("\n" + "=" * 92)
        print("%s -- %d seeds, n=200, BINS=15, matched Gamma* = %s" % (NAME[camp], len(ds), G0))
        print("=" * 92)
        for reg, rk in (("uncap", "uncap"), ("cap30", "cap30")):
            keys = [k for k in ds[0]["cells"] if k.endswith("_" + reg)]
            if not keys: continue
            sh = SH["campaigns"][camp]["arms"][SHARP_ARM[camp]]["regimes"][reg]["mean"]
            gg = SH["campaigns"][camp]["gammas"]
            mi = min(range(len(gg)), key=lambda i: abs(float(gg[i]) - float(G0)))
            R = RF[camp][rk]
            print("\n  [%s]  Sharp-O-X %.3f (matched) / %.3f (best) | naive-lin %.3f | naive-np %.3f | oracle %.3f"
                  % (reg, sh[mi], max(sh), R["naive"][0], R["naive_np"][0],
                     np.mean([d["refs"][keys[0]]["oracle"] for d in ds])))

            def vals(arm, m, g, k):
                out = []
                for d in ds:
                    try:
                        v = d["cells"][k][arm][m][g]
                    except KeyError:
                        continue
                    if v == v: out.append(v)
                return out

            print("    %-16s %-18s %9s %9s   %s" % ("arm", "method", "matched", "best", "best at G"))
            store = {}
            for arm in arms:
                for m in OW + ["IPW-O-X"]:
                    mv = max((np.mean(vals(arm, m, G0, k)) for k in keys
                              if vals(arm, m, G0, k)), default=float("nan"))
                    cand = [(np.mean(vals(arm, m, g, k)), g) for g in gam for k in keys
                            if vals(arm, m, g, k)]
                    bv, bg = max(cand) if cand else (float("nan"), "-")
                    store[(arm, m)] = mv
                    print("    %-16s %-18s %9.3f %9.3f   G=%s" % (arm, m, mv, bv, bg))
            # paired difference, per seed, at matched Gamma, best over (c_eps, cap-regime keys)
            print("    paired (binned15_Lnone - free-pi_L3) at matched Gamma, per seed:")
            for m in OW:
                diffs = []
                for d in ds:
                    a = [d["cells"][k]["binned15_Lnone"][m][G0] for k in keys
                         if "binned15_Lnone" in d["cells"][k]]
                    b = [d["cells"][k]["free-pi_L3"][m][G0] for k in keys
                         if "free-pi_L3" in d["cells"][k]]
                    a = [v for v in a if v == v]; b = [v for v in b if v == v]
                    if a and b: diffs.append(max(a) - max(b))
                if diffs:
                    print("      %-18s %+.3f +- %.3f  (%d seeds, %d positive)"
                          % (m, np.mean(diffs), np.std(diffs), len(diffs), sum(1 for v in diffs if v > 0)))


if __name__ == "__main__":
    main()
