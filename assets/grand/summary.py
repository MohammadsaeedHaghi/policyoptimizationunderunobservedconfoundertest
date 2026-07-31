#!/usr/bin/env python3
"""Final summary tables for the grand campaign: matched-Gamma AND best-cell, against BOTH naives.

Two tables per campaign per cap regime:
  * MATCHED Gamma -- the principled protocol. Gamma is fixed at the DGP's Gamma*, which is known
    by construction; only L and c_eps are chosen. This is the headline number.
  * BEST CELL -- Gamma, L and c_eps all free. This is an ORACLE-TUNED upper bound, not something
    a practitioner could achieve, and it is labelled as such. Reported because the earlier grids
    were truncated at Gamma = 8 and we need to see where the optimum actually lies; cells sitting
    on a grid edge are flagged, because an edge optimum means the sweep has not converged.

Baselines are the cap-respecting ones from capped_refs.py, BOTH model classes:
  naive-linear   (common.outcome_means -> LinearRegression, what the runners use)
  naive-nonpar   (15-bin per-arm means; the Gamma = 1 limit of the sharp baseline)
plus Sharp-O-X from sharp_all.json. The stronger naive is the honest comparator, so it is
starred -- a referee will always pick whichever one beats us.

Usage: python3 assets/grand/summary.py [gs|km|kz ...]
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from aggregate import load, best_cells, LABELS

MATCHED = {"gs": "5", "km": "4.4817", "kz": "4.4817"}
SHARP_ARM = {"gs": "L=5 (main)", "km": "e^1.5 (main)", "kz": "e^1.5 (main)"}
REG_ORDER = ["uncap", "cap30", "cap40", "cap50"]


def sharp_for(camp, regime):
    """(best value over the sharp Gamma grid, its Gamma, value at the matched Gamma)."""
    p = HERE / "sharp_all.json"
    if not p.exists(): return None
    S = json.load(open(p))
    c = S["campaigns"].get(camp)
    if not c: return None
    arm = c["arms"].get(SHARP_ARM[camp])
    if not arm or regime not in arm["regimes"]: return None
    vals = arm["regimes"][regime]["mean"]; gg = c["gammas"]
    bi = int(np.argmax(vals))
    mi = min(range(len(gg)), key=lambda i: abs(float(gg[i]) - float(MATCHED[camp])))
    return vals[bi], gg[bi], vals[mi]


def refs_for(camp, regime):
    p = HERE / "capped_refs.json"
    if not p.exists(): return None
    R = json.load(open(p))["refs"].get(camp, {}).get(regime)
    if not R: return None
    return {k: R[k][0] for k in R}


def main():
    camps = sys.argv[1:] or ["gs", "km", "kz"]
    for camp in camps:
        got = load(camp)
        if not got:
            print("== %s: nothing landed" % camp); continue
        surf, _rf, meta, ns = got
        G0 = MATCHED[camp]
        if G0 not in meta["gammas"]:
            G0 = min(meta["gammas"], key=lambda g: abs(float(g) - float(MATCHED[camp])))
        print("\n" + "=" * 96)
        print("%s  --  %d seeds, n=%d, matched Gamma* = %s" % (LABELS[camp], ns, meta["N_train"], G0))
        print("=" * 96)
        for reg in REG_ORDER:
            keys = [ck for ck in surf if ck.endswith("_" + reg)]
            if not keys: continue
            R = refs_for(camp, reg) or {}
            sh = sharp_for(camp, reg)
            print("\n  [%s]  oracle %s | naive-linear %s | naive-nonpar %s | never %s"
                  % (reg,
                     ("%.3f" % R["oracle"]) if R else "n/a",
                     ("%.3f" % R["naive"]) if R else "n/a",
                     ("%.3f" % R["naive_np"]) if R else "n/a",
                     ("%.3f" % R["never"]) if R else "n/a"))
            best_naive = max(R.get("naive", -9e9), R.get("naive_np", -9e9)) if R else None
            rows = []
            for m in meta["methods"]:
                cand = [(surf[ck][m][G0][l], l, ck.split("_")[0][2:]) for ck in keys
                        for l in meta["Lgrid"] if surf[ck][m][G0][l] == surf[ck][m][G0][l]]
                if not cand: continue
                v, l, ce = max(cand, key=lambda t: t[0])
                rows.append((v, m, l, ce))
            bc = best_cells(surf, meta, reg)
            print("    %-18s %9s %-14s | %9s %-22s" %
                  ("method", "MATCHED", "(L, c_eps)", "BEST", "(Gamma, L, c_eps)"))
            for v, m, l, ce in sorted(rows, reverse=True):
                bv, bg, bl, bce = bc[m]
                edge = []
                if bg in (meta["gammas"][0], meta["gammas"][-1]): edge.append("G")
                if bl in (meta["Lgrid"][0], meta["Lgrid"][-1]): edge.append("L")
                if bce in ("0.5", "2"): edge.append("e")
                star = " *" if (best_naive is not None and v > best_naive) else "  "
                print("    %-18s %9.3f %-14s |%10.3f %-22s %s%s"
                      % (m, v, "(%s, %s)" % (l, ce), bv, "(%s, %s, %s)" % (bg, bl, bce),
                         ("edge:" + "".join(edge)) if edge else "", star))
            if sh:
                print("    %-18s %9.3f %-14s |%10.3f %-22s" %
                      ("Sharp-O-X", sh[2], "(bins=15)", sh[0], "(%s, bins=15)" % sh[1]))
            if best_naive is not None:
                print("    (* = beats the stronger naive, %.3f)" % best_naive)


if __name__ == "__main__":
    main()
