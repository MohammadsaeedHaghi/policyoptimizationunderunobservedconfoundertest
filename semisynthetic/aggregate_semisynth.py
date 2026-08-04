#!/usr/bin/env python3
"""Aggregate the semi-synthetic risk-score-paper campaign into summary_semisynth.json."""
import json, glob
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OX = ["IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X"]
OW = ["IPW-O-W", "DoublyRobust-O-W", "Hajek-O-W"]
XX = ["IPW-X-X", "DoublyRobust-X-X", "Direct-X-X"]
RAT_OK = ["IPW-O-W", "DoublyRobust-O-W"]
ORDER = OX + OW + XX + ["SharpHess", "Kallus"]


def st(v):
    return {"mean": float(np.mean(v)), "sd": float(np.std(v)), "n": len(v)}


DS = "bank_marketing"


def summarise(gamma):
    # the SLURM array names files from a shell string ("0.0"), while "%g" yields "0" --
    # accept both spellings rather than depending on one formatter.
    files = sorted(set(glob.glob(str(RES / ("%s_g%%g_d*_s*.json" % DS % gamma))))
                   | set(glob.glob(str(RES / ("%s_g%%s_d*_s*.json" % DS % gamma)))))
    if not files:
        return None
    acc, objacc, xxacc, hess, kal = {}, {}, {}, [], []
    hessk = []          # the k-NN-nuisance arm, kept alongside the paper-aligned neural one
    refs, meta, G, Ls, ceps = {}, None, None, None, None
    cellacc = {}          # (dgp_seed, method, c_eps, L) -> per-split values
    dgp_bc, dgp_or = {}, {}   # dgp_seed -> per-split best-constant / oracle references
    dgp_refs = {}         # dgp_seed -> list of headrooms
    for f in files:
        d = json.loads(Path(f).read_text())
        meta = meta or d["meta"]; G = G or d["Gamma"]
        Ls = Ls or d["lipschitz"]; ceps = ceps or [("%g" % c) for c in d["ceps"]]
        for k, v in d["refs"].items():
            refs.setdefault(k, []).append(v)
        _ds = d.get("dgp_seed", 0)
        _R = d["refs"]
        dgp_refs.setdefault(_ds, []).append(
            _R["oracle"] - max(_R["never_treat"], _R["all_treat"]))
        # accumulate per (dgp_seed, method, cell) so a draw's BEST CELL can be chosen the same
        # way the pooled table chooses it -- max over cells of the mean, not mean of per-split
        # maxima. Those differ by Jensen and would make the charts disagree with the tables.
        for m, byC in d["grid"].get("rat0", {}).items():
            for c, byL in byC.items():
                for L, v in byL.items():
                    if v == v: cellacc.setdefault((_ds, m, c, L), []).append(v)
        for m, byL in d.get("xx", {}).items():
            for L, o in byL.items():
                if o["value"] == o["value"]: cellacc.setdefault((_ds, m, "-", L), []).append(o["value"])
        for _g, o in d.get("hess", {}).items(): cellacc.setdefault((_ds, "SharpHess", "-", "-"), []).append(o["value"])
        for _g, o in d.get("hess_knn", {}).items(): cellacc.setdefault((_ds, "SharpHess-kNN", "-", "-"), []).append(o["value"])
        for _g, o in d.get("kallus", {}).items(): cellacc.setdefault((_ds, "Kallus", "-", "-"), []).append(o["value"])
        dgp_bc.setdefault(_ds, []).append(max(_R["never_treat"], _R["all_treat"]))
        dgp_or.setdefault(_ds, []).append(_R["oracle"])
        for rk, byM in d["grid"].items():
            for m, byC in byM.items():
                for c, byL in byC.items():
                    for L, v in byL.items():
                        if v == v:
                            acc.setdefault((rk, m, c, L), []).append(v)
                            o = d["objective"][rk][m][c].get(L)
                            if o is not None:
                                objacc.setdefault((rk, m, c, L), []).append(o)
        for m, byL in d.get("xx", {}).items():
            for L, o in byL.items():
                xxacc.setdefault((m, L), []).append(o["value"])
        for _g, o in d.get("hess", {}).items(): hess.append(o["value"])
        for _g, o in d.get("hess_knn", {}).items(): hessk.append(o["value"])
        for _g, o in d.get("kallus", {}).items(): kal.append(o["value"])

    R = {k: float(np.mean(v)) for k, v in refs.items()}
    Rsd = {k: float(np.std(v)) for k, v in refs.items()}
    out = {"gamma": gamma, "Gamma": G, "n_seeds": len(files), "meta": meta,
           "refs": R, "refs_sd": Rsd, "ceps": ceps, "Ls": Ls, "rows": {}, "rows_rat": {}}

    def best_over(rk, m):
        cand = [(c, L, st(v)) for (r_, m_, c, L), v in acc.items() if r_ == rk and m_ == m]
        if not cand:
            return None
        c, L, s = max(cand, key=lambda t: t[2]["mean"])
        s = dict(s); s.update(c_eps=c, L=L)
        return s

    for m in OX + OW:
        b = best_over("rat0", m)
        if b: out["rows"][m] = b
    for m in RAT_OK:
        b = best_over("rat1", m)
        if b: out["rows_rat"][m] = b
    for m in XX:
        cand = [(L, st(v)) for (m_, L), v in xxacc.items() if m_ == m]
        if cand:
            L, s = max(cand, key=lambda t: t[1]["mean"]); s = dict(s); s.update(c_eps="-", L=L)
            out["rows"][m] = s
    if hess: out["rows"]["SharpHess"] = dict(st(hess), c_eps="-", L="-")
    if hessk: out["rows"]["SharpHess-kNN"] = dict(st(hessk), c_eps="-", L="-")
    if kal: out["rows"]["Kallus"] = dict(st(kal), c_eps="-", L="-")

    # transport margin: O-W minus its O-X twin at IDENTICAL L (O-X is c_eps-free -> use its cell)
    marg = {}
    for a, b in (("IPW-O-W", "IPW-O-X"), ("DoublyRobust-O-W", "DoublyRobust-O-X")):
        for rk in ("rat0", "rat1"):
            dd = []
            for L in Ls:
                for c in ceps:
                    va = acc.get((rk, a, c, L)); vb = acc.get((rk, b, ceps[0], L))
                    if va and vb and len(va) == len(vb):
                        dd.append(float(np.mean(va) - np.mean(vb)))
            if dd: marg["%s|%s" % (a, rk)] = float(np.mean(dd))
    out["transport_margin"] = marg

    # C4 effect: same method, same cell, rationality on minus off
    c4 = {}
    for m in RAT_OK:
        dd, do = [], []
        for c in ceps:
            for L in Ls:
                v1, v0 = acc.get(("rat1", m, c, L)), acc.get(("rat0", m, c, L))
                if v1 and v0 and len(v1) == len(v0):
                    dd.append(float(np.mean(v1) - np.mean(v0)))
                o1, o0 = objacc.get(("rat1", m, c, L)), objacc.get(("rat0", m, c, L))
                if o1 and o0 and len(o1) == len(o0):
                    do.append(float(np.mean(o1) - np.mean(o0)))
        if dd: c4[m] = {"value_delta": float(np.mean(dd)), "value_delta_max": float(np.max(dd)),
                        "objective_delta": float(np.mean(do)) if do else None,
                        "objective_delta_min": float(np.min(do)) if do else None}
    out["c4_effect"] = c4
    # normalised value per DGP draw -> lets us report DGP-draw spread separately from split noise
    # per draw: best cell by that draw's own mean, normalised by that draw's own references
    per_dgp = {}
    for (ds_, m, c, L), v in cellacc.items():
        per_dgp.setdefault(ds_, {}).setdefault(m, []).append((float(np.mean(v)), len(v)))
    out["per_dgp"] = {}
    for ds_, byM in per_dgp.items():
        bc = float(np.mean(dgp_bc[ds_])); sc = float(np.mean(dgp_or[ds_])) - bc
        out["per_dgp"][str(ds_)] = {m: {"mean": (max(x[0] for x in lst) - bc) / sc,
                                        "raw": max(x[0] for x in lst),
                                        "n": max(x[1] for x in lst)}
                                    for m, lst in byM.items()}
    out["per_dgp_headroom"] = {str(k): float(np.mean(v)) for k, v in dgp_refs.items()}
    out["n_dgp_draws"] = len(per_dgp)
    out["surface"] = {rk: {m: {c: {L: (float(np.mean(acc[(rk, m, c, L)]))
                                      if (rk, m, c, L) in acc else None) for L in Ls}
                               for c in ceps} for m in (OX + OW if rk == "rat0" else RAT_OK)}
                      for rk in ("rat0", "rat1")}
    return out


import argparse
_ap = argparse.ArgumentParser(); _ap.add_argument("--dataset", default="bank_marketing")
DS = _ap.parse_args().dataset
S = [x for x in (summarise(g) for g in (0.0, 1.0, 1.5, 2.0, 3.0, 4.0)) if x]
for _s in S: _s["dataset"] = DS
json.dump(S, open(HERE / ("summary_%s.json" % DS), "w"), indent=1)
print("\n=== %s ===" % DS)

print("%-6s %9s %6s | %8s %8s %8s %9s" % ("gamma", "Gamma", "seeds", "oracle", "never", "all", "headroom"))
print("-" * 62)
for s in S:
    R = s["refs"]; bc = max(R["never_treat"], R["all_treat"])
    print("%-6.1f %9.3f %6d | %8.4f %8.4f %8.4f %9.4f"
          % (s["gamma"], s["Gamma"], s["n_seeds"], R["oracle"], R["never_treat"],
             R["all_treat"], R["oracle"] - bc))

print("\n### best cell per method (matched Gamma = e^{2g}), rationality OFF\n")
hdr = "%-6s" % "gamma" + "".join("%11s" % m.replace("DoublyRobust", "DR")[:10] for m in ORDER)
print(hdr); print("-" * len(hdr))
for s in S:
    line = "%-6.1f" % s["gamma"]
    vals = {m: s["rows"][m]["mean"] for m in ORDER if m in s["rows"]}
    bv = max(vals.values()) if vals else None
    for m in ORDER:
        line += "%11s" % (("*%.4f" % vals[m]) if m in vals and abs(vals[m] - bv) < 1e-12
                          else ("%.4f" % vals[m] if m in vals else "-"))
    print(line)

print("\n### C4 (historical rationality) ON minus OFF, same cell\n")
print("%-6s %14s %14s %14s %14s" % ("gamma", "IPW-O-X", "DR-O-X", "IPW-O-W", "DR-O-W"))
print("-" * 66)
for s in S:
    line = "%-6.1f" % s["gamma"]
    for m in RAT_OK:
        c = s["c4_effect"].get(m)
        line += "%14s" % ("%+.4f" % c["value_delta"] if c else "-")
    print(line)

print("\n### transport margin (O-W minus its O-X twin, same L)\n")
print("%-6s %14s %14s %14s %14s" % ("gamma", "IPW rat0", "IPW rat1", "DR rat0", "DR rat1"))
print("-" * 66)
for s in S:
    m_ = s["transport_margin"]
    print("%-6.1f %14s %14s %14s %14s" % (s["gamma"],
          "%+.4f" % m_["IPW-O-W|rat0"] if "IPW-O-W|rat0" in m_ else "-",
          "%+.4f" % m_["IPW-O-W|rat1"] if "IPW-O-W|rat1" in m_ else "-",
          "%+.4f" % m_["DoublyRobust-O-W|rat0"] if "DoublyRobust-O-W|rat0" in m_ else "-",
          "%+.4f" % m_["DoublyRobust-O-W|rat1"] if "DoublyRobust-O-W|rat1" in m_ else "-"))

print("\nwrote summary_%s.json" % DS)
