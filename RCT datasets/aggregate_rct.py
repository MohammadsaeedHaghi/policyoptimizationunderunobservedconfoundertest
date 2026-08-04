#!/usr/bin/env python3
"""Aggregate the per-cell RCT experiment results into summary_rct.json."""
import json, glob
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
MATCHED = "4"
OX = ["IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X"]
OW = ["IPW-O-W", "DoublyRobust-O-W"]
XX = ["IPW-X-X", "DoublyRobust-X-X", "Direct-X-X"]
ORDER = OX + OW + XX + ["SharpHess", "Kallus"]
NAMES = ["ihdp", "twins", "ist"]


def summarise(name):
    files = sorted(glob.glob(str(RES / (name + "_s*_ce*.json"))))
    if not files: return None
    acc, refs, xxacc, hess, kal = {}, {}, {}, {}, {}
    meta, gammas, Ls = None, None, None
    seeds = set()
    for f in files:
        d = json.loads(Path(f).read_text())
        meta = meta or d["meta"]; gammas = gammas or ["%g" % g for g in d["gammas"]]
        Ls = Ls or d["lipschitz"]
        for key, c in d["cells"].items():
            sd = int(key.split("_")[0][1:]); ce = float(key.split("ce")[1]); seeds.add(sd)
            for m, gg in c["grid"].items():
                for g, ll in gg.items():
                    for L, v in ll.items():
                        if v == v: acc.setdefault((m, g, L, ce), []).append(v)
            if "refs" in c:
                for k, v in c["refs"].items(): refs.setdefault(k, []).append(v)
            for m, byL in c.get("xx", {}).items():
                for L, o in byL.items(): xxacc.setdefault((m, L), []).append(o["value"])
            for g, o in c.get("hess", {}).items(): hess.setdefault(g, []).append(o["value"])
            for g, o in c.get("kallus", {}).items(): kal.setdefault(g, []).append(o["value"])

    def st(v): return {"mean": float(np.mean(v)), "sd": float(np.std(v)), "n": len(v)}

    out = {"dataset": name, "meta": meta, "gammas": gammas, "Ls": Ls,
           "n_seeds": len(seeds), "n_cells": len(files),
           "refs": {k: float(np.mean(v)) for k, v in refs.items()},
           "refs_sd": {k: float(np.std(v)) for k, v in refs.items()}, "rows": {}}
    for m in OX + OW:
        best = matched = None
        for (mm, g, L, ce), v in acc.items():
            if mm != m: continue
            s = st(v); s.update(Gamma=g, L=L, c_eps=ce)
            if best is None or s["mean"] > best["mean"]: best = s
            if g == MATCHED and (matched is None or s["mean"] > matched["mean"]): matched = s
        if best: out["rows"][m] = {"best": best, "matched_best_L": matched}
    for m in XX:
        cand = [(L, st(v)) for (mm, L), v in xxacc.items() if mm == m]
        if not cand: continue
        b = max(cand, key=lambda t: t[1]["mean"])
        d = dict(b[1]); d.update(Gamma="-", L=b[0], c_eps="-")
        out["rows"][m] = {"best": d, "matched_best_L": d}
    if hess:
        b = max(hess.items(), key=lambda kv: np.mean(kv[1]))
        out["rows"]["SharpHess"] = {"best": dict(st(b[1]), Gamma=b[0], L="-", c_eps="-"),
                                    "matched_best_L": dict(st(hess.get(MATCHED, b[1])),
                                                           Gamma=MATCHED, L="-", c_eps="-")}
    if kal:
        b = max(kal.items(), key=lambda kv: np.mean(kv[1]))
        out["rows"]["Kallus"] = {"best": dict(st(b[1]), Gamma=b[0], L="-", c_eps="-"),
                                 "matched_best_L": dict(st(kal.get(MATCHED, b[1])),
                                                        Gamma=MATCHED, L="-", c_eps="-")}
    # transport margin at identical Gamma and L, c_eps = 1
    marg = {}
    for a, b in (("IPW-O-W", "IPW-O-X"), ("DoublyRobust-O-W", "DoublyRobust-O-X")):
        dd = []
        for g in gammas:
            for L in Ls:
                va, vb = acc.get((a, g, L, 1.0)), acc.get((b, g, L, 1.0))
                if va and vb and len(va) == len(vb): dd.append(float(np.mean(va) - np.mean(vb)))
        if dd: marg[a] = {"mean_over_grid": float(np.mean(dd)),
                          "max": float(np.max(dd)), "min": float(np.min(dd))}
    out["transport_margin"] = marg
    # full surface for plotting: mean over seeds at c_eps = 1
    surf = {}
    for m in OX + OW:
        surf[m] = {g: {L: (float(np.mean(acc[(m, g, L, 1.0)])) if (m, g, L, 1.0) in acc else None)
                       for L in Ls} for g in gammas}
    out["surface"] = surf
    out["xx_by_L"] = {m: {L: float(np.mean(v)) for (mm, L), v in xxacc.items() if mm == m}
                      for m in XX}
    out["hess_by_gamma"] = {g: float(np.mean(v)) for g, v in hess.items()}
    out["kallus_by_gamma"] = {g: float(np.mean(v)) for g, v in kal.items()}
    return out


S = [x for x in (summarise(n) for n in NAMES) if x]
json.dump(S, open(HERE / "summary_rct.json", "w"), indent=1)

print("%-7s %6s %6s | %8s %8s %8s" % ("data", "cells", "seeds", "oracle", "never", "all"))
print("-" * 52)
for s in S:
    R = s["refs"]
    print("%-7s %6d %6d | %8.3f %8.3f %8.3f"
          % (s["dataset"], s["n_cells"], s["n_seeds"], R.get("oracle", float("nan")),
             R.get("never_treat", float("nan")), R.get("all_treat", float("nan"))))
for key, tag in (("matched_best_L", "MATCHED Gamma = 4"), ("best", "BEST cell on the whole grid")):
    print("\n### %s\n" % tag)
    hdr = "%-7s" % "data" + "".join("%10s" % m.replace("DoublyRobust", "DR")[:9] for m in ORDER)
    print(hdr); print("-" * len(hdr))
    for s in S:
        line = "%-7s" % s["dataset"]
        vals = {m: s["rows"][m][key]["mean"] for m in ORDER
                if m in s["rows"] and s["rows"][m].get(key)}
        bv = max(vals.values()) if vals else None
        for m in ORDER:
            line += "%10s" % (("*%.3f" % vals[m]) if m in vals and abs(vals[m] - bv) < 1e-12
                              else ("%.3f" % vals[m] if m in vals else "-"))
        print(line)
    # normalized: fraction of achievable gain recovered
    line = "%-7s" % "NORM"
    for m in ORDER:
        vs = [(s["rows"][m][key]["mean"] - s["refs"]["never_treat"])
              / (s["refs"]["oracle"] - s["refs"]["never_treat"])
              for s in S if m in s["rows"] and s["rows"][m].get(key)]
        line += "%10s" % ("%.3f" % np.mean(vs) if vs else "-")
    print(line)
print("\n### transport margin (O-W minus its O-X twin, same Gamma and L)\n")
for s in S:
    print("  %-7s %s" % (s["dataset"], {k.split("-")[0]: round(v["mean_over_grid"], 4)
                                        for k, v in s["transport_margin"].items()}))
print("\nwrote summary_rct.json")
