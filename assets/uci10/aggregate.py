#!/usr/bin/env python3
"""Aggregate the 10-dataset UCI semi-synthetic campaign into tables.

Emits assets/uci10/summary.json plus a plain-text report on stdout:
  * per dataset: every method at the MATCHED Gamma (= 4, the DGP's exact Lambda) and at its BEST
    cell over the whole L x c_eps x Gamma grid, against oracle / naive-DR / never- / all-treat
  * the win-loss ledger across the 10 datasets
  * the transport margin (O-W minus O-X under identical Gamma and L) -- the quantity the
    headline claim rests on

Usage: python3 assets/uci10/aggregate.py
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
MATCHED = "4"
OX = ["IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X"]
OW = ["IPW-O-W", "DoublyRobust-O-W"]          # Hajek-O-W dropped from the report
XX = ["IPW-X-X", "DoublyRobust-X-X", "Direct-X-X"]
NAMES = ["adult", "bank_marketing", "credit_default", "online_shoppers", "mushroom",
         "wine_quality", "spambase", "support2", "aids_clinical", "communities_crime"]


def load(name):
    f = RES / (name + ".json")
    return json.loads(f.read_text()) if f.exists() else None


def collect(d):
    """-> per-method dicts keyed by (Gamma, L, c_eps) with the list of per-seed values."""
    cells = d["cells"]; seeds = d["seeds"]
    acc = {}
    refs = {k: [] for k in ("oracle", "never_treat", "all_treat", "naive_dr")}
    xx = {m: [] for m in XX}
    hess = {}; kal = {}
    for s in range(seeds):
        for ce in d["ceps"]:
            c = cells.get("s%d_ce%g" % (s, ce))
            if c is None: continue
            for m, gg in c["grid"].items():
                for g, ll in gg.items():
                    for L, v in ll.items():
                        if v != v: continue
                        acc.setdefault((m, g, L, ce), []).append(v)
            if ce != d["ceps"][0]: continue
            for k in refs: refs[k].append(c["refs"][k])
            for m in XX:
                if m in c.get("xx", {}): xx[m].append(c["xx"][m]["value"])
            for g, o in c.get("hess", {}).items(): hess.setdefault(g, []).append(o["value"])
            for g, o in c.get("kallus", {}).items(): kal.setdefault(g, []).append(o["value"])
    return acc, refs, xx, hess, kal


def summarise(name):
    d = load(name)
    if d is None: return None
    acc, refs, xx, hess, kal = collect(d)
    R = {k: float(np.mean(v)) for k, v in refs.items() if v}
    Rsd = {k: float(np.std(v)) for k, v in refs.items() if v}      # seed spread of the references
    out = {"dataset": name, "meta": d["meta"], "refs": R, "refs_sd": Rsd,
           "n_seeds": d["seeds"], "rows": {}}

    def stat(vals): return {"mean": float(np.mean(vals)), "sd": float(np.std(vals)), "n": len(vals)}

    for m in OX + OW:
        best = None; matched = None
        for (mm, g, L, ce), v in acc.items():
            if mm != m: continue
            s = stat(v); s.update(Gamma=g, L=L, c_eps=ce)
            if best is None or s["mean"] > best["mean"]: best = s
            if g == MATCHED and (matched is None or s["mean"] > matched["mean"]): matched = s
        if best: out["rows"][m] = {"best": best, "matched_best_L": matched}
    for m in XX:
        if xx[m]: out["rows"][m] = {"best": stat(xx[m]), "matched_best_L": stat(xx[m])}
    if hess:
        b = max(hess.items(), key=lambda kv: np.mean(kv[1]))
        out["rows"]["SharpHess"] = {"best": dict(stat(b[1]), Gamma=b[0], L="-", c_eps="-"),
                                    "matched_best_L": dict(stat(hess.get(MATCHED, b[1])), Gamma=MATCHED, L="-", c_eps="-")}
    if kal:
        b = max(kal.items(), key=lambda kv: np.mean(kv[1]))
        out["rows"]["Kallus"] = {"best": dict(stat(b[1]), Gamma=b[0], L="-", c_eps="-"),
                                 "matched_best_L": dict(stat(kal.get(MATCHED, b[1])), Gamma=MATCHED, L="-", c_eps="-")}

    # transport margin: O-W minus its O-X twin at IDENTICAL Gamma and L (c_eps = 1)
    marg = {}
    for a, b in (("IPW-O-W", "IPW-O-X"), ("DoublyRobust-O-W", "DoublyRobust-O-X")):
        dd = []
        for g in [str(x) for x in ("1.5", "2", "4", "8")]:
            for L in ("inf", "3", "1"):
                va, vb = acc.get((a, g, L, 1.0)), acc.get((b, g, L, 1.0))
                if va and vb and len(va) == len(vb):
                    dd.append(float(np.mean(va) - np.mean(vb)))
        if dd: marg[a] = {"mean_over_grid": float(np.mean(dd)), "max": float(np.max(dd)), "min": float(np.min(dd))}
    out["transport_margin"] = marg
    return out


def bar(s): return "-" * s


def main():
    S = [summarise(n) for n in NAMES]
    S = [s for s in S if s]
    ORDER = OX + OW + XX + ["SharpHess", "Kallus"]

    print("\n" + "=" * 108)
    print("UCI 10-DATASET SEMI-SYNTHETIC CAMPAIGN   n_train=300, no capacity constraint, "
          "Gamma* = 4 exact by construction")
    print("=" * 108)

    print("\n### Benchmark construction\n")
    print("%-19s %7s %6s %7s %7s | %7s %7s %7s %7s" %
          ("dataset", "N_pop", "kind", "cor(x,u)", "cor(x,S)", "oracle", "naive", "never", "all"))
    print(bar(96))
    for s in S:
        m = s["meta"]; R = s["refs"]
        print("%-19s %7d %6s %+7.2f %+7.2f | %7.3f %7.3f %7.3f %7.3f" %
              (s["dataset"], m["n_total"], m["outcome_kind"][:4], m["corr_x_u"], m["corr_x_S"],
               R["oracle"], R["naive_dr"], R["never_treat"], R["all_treat"]))

    for tag, key in (("MATCHED Gamma = 4 (best L, c_eps)", "matched_best_L"),
                     ("BEST cell over the whole Gamma x L x c_eps grid", "best")):
        print("\n\n### Average test outcome -- %s\n" % tag)
        hdr = "%-19s" % "dataset" + "".join("%9s" % m.replace("DoublyRobust", "DR").replace("SharpHess", "Hess")[:8] for m in ORDER)
        print(hdr); print(bar(len(hdr)))
        for s in S:
            best_val = max((s["rows"][m][key]["mean"] for m in ORDER if m in s["rows"]), default=None)
            line = "%-19s" % s["dataset"]
            for m in ORDER:
                r = s["rows"].get(m)
                if not r or not r.get(key): line += "%9s" % "-"; continue
                v = r[key]["mean"]
                line += "%9s" % (("*%.3f" % v) if abs(v - best_val) < 1e-9 else "%.3f" % v)
            print(line)
        # ---- cross-dataset summary rows ------------------------------------------------
        # Three aggregates, because no single one is honest on its own:
        #   MEAN        plain average of the 10 outcomes -- dominated by datasets whose outcome
        #               simply lives on a bigger scale (mushroom 0.72 vs wine_quality 0.29).
        #   V/oracle    the ratio asked for. Scale-free but NOT shift-free: wine_quality and
        #               communities_crime have never-treat BELOW zero, so a method can post a
        #               high ratio while barely beating the free constant policy.
        #   normalized  (V - never) / (oracle - never): the fraction of the ACHIEVABLE gain
        #               recovered. Shift- and scale-invariant, 0 = no better than never-treating,
        #               1 = oracle. This is the one to compare methods on.
        for lab, fn in (("MEAN", lambda v, R: v),
                        ("MEAN V/oracle", lambda v, R: v / R["oracle"]),
                        ("MEAN normalized", lambda v, R: (v - R["never_treat"])
                                                         / (R["oracle"] - R["never_treat"]))):
            line = "%-19s" % lab
            for m in ORDER:
                vs = [fn(s2["rows"][m][key]["mean"], s2["refs"]) for s2 in S
                      if m in s2["rows"] and s2["rows"][m].get(key)]
                line += "%9s" % ("%.3f" % np.mean(vs) if vs else "-")
            print(bar(len(hdr)) if lab == "MEAN" else "", end="" if lab != "MEAN" else "\n")
            print(line)
        print("  (* = best method on that dataset; oracle/naive are in the table above)")
        print("  MEAN = plain average over the 10 datasets.  V/oracle = mean ratio to the oracle.")
        print("  normalized = mean (V - never) / (oracle - never), the fraction of achievable gain")
        print("  recovered: 0 = no better than never treating, 1 = oracle. Shift- and scale-free.")

    print("\n\n### Win ledger at matched Gamma = 4  (how often each method is best of all methods)\n")
    wins = {m: 0 for m in ORDER}
    beats_naive = {m: 0 for m in ORDER}; beats_hess = {m: 0 for m in ORDER}
    for s in S:
        vals = {m: s["rows"][m]["matched_best_L"]["mean"] for m in ORDER
                if m in s["rows"] and s["rows"][m].get("matched_best_L")}
        if not vals: continue
        wins[max(vals, key=vals.get)] += 1
        for m, v in vals.items():
            if v > s["refs"]["naive_dr"]: beats_naive[m] += 1
            if "SharpHess" in vals and v > vals["SharpHess"]: beats_hess[m] += 1
    print("%-20s %6s %14s %14s" % ("method", "wins", "beats naive", "beats Hess"))
    print(bar(58))
    for m in ORDER:
        print("%-20s %6d %10d/%d %12d/%d" % (m, wins[m], beats_naive[m], len(S), beats_hess[m], len(S)))

    print("\n\n### Transport margin  (O-W minus its O-X twin, identical Gamma and L, c_eps=1)\n")
    print("%-19s %14s %14s" % ("dataset", "IPW", "DoublyRobust"))
    print(bar(49))
    tot = {a: [] for a in OW}
    for s in S:
        line = "%-19s" % s["dataset"]
        for a in OW:
            mm = s["transport_margin"].get(a)
            if mm: line += "%+14.3f" % mm["mean_over_grid"]; tot[a].append(mm["mean_over_grid"])
            else: line += "%14s" % "-"
        print(line)
    print(bar(49))
    print("%-19s" % "MEAN" + "".join("%+14.3f" % np.mean(tot[a]) if tot[a] else "%14s" % "-" for a in OW))

    json.dump(S, open(HERE / "summary.json", "w"), indent=1)
    print("\nsaved %s\n" % (HERE / "summary.json"))


if __name__ == "__main__":
    main()
