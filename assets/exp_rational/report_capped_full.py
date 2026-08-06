#!/usr/bin/env python3
"""Complete capped table: every method against every (config, cap), with ranks."""
import json, glob, collections
import numpy as np

OW = ["IPW-O-W", "DoublyRobust-O-W", "Hajek-O-W"]
OX = ["IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X"]
XX = ["IPW-X-X", "DoublyRobust-X-X", "Direct-X-X"]
BL = ["Kallus", "SharpHess-kNN", "SharpHess"]
ORDER = OW + OX + XX + ["naive"] + BL
LBL = {"DoublyRobust-O-W": "DR-O-W", "DoublyRobust-O-X": "DR-O-X", "DoublyRobust-X-X": "DR-X-X",
       "SharpHess": "Hess(neural)", "SharpHess-kNN": "Hess(k-NN)", "naive": "naive plug-in"}
FAM = {**{m: "O-W" for m in OW}, **{m: "O-X" for m in OX}, **{m: "X-X" for m in XX},
       "naive": "-", "Kallus": "pub", "SharpHess": "pub", "SharpHess-kNN": "pub"}

byk = collections.defaultdict(list)
for f in glob.glob("capped_v2/*_s*.json"):
    d = json.load(open(f)); byk[(d["tag"], d["cap"])].append(d["cell"])
COLS = sorted(byk, key=lambda t: (t[0], -t[1]))


def nz(v, c): return (v - c["bc"]) / c["sc"]


def cell_stats(cs, m):
    """(best mean, its sd, best (c_eps,L)) over the grid, plus the per-L means at c_eps=1."""
    cand, byL = [], {}
    if m in cs[0].get("grid", {}):
        for ce in cs[0]["grid"][m]:
            for lk in cs[0]["grid"][m][ce]:
                vs = [nz(c["grid"][m][ce][lk], c) for c in cs if lk in c["grid"][m].get(ce, {})]
                if len(vs) == len(cs):
                    cand.append((float(np.mean(vs)), float(np.std(vs)), "ce=%s,L=%s" % (ce, lk)))
                    if ce == "1": byL[lk] = float(np.mean(vs))
    elif m in cs[0].get("xx", {}):
        for lk in cs[0]["xx"][m]:
            vs = [nz(c["xx"][m][lk], c) for c in cs if lk in c["xx"].get(m, {})]
            if len(vs) == len(cs):
                cand.append((float(np.mean(vs)), float(np.std(vs)), "L=%s" % lk))
                byL[lk] = float(np.mean(vs))
    elif m == "naive":
        vs = [nz(c["naive"], c) for c in cs]
        cand.append((float(np.mean(vs)), float(np.std(vs)), "-"))
    else:
        vs = [nz(c["baselines"][m], c) for c in cs if m in c.get("baselines", {})]
        if vs: cand.append((float(np.mean(vs)), float(np.std(vs)), "-"))
    return (max(cand) if cand else (float("nan"),) * 2 + ("-",)), byL


TAB, DET = {}, {}
for k in COLS:
    for m in ORDER:
        TAB[(m, k)], DET[(m, k)] = cell_stats(byk[k], m)

hdr = "  ".join("%-16s" % ("%s cap%d%%" % (t.replace("cap_", ""), 100 * c)) for t, c in COLS)
print("=" * (26 + len(hdr)))
print("CAPPED RESULTS -- normalised value (0 = best feasible constant policy, 1 = capped oracle)")
print("10 seeds, n=400, matched Gamma* = 5; best (c_eps, L) cell per method, +- across-seed sd")
print("=" * (26 + len(hdr)))
print("%-16s %-5s %s" % ("method", "fam", hdr))
print("-" * (26 + len(hdr)))
ranks = {k: sorted(ORDER, key=lambda m: -TAB[(m, k)][0]) for k in COLS}
for m in ORDER:
    row = "%-16s %-5s " % (LBL.get(m, m), FAM[m])
    for k in COLS:
        v, sd, _ = TAB[(m, k)]
        r = ranks[k].index(m) + 1
        row += "%7.3f+-%.3f#%-2d " % (v, sd, r)
    print(row)
print("-" * (26 + len(hdr)))
print("%-16s %-5s " % ("headroom", "") +
      "  ".join("%-16.3f" % np.mean([c["sc"] for c in byk[k]]) for k in COLS))
print("%-16s %-5s " % ("capped oracle", "") +
      "  ".join("%-16.4f" % np.mean([c["refs"]["oracle_cap"] for c in byk[k]]) for k in COLS))
print("\n#N = rank within the column, 1 = best of the 13 methods.\n")

print("=" * 78)
print("THE SAME COMPARISON STRATIFIED BY L (c_eps = 1) -- where the transport term earns its place")
print("=" * 78)
for k in COLS:
    print("\n%s, cap %d%%" % (k[0], 100 * k[1]))
    print("   %-16s %9s %9s %9s" % ("method", "L=inf", "L=3", "L=1"))
    for m in OW + OX + XX:
        d = DET[(m, k)]
        if not d: continue
        print("   %-16s %9s %9s %9s" % (LBL.get(m, m),
              *["%+.3f" % d[l] if l in d else "-" for l in ("inf", "3", "1")]))
    pa = {l: DET[("IPW-O-W", k)].get(l) for l in ("inf", "3", "1")}
    pb = {l: DET[("IPW-O-X", k)].get(l) for l in ("inf", "3", "1")}
    print("   %-16s %9s %9s %9s" % ("--> IPW margin",
          *["%+.3f" % (pa[l] - pb[l]) if pa[l] is not None and pb[l] is not None else "-"
            for l in ("inf", "3", "1")]))
