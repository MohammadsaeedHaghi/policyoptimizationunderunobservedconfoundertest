#!/usr/bin/env python3
"""Detailed capped table: every method, every cell of the (c_eps, L) grid summarised, per budget."""
import json, glob, collections
import numpy as np

OW = ["IPW-O-W", "DoublyRobust-O-W", "Hajek-O-W"]
OX = ["IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X"]
XX = ["IPW-X-X", "DoublyRobust-X-X", "Direct-X-X"]
BL = ["Kallus", "SharpHess-kNN", "SharpHess"]
LBL = {"DoublyRobust-O-W": "DR-O-W", "DoublyRobust-O-X": "DR-O-X", "DoublyRobust-X-X": "DR-X-X",
       "SharpHess": "Hess (neural)", "SharpHess-kNN": "Hess (k-NN)", "naive": "naive plug-in"}

byk = collections.defaultdict(list)
for f in glob.glob("capped_v2/*_s*.json"):
    d = json.load(open(f)); byk[(d["tag"], d["cap"])].append(d["cell"])


def nz(v, c):
    return (v - c["bc"]) / c["sc"]


for (tag, cap), cs in sorted(byk.items()):
    print("=" * 96)
    print("%s   cap = %.0f%%   (%d seeds, n=400, matched Gamma* = 5)   headroom %.3f"
          % (tag, 100 * cap, len(cs), float(np.mean([c["sc"] for c in cs]))))
    r0 = cs[0]["refs"]
    print("   references: capped oracle %.4f | never-treat %.4f | random-%d%% %.4f"
          % (float(np.mean([c["refs"]["oracle_cap"] for c in cs])),
             float(np.mean([c["refs"]["never"] for c in cs])), 100 * cap,
             float(np.mean([c["refs"]["rand_cap"] for c in cs]))))
    print()
    print("   %-16s %8s %7s | %s" % ("method", "best", "sd", "every (c_eps, L) cell, mean over seeds"))
    print("   " + "-" * 90)
    best_of = {}
    for m in OW + OX + XX:
        cells_, isxx = (cs[0]["xx"] if m in XX else cs[0]["grid"]).get(m, {}), m in XX
        if not cells_:
            print("   %-16s %8s" % (LBL.get(m, m), "MISSING")); continue
        detail, cand = [], []
        if isxx:
            for lk in cells_:
                vs = [nz(c["xx"][m][lk], c) for c in cs if lk in c["xx"].get(m, {})]
                if len(vs) == len(cs):
                    cand.append((float(np.mean(vs)), float(np.std(vs))))
                    detail.append("L=%s %+.3f" % (lk, np.mean(vs)))
        else:
            for ce in cells_:
                for lk in cells_[ce]:
                    vs = [nz(c["grid"][m][ce][lk], c) for c in cs if lk in c["grid"][m].get(ce, {})]
                    if len(vs) == len(cs):
                        cand.append((float(np.mean(vs)), float(np.std(vs))))
                        detail.append("ce=%s L=%s %+.3f" % (ce, lk, np.mean(vs)))
        if not cand:
            print("   %-16s %8s" % (LBL.get(m, m), "no cell")); continue
        b = max(cand); best_of[m] = b
        print("   %-16s %8.3f %7.3f | %s" % (LBL.get(m, m), b[0], b[1], "  ".join(detail)))
    nv = [nz(c["naive"], c) for c in cs]
    best_of["naive"] = (float(np.mean(nv)), float(np.std(nv)))
    print("   %-16s %8.3f %7.3f |" % ("naive plug-in", np.mean(nv), np.std(nv)))
    for m in BL:
        vs = [nz(c["baselines"][m], c) for c in cs if m in c.get("baselines", {})]
        if vs:
            best_of[m] = (float(np.mean(vs)), float(np.std(vs)))
            print("   %-16s %8.3f %7.3f |" % (LBL.get(m, m), np.mean(vs), np.std(vs)))

    marg = {}
    for a, b in (("IPW-O-W", "IPW-O-X"), ("DoublyRobust-O-W", "DoublyRobust-O-X")):
        dd = []
        for ce in cs[0]["grid"].get(a, {}):
            for lk in cs[0]["grid"][a][ce]:
                pa = [nz(c["grid"][a][ce][lk], c) for c in cs if lk in c["grid"][a].get(ce, {})]
                pb = [nz(c["grid"][b]["1"][lk], c) for c in cs if lk in c["grid"][b].get("1", {})]
                if len(pa) == len(pb) == len(cs):
                    dd.append(float(np.mean(np.array(pa) - np.array(pb))))
        if dd:
            marg[a] = (float(np.mean(dd)), float(np.max(dd)))
    ow = max(best_of[m][0] for m in OW if m in best_of)
    riv = {m: v[0] for m, v in best_of.items() if m not in OW}
    tm = max(riv, key=riv.get)
    print()
    print("   transport margin, paired at identical (c_eps, L): %s"
          % {k: "mean %+.3f / max %+.3f" % v for k, v in marg.items()})
    print("   best O-W %.3f | toughest rival %s %.3f | gap %+.3f %s"
          % (ow, LBL.get(tm, tm), riv[tm], ow - riv[tm], "PASS" if ow > riv[tm] else "LOSS"))
