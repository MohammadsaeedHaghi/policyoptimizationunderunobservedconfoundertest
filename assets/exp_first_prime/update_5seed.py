"""Update the N=10000 first-prime value chart (5-seed means) and tables (mean +- SD over 5 seeds, all 8 Γ incl 20)."""
import json, re
from pathlib import Path
ROOT = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
HERE = ROOT / "assets" / "exp_first_prime"
A = json.loads((HERE / "prime_n10000_5seed_results.json").read_text())
GA = A["gammas"]; nS = len(A["seeds"])
FAM = {"IPW": "#1f77b4", "DoublyRobust": "#2ca02c", "Direct": "#ff7f0e", "Hajek": "#9467bd", "Oracle": "#444444"}
ROWS = ["IPW-X-X", "DoublyRobust-X-X", "Direct-X-X", "IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X"]
def fam(m): return m.split("-")[0]
def gk(g): g = float(g); return str(int(g)) if g == int(g) else str(g)

h = (ROOT / "index.html").read_text()

# 1) value chart: 5-seed means at the chart's Γ (first 7: drop Γ=20 to keep the axis 1..8)
m = re.search(r'var PRIME_N10K = (.*?);\n', h); P = json.loads(m.group(1))
for rg in ("uncap", "cap"):
    R = A["regimes"][rg]; cx = P[rg + "_value"]["x"]; idx = [GA.index(g) for g in cx]
    for s in P[rg + "_value"]["series"]:
        if s["id"] == "Oracle": s["y"] = [R["oracle_mean"]] * len(cx)
        elif s["id"] in R["mean"]: s["y"] = [R["mean"][s["id"]][i] for i in idx]
    for hl in P[rg + "_value"].get("hlines", []):
        if "oracle" in hl.get("label", "").lower(): hl["y"] = R["oracle_mean"]
    P[rg + "_value"]["note"] = "CAPPED" if rg == "cap" else "UNCAPPED"
    P[rg + "_value"]["note"] += " (treat<=%s): realised test E[Y] vs Γ, N_train=10000, 5-seed mean. Colour=family, shape=uncertainty set (XX dashed, OX squares)." % ("50%" if rg == "cap" else "100%")
h = h[:m.start()] + "var PRIME_N10K = " + json.dumps(P, separators=(",", ":")) + ";\n" + h[m.end():]

# 2) tables: mean +- SD, all 8 Γ, bold best mean per Γ
def table(rg):
    R = A["regimes"][rg]; mean = R["mean"]; sd = R["sd"]
    best = [max(mean[m][i] for m in ROWS) for i in range(len(GA))]
    head = "<tr><th>method</th>" + "".join("<th>Γ=%s</th>" % gk(g) for g in GA) + "</tr>"
    rows = []
    for m in ROWS:
        cells = "".join("<td%s>%.3f&plusmn;%.3f</td>" % (" style='font-weight:700'" if abs(mean[m][i] - best[i]) < 1e-9 else "", mean[m][i], sd[m][i]) for i in range(len(GA)))
        rows.append("<tr><td><span style='display:inline-block;width:9px;height:9px;border-radius:2px;background:%s;margin-right:6px'></span>%s</td>%s</tr>" % (FAM[fam(m)], m, cells))
    orow = "<tr><td><span style='display:inline-block;width:9px;height:9px;border-radius:2px;background:%s;margin-right:6px'></span>Oracle (best-means policy, on test)</td>%s</tr>" % (FAM["Oracle"], "".join("<td>%.3f&plusmn;%.3f</td>" % (R["oracle_mean"], R["oracle_sd"]) for _ in GA))
    cap = "treat &le; 100%" if rg == "uncap" else "treat &le; 50%"
    return ("<table class='restab'>%s%s%s</table>\n<p class='muted' style='margin:6px 0 0'>%s: realised test E[Y] by Γ, N_train=10000, <strong>mean &plusmn; SD over 5 seeds</strong>. Bold = best mean at that Γ. XX methods are Γ-free; OX sweep Γ (incl. Γ=20).</p>"
            % (head, "".join(rows), orow, cap))
for rg, bid in (("uncap", "restab-p10k-uncap"), ("cap", "restab-p10k-cap")):
    h = re.sub(r'(<div class="restab-box" id="%s">\s*).*?(\s*</div>)' % bid, lambda mm: mm.group(1) + table(rg) + mm.group(2), h, count=1, flags=re.S)

# 3) intro + footer text -> say 5 seeds
h = h.replace("trained at \\(N=10000\\) (one seed), evaluated on a test set of \\(N_{\\text{test}}=20000\\)",
              "trained at \\(N=10000\\) over <strong>5 seeds</strong> (mean \\(\\pm\\) SD), evaluated on a test set of \\(N_{\\text{test}}=20000\\) per seed")
(ROOT / "index.html").write_text(h)
print("updated value chart (5-seed means) + tables (mean+-SD, 8 Γ)")
for rg in ("uncap", "cap"):
    R = A["regimes"][rg]; print("[%s] oracle=%.4f+-%.4f  DR-O-X@Γ=20=%.4f+-%.4f" % (rg, R["oracle_mean"], R["oracle_sd"], R["mean"]["DoublyRobust-O-X"][-1], R["sd"]["DoublyRobust-O-X"][-1]))
