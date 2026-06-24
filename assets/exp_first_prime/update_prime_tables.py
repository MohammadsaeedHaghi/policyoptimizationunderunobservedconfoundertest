"""Rewrite the first-prime result tables to include the now-complete OW methods (IPW-O-W, DoublyRobust-O-W),
and inject a short finding sentence. Replaces the restab-box contents for both regimes in index.html."""
import json, re
from pathlib import Path
ROOT = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
HERE = ROOT / "assets" / "exp_first_prime"
R = json.loads((HERE / "prime_results.json").read_text())
OW = {"uncap": json.loads((HERE / "prime_ow_uncap_results.json").read_text()),
      "cap": json.loads((HERE / "prime_ow_cap_results.json").read_text())}
G = R["gammas"]
def gk(g): g = float(g); return str(int(g)) if g == int(g) else str(g)
FAM = {"IPW": "#1f77b4", "DoublyRobust": "#2ca02c", "Direct": "#ff7f0e", "Hajek": "#9467bd", "Oracle": "#444444"}
def fam(m): return m.split("-")[0]
ORDER = ["IPW-X-X", "DoublyRobust-X-X", "Direct-X-X", "IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X", "IPW-O-W", "DoublyRobust-O-W"]

def vals(rg, m):
    if m in ("IPW-O-W", "DoublyRobust-O-W"): return OW[rg]["value"][m]      # full-grid w/ None
    return R["regimes"][rg]["value"][m]
def fmt(v): return "&middot;" if v is None else "%.3f" % v

def table(rg):
    orc = R["regimes"][rg]["oracle"]
    head = "<tr><th>method</th>" + "".join("<th>&Gamma;=%s</th>" % gk(g) for g in G) + "</tr>"
    # best per Γ over methods with a value there
    best = []
    for i in range(len(G)):
        vv = [vals(rg, m)[i] for m in ORDER if vals(rg, m)[i] is not None]
        best.append(max(vv) if vv else None)
    rows = []
    for m in ORDER:
        v = vals(rg, m)
        cells = "".join("<td%s>%s</td>" % (" style='font-weight:700'" if (v[i] is not None and best[i] is not None and abs(v[i] - best[i]) < 1e-9) else "", fmt(v[i])) for i in range(len(G)))
        rows.append("<tr><td><span style='display:inline-block;width:9px;height:9px;border-radius:2px;background:%s;margin-right:6px'></span>%s</td>%s</tr>" % (FAM[fam(m)], m, cells))
    orow = "<tr><td><span style='display:inline-block;width:9px;height:9px;border-radius:2px;background:%s;margin-right:6px'></span>Oracle (best-means policy, on test)</td>%s</tr>" % (FAM["Oracle"], "".join("<td>%s</td>" % ("%.3f" % orc) for _ in G))
    cap = "treat &le; 100%" if rg == "uncap" else "treat &le; 50%"
    note = ("%s: realised test E[Y] by &Gamma;, N_train=%d, single seed. Bold = best method at that &Gamma;. XX flat; OX over the full &Gamma; grid; OW (Wasserstein) solved on a trimmed grid {1,4,7.46} (&middot; = not solved). "
            % (cap, R["N_train"]))
    return "<table class='restab'>%s%s%s</table>\n<p class='muted' style='margin:6px 0 0'>%s</p>" % (head, "".join(rows), orow, note)

h = (ROOT / "index.html").read_text()
for rg, tid in (("uncap", "restab-prime-uncap"), ("cap", "restab-prime-cap")):
    pat = re.compile(r'(<div class="restab-box" id="%s">\s*).*?(\s*</div>)' % tid, re.S)
    assert pat.search(h), "restab box not found: " + tid
    h = pat.sub(lambda mm: mm.group(1) + table(rg) + mm.group(2), h, count=1)
(ROOT / "index.html").write_text(h)
print("updated first-prime tables (both regimes) with OW rows")
for rg in ("uncap", "cap"):
    print(rg, "OW @matched 7.46: IPW-O-W=%s DR-O-W=%s" % (vals(rg, "IPW-O-W")[G.index(R["matched_gamma"])], vals(rg, "DoublyRobust-O-W")[G.index(R["matched_gamma"])]))
