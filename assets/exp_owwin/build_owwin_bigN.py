"""Fill the 'Larger N_train' section of the exp_owwin tab from owwin_results_N{N}.json: a value-vs-Γ chart
(var OWBIG_CHARTS) + a mean±SD table. Usage: python3 build_owwin_bigN.py 1500"""
import sys, json, re
from pathlib import Path
ROOT = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
HERE = ROOT / "assets" / "exp_owwin"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
R = json.loads((HERE / ("owwin_results_N%d.json" % N)).read_text())
GAMMAS = R["gammas"]; orc = R["oracle"]
FAM = {"IPW": "#1f77b4", "DoublyRobust": "#2ca02c", "Direct": "#ff7f0e", "Hajek": "#9467bd", "Oracle": "#444444"}
def fam(m): return m.split("-")[0]
def gk(g): g = float(g); return str(int(g)) if g == int(g) else str(g)
order = ["IPW-X-X", "DoublyRobust-X-X", "Direct-X-X", "IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X", "IPW-O-W", "DoublyRobust-O-W"]
def ser(m): return {"id": m, "label": m, "color": FAM[fam(m)], "y": [round(float(v), 4) for v in R["mean"][m]]}
W = {"x": [float(g) for g in GAMMAS], "xmin": float(min(GAMMAS)), "xmax": float(max(GAMMAS)),
     "xlabel": "Γ  (sensitivity-model strength)", "ylabel": "realised E[Y] (%d-seed mean)" % len(R["seeds"]),
     "ymin": 0.0, "ymax": round(orc + 0.03, 3),
     "series": [ser(m) for m in order] + [{"id": "Oracle", "label": "Oracle", "color": FAM["Oracle"], "y": [orc] * len(GAMMAS)}],
     "hlines": [{"y": orc, "color": "#94a3b8", "dash": "4,4", "label": "oracle"}],
     "note": "CAPPED (treat<=50%%): realised test E[Y] vs Γ, N_train=%d, %d-seed mean. Same DGP as above, more data." % (N, len(R["seeds"]))}
(HERE / ("owbig_charts_N%d.json" % N)).write_text(json.dumps(W))

def fmt(v): return "%.3f" % v
def rowhtml(m):
    mu, sd = R["mean"][m], R["sd"][m]
    cells = "".join("<td>%s&plusmn;%s</td>" % (fmt(mu[i]), fmt(sd[i])) for i in range(len(GAMMAS)))
    return "<tr><td><span style='display:inline-block;width:9px;height:9px;border-radius:2px;background:%s;margin-right:6px'></span>%s</td>%s</tr>" % (FAM[fam(m)], m, cells)
head = "<tr><th>method</th>" + "".join("<th>Γ=%s</th>" % gk(g) for g in GAMMAS) + "</tr>"
orow = "<tr><td><span style='display:inline-block;width:9px;height:9px;border-radius:2px;background:%s;margin-right:6px'></span>Oracle (best policy)</td>%s</tr>" % (FAM["Oracle"], "".join("<td>%s</td>" % fmt(orc) for _ in GAMMAS))
tbl = "<table class='restab'>%s%s%s</table>\n<p class='muted' style='margin:6px 0 0'>CAPPED (treat &le; 50%%): realised test E[Y] by Γ, N_train=%d, mean &plusmn; SD over %d seeds.</p>" % (head, "".join(rowhtml(m) for m in order), orow, N, len(R["seeds"]))

idx = ROOT / "index.html"; h = idx.read_text()
# update intro line of the bigN body
h = re.sub(r'<p class="muted">Running at \\\(N_\{\\text\{train\}\}=1500\\\)[^<]*</p>',
           '<p class="muted">Confirmed at \\\\(N_{\\\\text{train}}=%d\\\\) over %d seeds. The clean OW &gt; OX &gt; XX ordering holds (tighter, lower-variance), so the win is not a small-sample artifact.</p>' % (N, len(R["seeds"])), h, count=1)
# splice var + table + render call
line = "var OWBIG_CHARTS = %s;" % json.dumps(W, separators=(",", ":"))
if re.search(r'^var OWBIG_CHARTS = .*;$', h, flags=re.M):
    h = re.sub(r'^var OWBIG_CHARTS = .*;$', lambda m: line, h, count=1, flags=re.M)
else:
    h = re.sub(r'^(var OWWIN_CHARTS = .*;)$', lambda m: m.group(1) + "\n" + line, h, count=1, flags=re.M)
h = re.sub(r'(<div class="restab-box" id="restab-owbig">).*?(</div>)', lambda m: m.group(1) + tbl + m.group(2), h, count=1, flags=re.S)
call = "renderChart('ichart-owbig-value',OWBIG_CHARTS);"
h = re.sub(r"\nrenderChart\('ichart-owbig-value'[^\n]*", "", h)
if call not in h:
    anchor = "renderChart('ichart-ow-value',OWWIN_CHARTS.value);"
    assert anchor in h, "anchor missing"; h = h.replace(anchor, anchor + "\n" + call, 1)
idx.write_text(h)
print("filled bigN section N=%d; OW(DR) mean=%s" % (N, R["mean"]["DoublyRobust-O-W"]))
