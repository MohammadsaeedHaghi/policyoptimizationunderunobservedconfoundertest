"""Build numeric results tables (realised test E[Y] by Γ) for the Fourth (continuous) and Fifth (discrete)
Wasserstein experiments from their saved JSONs, and splice each table into the right Results box in index.html.
Bolds the higher of (-O-W vs -O-X) within each estimator family per Γ. Idempotent (replaces prior table)."""
import json, re
from pathlib import Path
ROOT = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
COL = {"IPW": "#1f77b4", "DoublyRobust": "#2ca02c", "Hajek": "#9467bd", "Direct": "#ff7f0e", "Oracle": "#444444"}

def fmt(v): return ("%.3f" % v).replace("-0.000", "0.000")

def row(label, dot, vals, bold_mask):
    cells = "".join("<td%s>%s</td>" % (" style='font-weight:700'" if b else "", fmt(v)) for v, b in zip(vals, bold_mask))
    return "<tr><td><span style='display:inline-block;width:9px;height:9px;border-radius:2px;background:%s;margin-right:6px'></span>%s</td>%s</tr>" % (dot, label, cells)

def build_table(gammas, rows_spec, oracle, caption):
    # rows_spec: list of (label, family, vals).  Bold the larger of the O-W/O-X pair per family per Γ.
    G = len(gammas)
    by_fam = {}
    for lbl, fam, vals in rows_spec:
        if lbl.endswith("-O-W") or lbl.endswith("-O-X"):
            by_fam.setdefault(fam, {})[lbl[-3:]] = vals
    head = "<tr><th>method</th>" + "".join("<th>Γ=%s</th>" % (int(g) if float(g) == int(g) else g) for g in gammas) + "</tr>"
    body = []
    for lbl, fam, vals in rows_spec:
        mask = [False] * G
        suf = lbl[-3:]
        if suf in ("O-W", "O-X") and fam in by_fam and "O-W" in by_fam[fam] and "O-X" in by_fam[fam]:
            other = by_fam[fam]["O-X" if suf == "O-W" else "O-W"]
            mask = [vals[i] > other[i] + 1e-9 for i in range(G)]
        body.append(row(lbl, COL.get(fam, "#888"), vals, mask))
    orc = row("Oracle (ceiling)", COL["Oracle"], [oracle] * G, [False] * G)
    return ("<table class='restab'>%s%s%s</table>\n<p class='muted' style='margin:6px 0 0'>%s</p>"
            % (head, "".join(body), orc, caption))

# ---------- Fourth (continuous) ----------
fast = json.loads((ROOT / "assets/exp_wass/wass_fast.json").read_text())
wass = json.loads((ROOT / "assets/exp_wass/wass_wasserstein.json").read_text())
G4 = fast["gammas"]; flat = lambda v: [v] * len(G4)
rows4 = [
    ("IPW-X-X", "IPW", flat(fast["IPW-X-X"])),
    ("DoublyRobust-X-X", "DoublyRobust", flat(fast["DoublyRobust-X-X"])),
    ("Direct-X-X", "Direct", flat(fast["Direct-X-X"])),
    ("IPW-O-X", "IPW", fast["IPW-O-X"]),
    ("IPW-O-W", "IPW", [round(v, 3) for v in wass["IPW-O-W"]]),
    ("DoublyRobust-O-X", "DoublyRobust", fast["DoublyRobust-O-X"]),
    ("DoublyRobust-O-W", "DoublyRobust", [round(v, 3) for v in wass["DoublyRobust-O-W"]]),
    ("Hajek-O-X", "Hajek", fast["Hajek-O-X"]),
]
tbl4 = build_table(G4, rows4, fast["oracle"],
    "Realised test E[Y] by Γ, continuous X, N_train=1000, single seed. Bold = the Wasserstein (-O-W) or box (-O-X) member that wins its estimator family at that Γ. DoublyRobust-O-W beats DoublyRobust-O-X at every Γ (the box-only DR collapses to ~0.13 at Γ=4,8); IPW-O-W beats IPW-O-X at the matched Γ=8.")

# ---------- Fifth (discrete) ----------
disc = json.loads((ROOT / "assets/exp_wass_disc/disc_results.json").read_text())
V = disc["value_by_gamma"]; G5 = disc["gammas"]
rows5 = [
    ("IPW-X-X", "IPW", V["IPW-X-X"]),
    ("DoublyRobust-X-X", "DoublyRobust", V["DoublyRobust-X-X"]),
    ("IPW-O-X", "IPW", V["IPW-O-X"]),
    ("IPW-O-W", "IPW", V["IPW-O-W"]),
    ("DoublyRobust-O-X", "DoublyRobust", V["DoublyRobust-O-X"]),
    ("DoublyRobust-O-W", "DoublyRobust", V["DoublyRobust-O-W"]),
]
tbl5 = build_table(G5, rows5, disc["oracle"],
    "Realised test E[Y] by Γ, discrete X (9 levels), N_train=1000, single seed. Bold = family winner at that Γ. DoublyRobust-O-W beats DoublyRobust-O-X at every Γ>1 (box-only DR collapses 0.262 to 0.132); IPW-O-W beats IPW-O-X at Γ=2 and the matched Γ=8.")

h = (ROOT / "index.html").read_text()

def splice(h, anchor, marker_id, table_html):
    block = '<div class="restab-box" id="%s">\n%s\n</div>' % (marker_id, table_html)
    # remove any prior block with that id
    h = re.sub(r'<div class="restab-box" id="%s">.*?</div>\s*(?=<h4)' % marker_id, "", h, flags=re.S)
    assert anchor in h, "anchor missing: " + anchor
    return h.replace(anchor, anchor + "\n" + block, 1)

h = splice(h, '<div class="ichart" id="ichart-wass-value" style="max-width:900px"></div>', "restab-wass", tbl4)
h = splice(h, '<div class="ichart" id="ichart-disc-value" style="max-width:900px"></div>', "restab-disc", tbl5)

# add a tiny CSS for .restab once
if ".restab{" not in h:
    css = "\n.restab{border-collapse:collapse;margin:10px 0 2px;font-size:.86rem;width:100%;max-width:640px}\n.restab th,.restab td{border:1px solid #e2e8f0;padding:4px 9px;text-align:center}\n.restab th:first-child,.restab td:first-child{text-align:left;white-space:nowrap}\n.restab th{background:#f1f5f9;font-weight:600}\n.restab tr:last-child td{background:#fafafa;font-style:italic}\n"
    h = h.replace("</style>", css + "</style>", 1)

(ROOT / "index.html").write_text(h)
print("spliced results tables into Fourth + Fifth tabs")
