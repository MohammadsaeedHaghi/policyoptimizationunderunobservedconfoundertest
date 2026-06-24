"""Append a Γ=20 column to the two N=10000 first-prime tables (restab-p10k-uncap / restab-p10k-cap).
XX rows repeat their Γ-free value; OX rows use the freshly solved Γ=20 values; oracle repeats its on-test value.
Bolds the best method in the new column."""
import re, json
from pathlib import Path
ROOT = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
HERE = ROOT / "assets" / "exp_first_prime"
RJ = json.loads((HERE / "prime_n10000_results.json").read_text())
G20 = json.loads((HERE / "prime_n10000_gamma20.json").read_text())
ROWS = ["IPW-X-X", "DoublyRobust-X-X", "Direct-X-X", "IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X"]
h = (ROOT / "index.html").read_text()
for rg, bid in (("uncap", "restab-p10k-uncap"), ("cap", "restab-p10k-cap")):
    R = RJ["regimes"][rg]; orc = R["oracle"]
    vals = {m: (R["value"][m][0] if m.endswith("-X-X") else G20[rg][m]) for m in ROWS}
    best = max(vals.values())
    bm = re.search(r'(id="%s">\s*)(.*?)(\s*</table>)' % bid, h, re.S)
    assert bm, bid
    t = bm.group(2)
    # header: add a Γ=20 th before the first </tr>
    t = re.sub(r'(<tr><th>method</th>(?:<th>[^<]*</th>)+?)(</tr>)', r'\1<th>Γ=20</th>\2', t, count=1)
    # method rows: append a td before each row's </tr>
    for m in ROWS:
        b = " style='font-weight:700'" if abs(vals[m] - best) < 1e-9 else ""
        cell = "<td%s>%.3f</td>" % (b, vals[m])
        pat = re.compile(r'(</span>' + re.escape(m) + r'</td>(?:<td[^>]*>[^<]*</td>)+)(</tr>)')
        t2, k = pat.subn(r'\1' + cell + r'\2', t, count=1)
        assert k == 1, "row %s not matched in %s" % (m, bid)
        t = t2
    # oracle row
    pat = re.compile(r'(</span>Oracle \(best-means policy, on test\)</td>(?:<td[^>]*>[^<]*</td>)+)(</tr>)')
    t, k = pat.subn(r'\1<td>%.3f</td>' % orc + r'\2', t, count=1)
    assert k == 1, "oracle row not matched in " + bid
    h = h[:bm.start(2)] + t + h[bm.end(2):]
    print("%s: appended Γ=20 column ->" % rg, {m: vals[m] for m in ROWS}, "oracle", orc)
(ROOT / "index.html").write_text(h)
print("done")
