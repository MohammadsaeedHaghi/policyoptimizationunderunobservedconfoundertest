"""Score the first-prime oracle (best-means policy) on the realized TEST outcomes (te.Ypot), the same ruler as the
methods, instead of on the true means. Update both result JSONs, the PRIME_RES / PRIME_N10K chart oracle lines, the
table oracle rows, the finding/footer numbers, and make N_test explicit in the HTML."""
import json, re
from pathlib import Path
ROOT = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
ORC = {("1500", "uncap"): 0.8209, ("1500", "cap"): 0.818, ("10000", "uncap"): 0.8226, ("10000", "cap"): 0.8193}

# 1) update result JSONs
for fn, N in (("prime_results.json", "1500"), ("prime_n10000_results.json", "10000")):
    p = ROOT / "assets/exp_first_prime" / fn; d = json.loads(p.read_text())
    for rg in ("uncap", "cap"): d["regimes"][rg]["oracle"] = ORC[(N, rg)]
    p.write_text(json.dumps(d, indent=2))

h = (ROOT / "index.html").read_text()

# 2) chart vars: Oracle series + oracle hline
for var, N in (("PRIME_RES", "1500"), ("PRIME_N10K", "10000")):
    m = re.search(r'var %s = (.*?);\n' % var, h); P = json.loads(m.group(1))
    for rg in ("uncap", "cap"):
        orc = ORC[(N, rg)]; key = rg + "_value"
        for s in P[key]["series"]:
            if s["id"] == "Oracle": s["y"] = [orc] * len(s["y"])
        for hl in P[key].get("hlines", []):
            if "oracle" in hl.get("label", "").lower(): hl["y"] = orc; hl["label"] = "oracle (on test)"
    h = h[:m.start()] + "var %s = " % var + json.dumps(P, separators=(",", ":")) + ";\n" + h[m.end():]

# 3) table oracle rows
def orc_row(orc, n=7):
    return ("<tr><td><span style='display:inline-block;width:9px;height:9px;border-radius:2px;background:#444444;margin-right:6px'></span>Oracle (best-means policy, on test)</td>"
            + "".join("<td>%.3f</td>" % orc for _ in range(n)) + "</tr>")
boxes = {"restab-prime-uncap": ("1500", "uncap"), "restab-prime-cap": ("1500", "cap"),
         "restab-p10k-uncap": ("10000", "uncap"), "restab-p10k-cap": ("10000", "cap")}
orc_pat = re.compile(r"<tr><td><span[^>]*background:#444444[^>]*></span>Oracle[^<]*</td>(?:<td>[^<]*</td>)+</tr>")
for bid, (N, rg) in boxes.items():
    bm = re.search(r'(<div class="restab-box" id="%s">)(.*?)(</div>)' % bid, h, re.S)
    assert bm, bid
    seg = bm.group(2); seg2, k = orc_pat.subn(orc_row(ORC[(N, rg)]), seg)
    assert k == 1, "oracle row sub count %d in %s" % (k, bid)
    h = h[:bm.start(2)] + seg2 + h[bm.end(2):]

# 4) text: finding + N=10000 footer + make N_test explicit + note oracle is on test
def rep(old, new, n=1):
    global h; c = h.count(old); assert c == n, "expected %d of %r, found %d" % (n, old[:50], c); h = h.replace(old, new, n)
rep("oracle 0.820, essentially the ceiling", "oracle 0.821 (realised on the test set), essentially the ceiling")
rep("uncapped 0.823 vs oracle 0.820, capped 0.819 vs oracle 0.817 (both at the matched\n    \\(\\Gamma\\approx7.46\\))",
    "uncapped 0.823 vs oracle 0.823, capped 0.819 vs oracle 0.819 (they coincide because DoublyRobust-O-X deploys\n    the oracle's own policy; both at the matched \\(\\Gamma\\approx7.46\\))")
# make N_test explicit (first-prime is the only tab using "20k test set")
rep("evaluated on a 20k test\n    set", "evaluated on a test set of \\(N_{\\text{test}}=20000\\)")          # N=1500 results intro
rep("trained at \\(N=10000\\) (one seed, evaluated on a 20k test set)", "trained at \\(N=10000\\) (one seed), evaluated on a test set of \\(N_{\\text{test}}=20000\\)")
(ROOT / "index.html").write_text(h)
print("oracle now scored on test outcomes:", {k: v for k, v in ORC.items()})
print("remaining '20k test set' occurrences:", h.count("20k test set"))
