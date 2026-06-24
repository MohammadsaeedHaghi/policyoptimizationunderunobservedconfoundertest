"""After both add_prime_ow_gamma8 runs finish (Γ=8 merged into prime_ow_<regime>_results.json), patch the PRIME_RES
value + policy charts in index.html so the OW lines include the Γ=8 point. Single write (no race)."""
import json, re
from pathlib import Path
ROOT = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
HERE = ROOT / "assets" / "exp_first_prime"
OW = {"uncap": json.loads((HERE / "prime_ow_uncap_results.json").read_text()),
      "cap": json.loads((HERE / "prime_ow_cap_results.json").read_text())}
h = (ROOT / "index.html").read_text()
m = re.search(r'var PRIME_RES = (.*?);\n', h); P = json.loads(m.group(1))
for rg in ("uncap", "cap"):
    res = OW[rg]
    # value: replace OW series y with the full (now Γ=8-filled) arrays
    for s in P[rg + "_value"]["series"]:
        if s["id"] in ("IPW-O-W", "DoublyRobust-O-W"):
            s["y"] = res["value"][s["id"]]
    # policy at Γ=8: set the OW series y to the solved grid
    sl = P[rg + "_policy"]["seriesByGamma"].get("8")
    assert sl is not None, "no Γ=8 policy key in " + rg
    for s in sl:
        if s["id"] in ("IPW-O-W", "DoublyRobust-O-W"):
            s["y"] = res["policy"][s["id"]]["8"]
    print(rg, "Γ=8 -> IPW-O-W=%s DR-O-W=%s" % (res["value"]["IPW-O-W"][res["gammas"].index(8.0)],
                                               res["value"]["DoublyRobust-O-W"][res["gammas"].index(8.0)]))
h = h[:m.start()] + "var PRIME_RES = " + json.dumps(P, separators=(",", ":")) + ";\n" + h[m.end():]
(ROOT / "index.html").write_text(h)
print("patched PRIME_RES value + policy with Γ=8 OW points")
