"""Fast, NO-Gurobi screening of the 10 candidate DGP frameworks.

For each idea it computes truth-level quantities (exact, noise-free) that predict whether the DGP is a
good candidate for the OW-beats-everyone study, BEFORE running the heavy solver search:

  oracle   = value of the optimal observable policy (treat where E[CATE|X]>0)
  naive    = value of the CONFOUNDED-optimal policy (treat where the observed treated-minus-control
             contrast is positive) -> what a method that ignores confounding would chase
  gap      = oracle - naive  (how badly confounding misleads the naive choice; BIG = lots of headroom
             for a robust/balancing method to win)
  ctrl/all = never-treat and treat-all values (the trivial fallbacks)
  ofrac    = fraction of X levels the oracle treats (want partial, ~0.3-0.7, not 0 or 1)
  agree    = fraction of levels where naive agrees with oracle (LOW = confounding flips the decision)
  confound = max over X of |observed contrast - true CATE| (raw selection bias magnitude)

Run:  python3 assets/dgp_ideas/screen.py
"""
import importlib.util, glob, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

def _load(path):
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return name, mod

rows = []
for path in sorted(glob.glob(os.path.join(HERE, "dgp[0-9]*.py"))):
    name, m = _load(path); g = m.grid_truth()
    z = np.zeros_like(g["oracle"]); o = np.ones_like(g["oracle"])
    oracle = m.exact_value(g["oracle"]); naive = m.exact_value(g["naive"])
    ctrl = m.exact_value(z); allt = m.exact_value(o)
    ofrac = float(np.mean(g["oracle"])); agree = float(np.mean(g["oracle"] == g["naive"]))
    confound = float(np.max(np.abs(g["obs_contrast"] - g["cate"])))
    rows.append((name, oracle, naive, oracle - naive, ctrl, allt, ofrac, agree, confound))

rows.sort(key=lambda r: -r[3])
hdr = ("idea", "oracle", "naive", "gap", "ctrl", "all", "ofrac", "agree", "confnd")
print("%-36s %7s %7s %7s %7s %7s %6s %6s %7s" % hdr)
print("-" * 96)
for r in rows:
    print("%-36s %7.3f %7.3f %7.3f %7.3f %7.3f %6.2f %6.2f %7.3f" % r)
print("\nPromising = large gap, ofrac in ~[0.3,0.7], low agree, large confnd.")
print("(idea08 is a deliberate CONTRAST: confounding cancels in the contrast, so it should screen as")
print(" low-gap / high-agree even though confnd may look large. Use it as the negative control.)")
