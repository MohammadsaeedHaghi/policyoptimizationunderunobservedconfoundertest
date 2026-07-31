"""gstar continuous at coupling ALPHA = 4 (the peak of the coupling ablation).

The main arm uses ALPHA = 10, a very strong X-S coupling. The discrete ablation shows ALPHA = 4
is the better showcase on both measures: largest O-W margin over the naive (+0.396 vs +0.152 at
ALPHA = 10) and full transport-term payoff (O-W - O-X = +0.310). It is also the more defensible
setting, being a moderate rather than extreme coupling. Mutation wrapper: ALPHA is read at call
time inside p_s1/generate, so setting the module attribute before re-export is sufficient.
"""
import importlib.util, sys
from pathlib import Path
_sp = importlib.util.spec_from_file_location("gstar_cont_a4", Path(__file__).resolve().parent / "dgp_cont.py")
_m = importlib.util.module_from_spec(_sp); sys.modules["gstar_cont_a4"] = _m; _sp.loader.exec_module(_m)
_m.ALPHA = 4.0
for _k in dir(_m):
    if not _k.startswith("__"):
        globals()[_k] = getattr(_m, _k)
