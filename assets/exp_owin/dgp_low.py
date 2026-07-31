"""exp_owin variant: same structure, SMALLER confounder level effect (D0, D1 = 2.5, 3.5).

v2 has CATE sd 1.65 against outcome sd 9.18 -- signal/noise 0.18, worse than KZ18's 0.64 where
every robust method sank below never-treat. The culprit is D0 = 8: the confounder shifts outcome
LEVELS by +-8, which is what both the noise and the naive bias are made of. Cutting it to 2.5
cuts the bias (~2.4 -> ~0.75, still a substantial fraction of the CATE's 1.65 sd, so robustness
is still needed) while cutting the outcome sd ~4x, so the estimation problem becomes tractable.
Which of the two matters more is exactly what the pilot measures.
"""
import importlib.util, sys
from pathlib import Path
_sp = importlib.util.spec_from_file_location("owin_core_low", Path(__file__).resolve().parent / "dgp.py")
_m = importlib.util.module_from_spec(_sp); sys.modules["owin_core_low"] = _m; _sp.loader.exec_module(_m)
_m.D0, _m.D1 = 2.5, 3.5
for _k in dir(_m):
    if not _k.startswith("__"):
        globals()[_k] = getattr(_m, _k)
