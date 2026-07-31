"""exp_kz18 ablation wrapper: same KZ18 DGP, scalar covariate = the CATE index
x = beta_xT'X5 / 8 instead of the nominal-propensity index. This is the DO-NO-HARM arm:
the naive policy is already within 0.021 of the oracle here, so the question is whether
robustness costs anything when there is no harm to undo. Mutation pattern (module attr set
before re-export) so every function sees INDEX = "cate" at call time.
"""
import importlib.util, sys
from pathlib import Path

_sp = importlib.util.spec_from_file_location("kz18_core_cidx", Path(__file__).resolve().parent / "dgp.py")
_m = importlib.util.module_from_spec(_sp)
sys.modules["kz18_core_cidx"] = _m
_sp.loader.exec_module(_m)
_m.INDEX = "cate"

for _k in dir(_m):
    if not _k.startswith("__"):
        globals()[_k] = getattr(_m, _k)
