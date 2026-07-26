"""exp_owgap_alpha: the exp_owgap DGP with the S|X coupling strength ALPHA = 2.0.

Same DGP as assets/exp_owgap/dgp.py in every respect EXCEPT P(S=+1|X) = sigma(ALPHA*X) with
ALPHA = 2.0 (base owgap: 10.0). The sweep ALPHA in {1,2,4,6,10} varies how predictable the
hidden confounder S is from the observed X -- the mechanism the Wasserstein balance term
exploits -- while leaving the selection strength on S (CS=0.8) and the outcome model
(mu0=8S, mu1=9S+1.5X) untouched. CATE(X) = (2*sigma(ALPHA*X)-1) + 1.5X is odd and increasing
in X for every ALPHA, so the oracle rule stays "treat iff X>0"; only the oracle VALUE and
corr(X,S) change. Loaded by runners via --dgp; delegates everything to the base module.
"""
import importlib.util, sys
from pathlib import Path

_BASE = Path(__file__).resolve().parent.parent / "exp_owgap" / "dgp.py"
_spec = importlib.util.spec_from_file_location("owgap_base_a2", str(_BASE))
_m = importlib.util.module_from_spec(_spec); sys.modules[_spec.name] = _m; _spec.loader.exec_module(_m)
_m.ALPHA = 2.0

ALPHA, LEVELS, CAP, K, NOISE = _m.ALPHA, _m.LEVELS, _m.CAP, _m.K, _m.NOISE
p_s1, propensity, mu0, mu1 = _m.p_s1, _m.propensity, _m.mu0, _m.mu1
grid_truth, oracle_policy, generate, exact_value = _m.grid_truth, _m.oracle_policy, _m.generate, _m.exact_value
