"""Option 3: LINEAR (halfspace) policy class -- pi(x) = 1{beta'x + b0 >= 0} -- via MILP.

This is the class Kallus-Zhou optimise over (their CRLogit; Prop. 7 gives the conic/integer route
for non-simplex classes), so adding it makes that comparison like-for-like for the first time.
It also scales with dimension the way a partition cannot: d+1 parameters regardless of d, which
is the opposite failure mode to both our Lipschitz class (dies when distances concentrate) and
sharp's cells (die when b^d cells outrun the sample).

Formulation: binaries z_i with big-M linking, and pi[1,i] == z_i. The uncertainty set, the
objective and every dual are untouched -- only the policy side becomes integral, so the model
goes from LP to MILP. beta is box-normalised (|beta_j| <= 1) to fix the scale degeneracy.

linear_policy=False (default) reproduces the current solver exactly.
"""
from pathlib import Path
ROOT = Path("/home1/haghim/code 1.1")
TARGETS = ["methods/IPW-O-W/Uncapped/ipw_o_w_uncapped.py",
           "methods/IPW-O-X/Uncapped/ipw_o_x_uncapped.py"]
SIG_OLD = "    lipschitz=None,\n"
SIG_NEW = "    lipschitz=None,\n    linear_policy: bool = False,\n    linear_M: float = 20.0,\n"
BLOCK = '''
    # ---- LINEAR (halfspace) POLICY CLASS: pi(x) = 1{beta'x + b0 >= 0} -- see patch_linear_policy.py
    if linear_policy:
        _Xl = np.asarray(support_X, float)
        if _Xl.ndim == 1:
            _Xl = _Xl.reshape(-1, 1)
        _dl = _Xl.shape[1]
        _beta = m.addVars(_dl, lb=-1.0, ub=1.0, name="lbeta")
        _b0 = m.addVar(lb=-1.0, ub=1.0, name="lbeta0")
        _z = m.addVars(n, vtype=GRB.BINARY, name="lz")
        _Ml = float(linear_M)
        for _i in range(n):
            _lin = gp.quicksum(_beta[_j] * float(_Xl[_i, _j]) for _j in range(_dl)) + _b0
            m.addConstr(_lin >= 1e-4 - _Ml * (1 - _z[_i]), name="lin_hi_%d" % _i)
            m.addConstr(_lin <= -1e-4 + _Ml * _z[_i], name="lin_lo_%d" % _i)
            m.addConstr(pi[1, _i] == _z[_i], name="lin_pi_%d" % _i)
'''
ANCHOR = "    for i in range(n):                                        # each unit's policy is a distribution\n"
ANCHOR2 = "    for i in range(n):\n        m.addConstr(gp.quicksum(pi[k, i] for k in range(K)) == 1.0, name=f\"simplex_{i}\")\n"

def main():
    plans = []
    for rel in TARGETS:
        p = ROOT / rel; s = p.read_text()
        if "linear_policy" in s:
            print("SKIP (already patched):", rel); continue
        assert s.count(SIG_OLD) == 1, "%s sig x%d" % (rel, s.count(SIG_OLD))
        a = ANCHOR if s.count(ANCHOR) == 1 else ANCHOR2
        assert s.count(a) == 1, "%s anchor x%d" % (rel, s.count(a))
        plans.append((p, s.replace(SIG_OLD, SIG_NEW).replace(a, BLOCK + a)))
    for p, s in plans:
        p.write_text(s); print("patched", p.relative_to(ROOT))
    print("OK %d files" % len(plans))

if __name__ == "__main__":
    main()
