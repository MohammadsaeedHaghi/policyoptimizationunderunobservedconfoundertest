#!/usr/bin/env python3
"""Add the C4 "historical rationality" constraint to the four uncapped O-X / O-W solvers.

The constraint (condition C4 of 42350_MainPaper.pdf) restricts the ADVERSARY on the untreated
arm:

    sum_{i in N0} Y_i w_i  >=  sum_{i in N0} Y_i w_hat_i                                   (C4)

Reading: the historical decision maker, who saw more than the recorded covariates, used that
extra information to do at least as well as the observables alone would explain -- so the true
weights cannot make the untreated look worse than the nominal weights do.

Our solvers are DUALS of the inner minimisation, so one primal inequality becomes one new dual
variable kappa >= 0. Standard LP duality: a primal constraint g'w >= R (in a MINIMISATION)
contributes +R*kappa to the dual objective and -kappa*g_i to unit i's reduced cost. Here
g_i = Y_i for i in N0 and 0 otherwise. This is the same kappa that appears in the paper's own
Proposition 2 conic program.

  O-X families  the dual feasibility is an EQUALITY (stationarity):
                    mu_i - nu_i == c_i - beta_{T_i}          ->   ... - kappa*Y_i   for T_i = 0
  O-W families  it is an INEQUALITY (reduced cost >= 0):
                    c_i + (1/n) theta - mu_i + nu_i >= 0     ->   ... - kappa*Y_i   for arm 0

Note that C4 constrains the WEIGHTS, not the objective, so the raw outcome Y is used in all four
-- including the doubly-robust pair, whose objective multiplies w by the residual rather than by
Y. Also note C4 is invariant to centring the outcome: under the per-arm Hajek calibration
sum_{i in arm} w_i = n = sum_{i in arm} w_hat_i, a constant shift of Y cancels on both sides.

Default is rationality=False, so nothing already computed changes. Idempotent.
"""
import pathlib

ROOT = pathlib.Path("/home1/haghim/code 1.1")
BAK = pathlib.Path(__file__).resolve().parent / "solver_backups"
BAK.mkdir(exist_ok=True)

SIG_OLD = "    lipschitz=None,\n"
SIG_NEW = ("    rationality: bool = False,      # C4: sum_{N0} Y_i w_i >= sum_{N0} Y_i w_hat_i\n"
           "    lipschitz=None,\n")

KAPPA_BLOCK = '''
    # --- C4 "historical rationality" (see semisynthetic/patch_rationality.py) ---
    # One primal inequality on the adversary's weights over the UNTREATED arm becomes one dual
    # variable kappa >= 0: + R*kappa in this MAXIMISE objective, and -kappa*Y_i in the dual
    # feasibility row of every untreated unit. R is the nominal value of the same functional, so
    # w = w_hat is always feasible and the set is never empty.
    _kappa = None
    if rationality:
        _N0 = [int(i) for i in range(n) if int(T[i]) == 0]
        _R = float(sum(float(Y[i]) * float(w_hat[i]) for i in _N0))
        _kappa = m.addVar(lb=0.0, name="kappa_rationality")
        obj += _R * _kappa
'''

TARGETS = {
    "methods/IPW-O-X/Uncapped/ipw_o_x_uncapped.py": [
        ("        m.addConstr(mu[i] - nu[i] == (1.0 / n) * float(Y[i]) * pi[t, i] - beta[t], name=f\"stat_{i}\")",
         "        _rc = (_kappa * float(Y[i])) if (rationality and t == 0) else 0.0\n"
         "        m.addConstr(mu[i] - nu[i] == (1.0 / n) * float(Y[i]) * pi[t, i] - beta[t] - _rc,\n"
         "                    name=f\"stat_{i}\")"),
    ],
    "methods/DoublyRobust-O-X/Uncapped/doublyrobust_o_x_uncapped.py": [
        ("        m.addConstr(mu[i] - nu[i] == (1.0 / n) * float(resid[i]) * pi[t, i] - beta[t], name=f\"stat_{i}\")",
         "        _rc = (_kappa * float(Y[i])) if (rationality and t == 0) else 0.0\n"
         "        m.addConstr(mu[i] - nu[i] == (1.0 / n) * float(resid[i]) * pi[t, i] - beta[t] - _rc,\n"
         "                    name=f\"stat_{i}\")"),
    ],
    "methods/IPW-O-W/Uncapped/ipw_o_w_uncapped.py": [
        ("            m.addConstr((1.0 / n) * pi[k, i] * float(Y[i]) + (1.0 / n) * theta[k, i]\n"
         "                        - mu[i] + nu[i] - (_d if _d is not None else 0.0) >= 0.0,\n"
         "                        name=f\"feas_{k}_{i}\")",
         "            _rc = (_kappa * float(Y[i])) if (rationality and k == 0) else 0.0\n"
         "            m.addConstr((1.0 / n) * pi[k, i] * float(Y[i]) + (1.0 / n) * theta[k, i]\n"
         "                        - mu[i] + nu[i] - (_d if _d is not None else 0.0) - _rc >= 0.0,\n"
         "                        name=f\"feas_{k}_{i}\")"),
    ],
    "methods/DoublyRobust-O-W/Uncapped/doublyrobust_o_w_uncapped.py": [
        ("            m.addConstr((1.0 / n) * pi[k, i] * float(resid[i]) + (1.0 / n) * theta[k, i]\n"
         "                        - mu[i] + nu[i] >= 0.0, name=f\"feas_{k}_{i}\")",
         "            _rc = (_kappa * float(Y[i])) if (rationality and k == 0) else 0.0\n"
         "            m.addConstr((1.0 / n) * pi[k, i] * float(resid[i]) + (1.0 / n) * theta[k, i]\n"
         "                        - mu[i] + nu[i] - _rc >= 0.0, name=f\"feas_{k}_{i}\")"),
    ],
}

OBJ_LINE = "    m.setObjective(obj, GRB.MAXIMIZE)"

for rel, edits in TARGETS.items():
    p = ROOT / rel
    s = p.read_text()
    if "rationality" in s:
        print("  already patched: %s" % rel)
        continue
    (BAK / p.name).write_text(s)                       # backup before touching anything
    assert SIG_OLD in s, rel
    s = s.replace(SIG_OLD, SIG_NEW, 1)
    assert OBJ_LINE in s, rel
    s = s.replace(OBJ_LINE, KAPPA_BLOCK + OBJ_LINE, 1)
    for old, new in edits:
        assert old in s, "anchor missing in %s:\n%s" % (rel, old[:90])
        s = s.replace(old, new, 1)
    p.write_text(s)
    print("  patched: %s" % rel)

print("\nbackups in %s" % BAK)
