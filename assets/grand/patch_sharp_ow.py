#!/usr/bin/env python3
"""Add SHARPNESS (per-cell, per-arm weight calibration) to the O-W solvers -> Sharp-O-W.

WHAT IS ADDED
-------------
The O-W uncertainty set is currently  box  ∩  Wasserstein ball, with NO calibration at all.
(O-X has the pooled calibration sum_{i:T_i=k} W_i = n; the sharp baseline has the per-cell one.
O-W has neither -- it gets its stabilisation from the Hajek centering of w_hat instead.)
Sharpness adds, for every (arm k, cell j):

        sum_{i in cell j, T_i = k}  W_i  =  sum_{i in cell j, T_i = k}  w_hat_i          (*)

i.e. the adversary may redistribute weight WITHIN an (arm, cell) group but cannot change the
group total. This is the Dorn-Guo sharpness condition; it is what makes the plug-in sharp
baseline tighter than the plain box, and it is exactly the ingredient O-W lacks.

DUAL DERIVATION (the part that must not be got wrong)
----------------------------------------------------
The solver is a DUAL formulation. Its existing per-unit dual-feasibility row is

        c_i + (1/n) theta_{k,i} - mu_i + nu_i >= 0,     c_i = (1/n) pi_k(X_i) Y_i

which is the "<= c_i" dual-feasibility form  mu_i - nu_i - (1/n) theta_{k,i} <= c_i.
A new PRIMAL EQUALITY (*) over the group S_{k,j}, with FREE dual delta_{k,j}, contributes the
coefficient of W_i in (*) -- which is 1 -- to that row, and delta * RHS to the dual objective:

        objective   +=  sum_{k,j}  delta_{k,j} * R_{k,j},      R_{k,j} = sum_{S_{k,j}} w_hat_i
        feasibility :   c_i + (1/n) theta_{k,i} - mu_i + nu_i - delta_{k,j(i)} >= 0

Cross-check against the O-X solver, which already has the POOLED version of exactly this
constraint (sum_{i:T_i=k} W_i = n): there the objective carries `+ n * beta_k` and the
stationarity row reads `mu_i - nu_i == c_i - beta_t`, i.e. `mu_i - nu_i + beta_t = c_i` --
the same +delta on the left, the same delta*RHS in the objective. The signs agree.

INVARIANTS (checked by validate_sharp_ow.py, not assumed)
  1. sharp_cells=None reproduces the current solver byte-for-byte.
  2. Tighter set => higher worst case: obj(Sharp-O-W) >= obj(O-W) at identical Gamma/eps/L.
  3. At Gamma = 1 the box is a single point, so (*) is automatically satisfied and
     Sharp-O-W == O-W exactly.
  4. The dual optimum equals a BRUTE-FORCE primal solve of the inner minimisation at the
     returned policy. This is the decisive test of the derivation above.
"""
import sys
from pathlib import Path

ROOT = Path("/home1/haghim/code 1.1")
TARGETS = ["methods/IPW-O-W/Uncapped/ipw_o_w_uncapped.py",
           "methods/IPW-O-W/Capped/ipw_o_w_capped.py"]

SIG_OLD = "    lipschitz=None,\n"
SIG_NEW = "    lipschitz=None,\n    sharp_cells=None,\n"

BUILD = '''
    # ---- SHARPNESS (Dorn-Guo): per-(arm, cell) weight-total equalities -> Sharp-O-W ------------
    # sharp_cells: length-n integer cell id per unit (None => plain O-W, unchanged).
    # For each (arm k, cell j) present in the data we require the adversary to preserve the
    # NOMINAL weight total of that group; it may only redistribute within it. See patch_sharp_ow.py
    # for the dual derivation. Groups with a single unit pin that unit's weight to w_hat exactly,
    # which is the degenerate limit the caller controls through the coarseness of the cells.
    _sharp_groups = {}
    if sharp_cells is not None:
        _sc = np.asarray(sharp_cells).astype(int).ravel()
        if _sc.shape[0] != n:
            raise ValueError("sharp_cells must have length n.")
        for _k in range(K):
            for _j in np.unique(_sc):
                _idx = [int(i) for i in np.where((_sc == _j) & (T == _k))[0]]
                if _idx:
                    _sharp_groups[(_k, int(_j))] = (_idx, float(np.sum(w_hat[_idx])))
'''

OBJ_ANCHOR = "    m.setObjective(obj, GRB.MAXIMIZE)\n"
OBJ_NEW = ('''    delta = {}
    for _key, (_idx, _R) in _sharp_groups.items():          # free dual per (arm, cell) equality
        delta[_key] = m.addVar(lb=-GRB.INFINITY, name="delta_%d_%d" % _key)
        obj += delta[_key] * _R                             # + delta * (nominal group total)
'''
           + OBJ_ANCHOR)

FEAS_OLD = """            m.addConstr((1.0 / n) * pi[k, i] * float(Y[i]) + (1.0 / n) * theta[k, i]
                        - mu[i] + nu[i] >= 0.0, name=f"feas_{k}_{i}")"""
FEAS_NEW = """            _d = delta.get((k, int(_cell_of[i]))) if _sharp_groups else None
            m.addConstr((1.0 / n) * pi[k, i] * float(Y[i]) + (1.0 / n) * theta[k, i]
                        - mu[i] + nu[i] - (_d if _d is not None else 0.0) >= 0.0,
                        name=f"feas_{k}_{i}")"""

CELLMAP = '''    _cell_of = (np.asarray(sharp_cells).astype(int).ravel()
                if sharp_cells is not None else np.zeros(n, dtype=int))
'''


def main():
    plans = []
    for rel in TARGETS:
        p = ROOT / rel
        s = p.read_text()
        if "sharp_cells" in s:
            print("SKIP (already patched):", rel); continue
        assert s.count(SIG_OLD) == 1, "%s: signature anchor x%d" % (rel, s.count(SIG_OLD))
        assert s.count(OBJ_ANCHOR) == 1, "%s: objective anchor x%d" % (rel, s.count(OBJ_ANCHOR))
        assert s.count(FEAS_OLD) == 1, "%s: feasibility anchor x%d" % (rel, s.count(FEAS_OLD))
        anchor_box = "    a_box, b_box = marginal_sensitivity_box(w_hat, Gamma)\n"
        assert s.count(anchor_box) == 1, "%s: box anchor x%d" % (rel, s.count(anchor_box))
        new = s.replace(SIG_OLD, SIG_NEW)
        new = new.replace(anchor_box, anchor_box + BUILD + CELLMAP)
        new = new.replace(OBJ_ANCHOR, OBJ_NEW)
        new = new.replace(FEAS_OLD, FEAS_NEW)
        plans.append((p, new))
    for p, new in plans:
        p.write_text(new); print("patched", p.relative_to(ROOT))
    print("OK %d file(s)" % len(plans))


if __name__ == "__main__":
    main()
