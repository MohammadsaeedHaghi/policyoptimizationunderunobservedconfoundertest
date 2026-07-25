"""End-to-end smoke check (maximize=False emphasis) over EVERY algorithm + extensions.

NOT a real experiment: one tiny shared fixture (_smoke/fixture.npz), one Γ, one seed. Goal = "does each
algorithm run end-to-end and produce sane output", with the new maximize=False code path exercised wherever
it exists, and the whole pipeline run in the loss convention.

Per method we check: runs (no exception) · policy is a valid simplex (cols sum 1, in [0,1]) · objective
finite · capacity respected (capped) · Shapley+KNN extension to test X is a valid simplex of right shape.
For the maximize-capable methods (Kallus, Regret-O, Hajek-OW) we ALSO assert the convention identity
   solver(Y, maximize=False)  ==  solver(-Y, maximize=True)        (reward = -loss; should be bit-identical)
For the value-maximisers (no maximize flag) we additionally run them on -Y to confirm the loss direction runs.
Each method is isolated in try/except so one failure can't mask the others.
"""
from __future__ import annotations
import importlib.util, json, sys, traceback
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

TOL = 1e-6

def load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, str(_ROOT / relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# ---- shared fixture ----
F = np.load(_ROOT / "_smoke" / "fixture.npz")
Xtr, Ttr, Ytr = F["Xtr"], F["Ttr"], F["Ytr"]
Ypot_tr, mu_tr = F["Ypot_tr"], F["mu_tr"]
Xte = F["Xte"]; Ypot_te = F["Ypot_te"]
w = F["w"]; muhat = F["muhat"]; cap = tuple(float(c) for c in F["cap"])
K = int(F["n_arms"]); G = float(F["Gamma"])
MESH = 5

sh = load("ext_shapley", "extensions/Shapley/shapley.py")
kn = load("ext_knn", "extensions/KNN/knn.py")

def simplex_ok(pi, n):
    pi = np.asarray(pi)
    return bool(pi.shape == (K, n) and np.all(pi >= -1e-7) and np.all(pi <= 1 + 1e-7)
                and np.allclose(pi.sum(axis=0), 1.0, atol=TOL))

def usage_ok(usage, capped):
    if not capped:
        return True
    return bool(np.all(np.asarray(usage) <= np.array(cap) + 1e-6))

def ext_ok(support_X, pi):
    pi_sh = sh.extend_with_shapley_multiarm(Xte, support_X, pi)
    pi_kn = kn.extend_with_knn_multiarm(Xte, support_X, pi)
    return simplex_ok(pi_sh, len(Xte)) and simplex_ok(pi_kn, len(Xte))

results = {}

def record(method, checks, objs, notes=""):
    ok = all(c["passed"] for c in checks)
    results[method] = dict(method=method, overall=("PASS" if ok else "FAIL"),
                           checks=checks, objectives=objs, notes=notes)
    flag = "PASS" if ok else "FAIL"
    print(f"\n[{flag}] {method}   {notes}")
    for c in checks:
        print(f"    {'ok ' if c['passed'] else 'XX '}{c['name']:34s} {c.get('detail','')}")

# ---------------- per-method call adapters ----------------
def call_robust_w(fn, *, capped, Y, maximize):
    kw = dict(n_arms=K, Gamma=G, discretize=True, mesh=MESH)
    if capped: kw["cap"] = cap
    return fn(Xtr, Ttr, Y, w, **kw)            # R-OW (no maximize arg)
def call_robust(fn, *, capped, Y, maximize):
    return call_robust_w(fn, capped=capped, Y=Y, maximize=maximize)  # R-O, same shape
def call_regret(fn, *, capped, Y, maximize):
    kw = dict(n_arms=K, Gamma=G, maximize=maximize, discretize=True, mesh=MESH)
    if capped: kw["cap"] = cap
    return fn(Xtr, Ttr, Y, w, **kw)
def call_ipw(fn, *, capped, Y, maximize):
    kw = dict(n_arms=K, discretize=True, mesh=MESH)
    if capped: kw["cap"] = cap
    return fn(Xtr, Ttr, Y, w, **kw)
def call_po(fn, *, capped, Y, maximize):
    kw = dict(n_arms=K, discretize=True, mesh=MESH)
    if capped: kw["cap"] = cap
    return fn(Xtr, Ttr, Y, **kw)               # no weights
def call_dr(fn, *, capped, Y, maximize):
    kw = dict(n_arms=K, Gamma=G, discretize=True, mesh=MESH)
    if capped: kw["cap"] = cap
    mh = muhat if maximize else -muhat         # loss direction also negates the outcome model
    return fn(Xtr, Ttr, Y, w, mh, **kw)

# Each method's capped/uncapped live in different modules with different fn names -> drive explicitly:
def driver(method, cap_mod, cap_fn, unc_mod, unc_fn, call, has_max, val=False, dr=False):
    try:
        mc = load(cap_mod[0], cap_mod[1]); mu_ = load(unc_mod[0], unc_mod[1])
        fcap = getattr(mc, cap_fn); func = getattr(mu_, unc_fn)
    except Exception as e:
        record(method, [dict(name="import", passed=False, detail=repr(e))], {}); return
    checks, objs = [], {}
    try:
        rc = call(fcap, capped=True, Y=Ytr, maximize=False)
        checks += [dict(name="capped runs", passed=True),
                   dict(name="capped policy is valid simplex", passed=simplex_ok(rc.pi, rc.pi.shape[1])),
                   dict(name="capped objective finite", passed=bool(np.isfinite(rc.objective_value)),
                        detail=f"obj={rc.objective_value:.5f}"),
                   dict(name="capped usage <= cap", passed=usage_ok(rc.usage, True),
                        detail=f"usage={np.round(rc.usage,3).tolist()} cap={cap}"),
                   dict(name="capped Shapley+KNN extension valid", passed=ext_ok(rc.support_X, rc.pi))]
        objs["capped_obj"] = float(rc.objective_value)
        ru = call(func, capped=False, Y=Ytr, maximize=False)
        checks += [dict(name="uncapped runs", passed=True),
                   dict(name="uncapped policy is valid simplex", passed=simplex_ok(ru.pi, ru.pi.shape[1])),
                   dict(name="uncapped objective finite", passed=bool(np.isfinite(ru.objective_value)),
                        detail=f"obj={ru.objective_value:.5f}")]
        objs["uncapped_obj"] = float(ru.objective_value)
        if has_max:
            rn = call(fcap, capped=True, Y=-Ytr, maximize=True)
            dpi = float(np.max(np.abs(rc.pi - rn.pi))); dobj = float(abs(rc.objective_value - rn.objective_value))
            checks.append(dict(name="maximize=False(Y) == maximize=True(-Y)",
                               passed=(dpi < 1e-6 and dobj < 1e-6), detail=f"dpi={dpi:.1e} dobj={dobj:.1e}"))
        elif val:
            rn = call(fcap, capped=True, Y=-Ytr, maximize=False)
            checks.append(dict(name="loss-direction (-Y) runs + valid simplex",
                               passed=simplex_ok(rn.pi, rn.pi.shape[1]), detail="(no maximize flag)"))
    except Exception as e:
        checks.append(dict(name="exception", passed=False, detail=repr(e))); traceback.print_exc()
    record(method, checks, objs, notes=("maximize-capable" if has_max else "value-maximiser (no flag)"))

driver("R-OW", ("rowc","methods/IPW-O-W/Capped/ipw_o_w_capped.py"), "solve_ipw_o_w_capped",
       ("rowu","methods/IPW-O-W/Uncapped/ipw_o_w_uncapped.py"), "solve_ipw_o_w_uncapped", call_robust_w, False, val=True)
driver("R-O", ("roc","methods/IPW-O-X/Capped/ipw_o_x_capped.py"), "solve_ipw_o_x_capped",
       ("rou","methods/IPW-O-X/Uncapped/ipw_o_x_uncapped.py"), "solve_ipw_o_x_uncapped", call_robust, False, val=True)
driver("Hajek-OW", ("rowc2","methods/Hajek-O-W/Capped/hajek_o_w_capped.py"), "solve_hajek_o_w_capped",
       ("rowu2","methods/Hajek-O-W/Uncapped/hajek_o_w_uncapped.py"), "solve_hajek_o_w_uncapped", call_regret, True)
driver("Regret-O", ("roc2","methods/Hajek-O-X/Capped/hajek_o_x_capped.py"), "solve_hajek_o_x_capped",
       ("rou2","methods/Hajek-O-X/Uncapped/hajek_o_x_uncapped.py"), "solve_hajek_o_x_uncapped", call_regret, True)
driver("IPW", ("ipwc","methods/IPW-X-X/Capped/ipw_x_x_capped.py"), "solve_ipw_x_x_capped",
       ("ipwu","methods/IPW-X-X/Uncapped/ipw_x_x_uncapped.py"), "solve_ipw_x_x_uncapped", call_ipw, False, val=True)
driver("Direct-X-X", ("poc","methods/Direct-X-X/Capped/direct_x_x_capped.py"),
       "solve_direct_x_x_capped",
       ("pou","methods/Direct-X-X/Uncapped/direct_x_x_uncapped.py"),
       "solve_direct_x_x_uncapped", call_po, False, val=True)
driver("DoublyRobust", ("drc","methods/DoublyRobust-X-X/Capped/doublyrobust_x_x_capped.py"), "solve_doublyrobust_x_x_capped",
       ("dru","methods/DoublyRobust-X-X/Uncapped/doublyrobust_x_x_uncapped.py"), "solve_doublyrobust_x_x_uncapped",
       call_dr, False, val=True)
driver("R-OW-DoublyRobust", ("rowdrc","methods/DoublyRobust-O-W/Capped/doublyrobust_o_w_capped.py"), "solve_doublyrobust_o_w_capped",
       ("rowdru","methods/DoublyRobust-O-W/Uncapped/doublyrobust_o_w_uncapped.py"), "solve_doublyrobust_o_w_uncapped",
       call_dr, False, val=True)
driver("R-O-DoublyRobust", ("rodrc","methods/DoublyRobust-O-X/Capped/doublyrobust_o_x_capped.py"), "solve_doublyrobust_o_x_capped",
       ("rodru","methods/DoublyRobust-O-X/Uncapped/doublyrobust_o_x_uncapped.py"), "solve_doublyrobust_o_x_uncapped",
       call_dr, False, val=True)

# ---- Oracle (benchmark: takes a value matrix; no maximize flag) ----
try:
    oc = load("orc", "methods/Oracle/Capped/oracle_capped.py")
    ou = load("oru", "methods/Oracle/Uncapped/oracle_uncapped.py")
    checks, objs = [], {}
    rc = oc.solve_oracle_capped(Xtr, Ypot_tr, n_arms=K, cap=cap, discretize=True, mesh=MESH)
    ru = ou.solve_oracle_uncapped(Xtr, Ypot_tr, n_arms=K, discretize=True, mesh=MESH)
    rn = oc.solve_oracle_capped(Xtr, -Ypot_tr, n_arms=K, cap=cap, discretize=True, mesh=MESH)  # loss direction
    checks += [dict(name="capped runs (Full-info on Ypot)", passed=True),
               dict(name="capped policy is valid simplex", passed=simplex_ok(rc.pi, rc.pi.shape[1])),
               dict(name="capped usage <= cap", passed=usage_ok(rc.usage, True),
                    detail=f"usage={np.round(rc.usage,3).tolist()} cap={cap}"),
               dict(name="capped objective finite", passed=bool(np.isfinite(rc.objective_value)),
                    detail=f"obj={rc.objective_value:.5f}"),
               dict(name="capped Shapley+KNN extension valid", passed=ext_ok(rc.support_X, rc.pi)),
               dict(name="uncapped runs + valid simplex", passed=simplex_ok(ru.pi, ru.pi.shape[1])),
               dict(name="loss-direction (-Ypot) runs + valid simplex",
                    passed=simplex_ok(rn.pi, rn.pi.shape[1]), detail="(benchmark; no maximize flag)")]
    objs = dict(capped_obj=float(rc.objective_value), uncapped_obj=float(ru.objective_value))
    record("Oracle", checks, objs, notes="benchmark (value matrix; no flag)")
except Exception as e:
    record("Oracle", [dict(name="exception", passed=False, detail=repr(e))], {}); traceback.print_exc()

# ---- Kallus (parametric; maximize=False + identity; both odds & Wasserstein sets) ----
try:
    kal = load("kallus", "methods/Kallus/kallus.py")
    checks, objs = [], {}
    rk = kal.fit_kallus(Xtr, Ttr, Ytr, w, n_arms=K, Gamma=G, maximize=False, n_iters=8, n_restarts=2)
    pol = kal.predict_kallus(rk.theta, Xte)        # (n_te, K) per-unit softmax policy
    rows_ok = bool(pol.shape == (len(Xte), K) and np.allclose(pol.sum(axis=1), 1.0, atol=TOL)
                   and np.all(pol >= -1e-9))
    rkn = kal.fit_kallus(Xtr, Ttr, -Ytr, w, n_arms=K, Gamma=G, maximize=True, n_iters=8, n_restarts=2)
    dth = float(np.max(np.abs(rk.theta - rkn.theta))); dob = float(abs(rk.objective_value - rkn.objective_value))
    # Wasserstein-set variant smoke
    rkw = kal.fit_kallus(Xtr, Ttr, Ytr, w, n_arms=K, Gamma=G, maximize=False, wasserstein=True,
                         epsilon=[0.5, 0.5], n_iters=6, n_restarts=1)
    checks += [dict(name="odds-set fit runs (maximize=False)", passed=True,
                    detail=f"regret={rk.objective_value:.5f}"),
               dict(name="objective finite", passed=bool(np.isfinite(rk.objective_value))),
               dict(name="predict on test = valid per-unit softmax", passed=rows_ok),
               dict(name="maximize=False(Y) == maximize=True(-Y)", passed=(dth < 1e-6 and dob < 1e-6),
                    detail=f"dtheta={dth:.1e} dobj={dob:.1e}"),
               dict(name="Wasserstein-set fit runs", passed=bool(np.isfinite(rkw.objective_value)),
                    detail=f"regret={rkw.objective_value:.5f}")]
    objs = dict(odds_regret=float(rk.objective_value), wass_regret=float(rkw.objective_value))
    record("Kallus", checks, objs, notes="maximize-capable (parametric softmax)")
except Exception as e:
    record("Kallus", [dict(name="exception", passed=False, detail=repr(e))], {}); traceback.print_exc()

# ---- summary ----
print("\n" + "=" * 64)
n_pass = sum(1 for r in results.values() if r["overall"] == "PASS")
print(f"SUMMARY: {n_pass}/{len(results)} algorithms PASS")
for m, r in results.items():
    print(f"   {r['overall']:5s} {m}")
out = _ROOT / "_smoke" / "smoke_results.json"
out.write_text(json.dumps(results, indent=2))
print(f"\nresults -> {out}")
print("ALL_PASS" if n_pass == len(results) else "SOME_FAIL")
