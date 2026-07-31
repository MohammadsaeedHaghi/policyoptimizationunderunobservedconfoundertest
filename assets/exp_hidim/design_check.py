"""Design diagnostic: how far below the true oracle is the CELL-ORACLE CEILING?

The cell-oracle is the best policy that is constant within each of sharp's K cells (treat cell j
iff E[CATE | cell j] > 0). Sharp cannot beat it, no matter how well it estimates. If that ceiling
is close to the oracle, sharp is unbeatable by construction; if it is far below, there is room.
No LP solving here -- this is pure DGP geometry, so it is the cheap way to design the experiment.
"""
import sys, importlib.util
import numpy as np
from sklearn.cluster import KMeans
ROOT="/home1/haghim/code 1.1"; sys.path.insert(0,ROOT)

def load(dim):
    sp=importlib.util.spec_from_file_location("hd","%s/assets/exp_hidim/dgp.py"%ROOT)
    d=importlib.util.module_from_spec(sp); sys.modules["hd"]=d; sp.loader.exec_module(d)
    if dim!=d.D:                                  # rebuild the direction vectors at this dim
        e=np.ones(dim)/np.sqrt(dim)
        d.D=dim; d.A_CATE=2.0*e
        d.V_CONF=np.array([(-1.0)**j for j in range(dim)])/np.sqrt(dim)
        d.C_PROP=0.8*e; d.B0=0.5*np.array([(-1.0)**(j//2) for j in range(dim)])
    return d

NTE=60000
print("cell-oracle CEILING vs true oracle (higher gap = more room for a metric-based method)\n")
print("%-4s %-9s %8s %8s %8s %8s   %s" % ("d","cells","oracle","cellOrc","naive","never","gap(oracle-cellOrc)"))
for dim in (1,4,6,8):
    d=load(dim)
    obs,f=d.generate(NTE,0); X=obs["X"]; Y1,Y0=f["Y1"],f["Y0"]
    c=d.cate(X); orc=(c>0).astype(float)
    v_or=float(np.mean(orc*Y1+(1-orc)*Y0)); v_nev=float(np.mean(Y0))
    # infinite-data naive: biased contrast E[Y|T=1,x]-E[Y|T=0,x]
    ps=d.p_s1(X); e1=d.propensity(X,np.ones(len(X))); e0=d.propensity(X,-np.ones(len(X)))
    pt=ps*e1+(1-ps)*e0; pS_T1=ps*e1/pt
    hat=(X@d.A_CATE+d.THETA+d.DELTA*(2*pS_T1-1))
    pn=(hat>0).astype(float); v_nv=float(np.mean(pn*Y1+(1-pn)*Y0))
    for ncell in (15,30,60):
        sub=X[:8000]
        km=KMeans(n_clusters=ncell,n_init=4,random_state=0).fit(sub)
        lab=km.predict(X)
        pc=np.zeros(len(X))
        for j in range(ncell):
            m=lab==j
            if m.any(): pc[m]=1.0 if c[m].mean()>0 else 0.0
        v_cell=float(np.mean(pc*Y1+(1-pc)*Y0))
        print("%-4d %-9d %8.3f %8.3f %8.3f %8.3f   %+.3f" % (dim,ncell,v_or,v_cell,v_nv,v_nev,v_or-v_cell))
    print()
