"""Verify I1 & I3 in BOTH regimes: do IPW-O-W AND DR-O-W beat ALL competitors at small Γ? N=500, 5 seeds, c_eps=1."""
import sys, importlib.util
from pathlib import Path
import numpy as np
ROOT = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
sys.path.insert(0, str(ROOT)); import common
def L(rel, fn):
    s = importlib.util.spec_from_file_location(fn, str(ROOT / rel)); m = importlib.util.module_from_spec(s); sys.modules[fn] = m; s.loader.exec_module(m); return getattr(m, fn)
SS = {}
for reg, suf, Cap in [("uncap","uncapped","Uncapped"),("cap","capped","Capped")]:
    SS[(reg,"IPW-X-X")] = L("methods/IPW-X-X/%s/ipw_x_x_%s.py"%(Cap,suf),"solve_ipw_x_x_%s"%suf)
    SS[(reg,"DR-X-X")] = L("methods/DoublyRobust-X-X/%s/doublyrobust_x_x_%s.py"%(Cap,suf),"solve_doublyrobust_x_x_%s"%suf)
    SS[(reg,"Direct-X-X")] = L("methods/Direct-X-X/%s/direct_x_x_%s.py"%(Cap,suf),"solve_direct_x_x_%s"%suf)
    SS[(reg,"IPW-O-X")] = L("methods/IPW-O-X/%s/ipw_o_x_%s.py"%(Cap,suf),"solve_ipw_o_x_%s"%suf)
    SS[(reg,"DR-O-X")] = L("methods/DoublyRobust-O-X/%s/doublyrobust_o_x_%s.py"%(Cap,suf),"solve_doublyrobust_o_x_%s"%suf)
    SS[(reg,"Hajek-O-X")] = L("methods/Hajek-O-X/%s/hajek_o_x_%s.py"%(Cap,suf),"solve_hajek_o_x_%s"%suf)
    SS[(reg,"IPW-O-W")] = L("methods/IPW-O-W/%s/ipw_o_w_%s.py"%(Cap,suf),"solve_ipw_o_w_%s"%suf)
    SS[(reg,"DR-O-W")] = L("methods/DoublyRobust-O-W/%s/doublyrobust_o_w_%s.py"%(Cap,suf),"solve_doublyrobust_o_w_%s"%suf)
def sig(z): return 1.0/(1.0+np.exp(-np.clip(np.asarray(z,float),-40,40)))
LV=np.round(np.linspace(-1,1,7),6); K=2; N=500; SEEDS=list(range(5)); GAM=[1.5,2.0,3.0,4.0]; CE=1.0
CFG={"I1":dict(alpha=8.,cs=1.,cx=2.,d0=4.,d1=5.,b=1.,noise=.6),
     "I3":dict(alpha=10.,cs=.8,cx=2.,d0=5.,d1=6.,b=1.,noise=.6)}
def tg(res):
    s=res.support_X.ravel(); lv=np.array(sorted(set(np.round(s,6))))
    pol=np.array([float(res.pi[1,np.where(np.round(s,6)==round(float(c),6))[0][0]]) for c in lv])
    idx={round(float(c),6):i for i,c in enumerate(lv)}
    return np.array([pol[idx[round(float(v),6)]] for v in LV])
for cfg,p in CFG.items():
    ES=2*sig(p["alpha"]*LV)-1.0; eY0=p["d0"]*ES; eY1=p["d1"]*ES+p["b"]*LV
    def val(pg): pg=np.asarray(pg,float); return float(np.mean(pg*eY1+(1-pg)*eY0))
    orc=val((LV>0).astype(float))
    def gen(n,sd):
        rng=np.random.default_rng(sd); X=rng.choice(LV,size=n)
        S=np.where(rng.uniform(size=n)<sig(p["alpha"]*X),1.,-1.)
        e=np.clip(sig(p["cs"]*S-p["cx"]*X),.02,.98); T=(rng.uniform(size=n)<e).astype(int)
        Y=np.where(T==1,p["d1"]*S+p["b"]*X,p["d0"]*S)+rng.normal(0,p["noise"],size=n)
        return dict(X=X.reshape(-1,1),T=T,Y=Y)
    print("\n##### %s  oracle=%.3f #####"%(cfg,orc),flush=True)
    for reg in ["uncap","cap"]:
        kw={} if reg=="uncap" else {"cap":(1.,.5)}
        D=[]
        for sd in SEEDS:
            o=gen(N,sd); w,_=common.ipw_weights_from_data(o["X"],o["T"],K)
            wraw,_=common.ipw_weights_from_data(o["X"],o["T"],K,normalize=False)
            mu=common.outcome_means(o["X"],o["T"],o["Y"],n_arms=K,cross_fit=True)
            Dm=common.pairwise_distance_matrix(o["X"]); D.append((o,w,wraw,mu,Dm))
        M={}
        def avg(nm,g=None,ow=False,dr=False,h=False):
            vs=[]
            for o,w,wraw,mu,Dm in D:
                f=SS[(reg,nm)]
                if ow:
                    eps=tuple(common.tight_epsilon(Dm,o["T"],w,K,is_distance=True,c_eps=CE))
                    r=f(o["X"],o["T"],o["Y"],w,mu,n_arms=K,Gamma=g,discretize=False,zscore=False,epsilon=eps,**kw) if dr else f(o["X"],o["T"],o["Y"],w,n_arms=K,Gamma=g,discretize=False,zscore=False,epsilon=eps,**kw)
                elif g is not None:
                    if h: r=f(o["X"],o["T"],o["Y"],wraw,n_arms=K,Gamma=g,maximize=True,discretize=False,**kw)
                    elif dr: r=f(o["X"],o["T"],o["Y"],w,mu,n_arms=K,Gamma=g,discretize=False,**kw)
                    else: r=f(o["X"],o["T"],o["Y"],w,n_arms=K,Gamma=g,discretize=False,**kw)
                else:
                    if nm=="Direct-X-X": r=f(o["X"],o["T"],o["Y"],n_arms=K,discretize=False,**kw)
                    elif dr: r=f(o["X"],o["T"],o["Y"],w,mu,n_arms=K,discretize=False,**kw)
                    else: r=f(o["X"],o["T"],o["Y"],w,n_arms=K,discretize=False,**kw)
                vs.append(val(tg(r)))
            return float(np.mean(vs))
        M["IPW-X-X"]=avg("IPW-X-X"); M["DR-X-X"]=avg("DR-X-X",dr=True); M["Direct-X-X"]=avg("Direct-X-X")
        for nm,dr,h in [("IPW-O-X",False,False),("DR-O-X",True,False),("Hajek-O-X",False,True)]:
            M[nm]=max(avg(nm,g=g,dr=dr,h=h) for g in GAM)
        compmax=max(M.values())
        owv={g:(avg("IPW-O-W",g=g,ow=True,dr=False), avg("DR-O-W",g=g,ow=True,dr=True)) for g in GAM}
        winG=next((g for g in GAM if min(owv[g])>compmax+0.003),None)
        print(" [%s] comp:%s max=%.3f | IPW-OW=%s DR-OW=%s | OW-beats-all@Γ=%s"%(reg,{k:round(v,3) for k,v in M.items()},compmax,[round(owv[g][0],3) for g in GAM],[round(owv[g][1],3) for g in GAM],winG),flush=True)
print("\ndone",flush=True)
