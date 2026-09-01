import sys, os, json, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, 'results'); sys.path.insert(0, HERE)
from run_staph_rigorous import load_csv, EXPS, COVER, MAX_LINK, DEVICE, train_onestep, step_pred, _cols
from run_synthetic_rd import RD_GKAN
from run_new_datasets import build_knn_graph
import torch.nn.functional as F

def track_with_mask(df):
    ch=df.chamber.value_counts().idxmax(); df=df[df.chamber==ch]
    frames=sorted(df.frame.unique()); T=len(frames); fidx={f:i for i,f in enumerate(frames)}
    det={f:df[df.frame==f][['centroid-0','centroid-1','intensity_mean']].values for f in frames}
    tracks=[]
    for f in frames:
        ti=fidx[f]; D=det[f]; used=set()
        for tr in tracks:
            if D.shape[0]==0: break
            px,py=tr['pos']; d=np.hypot(D[:,0]-px,D[:,1]-py)
            for j in np.argsort(d):
                if j in used: continue
                if d[j]<=MAX_LINK:
                    tr['cells'][ti]=(D[j,0],D[j,1],D[j,2]); tr['pos']=(D[j,0],D[j,1]); used.add(j)
                break
        for j in range(D.shape[0]):
            if j not in used: tracks.append({'pos':(D[j,0],D[j,1]),'cells':{ti:(D[j,0],D[j,1],D[j,2])}})
    keep=[tr for tr in tracks if len(tr['cells'])>=COVER*T]
    if len(keep)<4: return None,None,None
    N=len(keep); traj=np.full((T,N),np.nan,np.float32); pos=np.zeros((N,2),np.float32); obs=np.zeros((T,N),bool)
    for n,tr in enumerate(keep):
        ts=sorted(tr['cells']); traj[ts,n]=[tr['cells'][t][2] for t in ts]; obs[ts,n]=True
        col=traj[:,n]; nz=np.isnan(col)
        if nz.any(): col[nz]=np.interp(np.where(nz)[0],np.where(~nz)[0],col[~nz]); traj[:,n]=col
        pos[n]=np.mean([tr['cells'][t][:2] for t in ts],axis=0)
    traj=(traj-np.nanmean(traj))/(np.nanstd(traj)+1e-9)
    return traj,pos,obs

def masked_score(pred, Yte, mask):           # preserve mean-of-per-transition-RMSE order
    vals=[]
    for k in range(pred.shape[1]):
        m=mask[k]
        if m.sum()>0: vals.append(float(torch.sqrt(((pred[:,k]-Yte[:,k])[m]**2).mean())))
    return float(np.mean(vals)) if vals else float('nan')

def eval_model(model, traj, L, tr_e, va_e, te_e, mask):
    train_onestep(model, traj, L, tr_e, va_e, te_e)         # trains + restores best ckpt
    X=torch.tensor(traj,dtype=torch.float32).to(DEVICE); Lt=L.to(DEVICE)
    Cte,Yte=_cols(X,te_e),_cols(X,[t+1 for t in te_e])
    with torch.no_grad(): pred=step_pred(model,Cte,Lt)
    allv=float(torch.sqrt(((pred-Yte)**2).mean(0)).mean())
    mt=torch.tensor(mask,dtype=torch.bool).to(DEVICE)
    return allv, masked_score(pred,Yte,mt)

out={}
for sid in EXPS:
    try: df=load_csv(sid)
    except Exception: continue
    traj,pos,obs=track_with_mask(df)
    if traj is None: continue
    T,N=traj.shape
    L_norm,W,deg=build_knn_graph(pos,k=min(6,N-1)); L=torch.tensor(L_norm,dtype=torch.float32)
    n_tr,n_va=int(0.6*T),int(0.7*T)
    tr_e=list(range(0,n_tr-1)); va_e=list(range(n_tr,n_va-1)); te_e=list(range(n_va,T-1))
    if len(tr_e)<5 or len(va_e)<2 or len(te_e)<3: continue
    mask=np.array([[obs[t,i] and obs[t+1,i] for i in range(N)] for t in te_e])   # (n_te, N)
    # persistence, both scorings
    pa=float(np.mean([np.sqrt(np.mean((traj[t+1]-traj[t])**2)) for t in te_e]))
    pm=[]
    for k,t in enumerate(te_e):
        m=mask[k]
        if m.sum()>0: pm.append(np.sqrt(np.mean((traj[t+1][m]-traj[t][m])**2)))
    pm=float(np.mean(pm)) if pm else float('nan')
    fa,fm,sa,sm=[],[],[],[]
    for s in range(3):
        torch.manual_seed(s); np.random.seed(s)
        a,b=eval_model(RD_GKAN(1,G=8,k=3,x_range=(-4,4)),traj,L,tr_e,va_e,te_e,mask); fa.append(a); fm.append(b)
        m2=RD_GKAN(1,G=8,k=3,x_range=(-4,4)); m2.D.data.zero_(); m2.D.requires_grad_(False)
        a,b=eval_model(m2,traj,L,tr_e,va_e,te_e,mask); sa.append(a); sm.append(b)
    out[sid]=dict(N=int(N),T=int(T),mask_frac=float(mask.mean()),
                  Full_all=float(np.mean(fa)),Full_masked=float(np.mean(fm)),
                  Self_all=float(np.mean(sa)),Self_masked=float(np.mean(sm)),
                  Pers_all=pa,Pers_masked=pm)
    print(f"{sid} maskfrac={mask.mean():.2f} | all F={np.mean(fa):.4f} S={np.mean(sa):.4f} P={pa:.4f}"
          f" | masked F={np.mean(fm):.4f} S={np.mean(sm):.4f} P={pm:.4f}", flush=True)
json.dump(out, open(os.path.join(RES, 'staph_masked_rescore.json'), 'w'), indent=1)
print("\nsaved")
