#!/usr/bin/env python3
"""
RIGOROUS S. aureus agr-QS temporal re-analysis from the ORIGINAL data
(data/StaphQS_BIAD1046/mm_savesV1.zip), using the authors' StarDist per-cell
segmentation CSVs (stardistdata/Pos01.csv) for exactly the 15 experiments used
in the paper (s06..s38).

Pipeline (all from the original segmentation; the only reconstruction step is
cell tracking, which for chamber-confined bacteria is standard nearest-centroid
linking):
  1. Read StarDist CSV (columns: frame, chamber, centroid-0/1, intensity_mean=
     agr fluorescence). Use the chamber with the most detections.
  2. Track cells across frames by greedy nearest-centroid linking
     (max link < MAX_LINK px); keep tracks present in >= COVER of frames.
  3. Build per-cell fluorescence trajectory (linear-interpolate gaps), normalize
     per experiment to zero mean / unit std; positions = mean centroid.
  4. k-NN graph (k=6) from positions.
  5. One-step prediction with a LEAKAGE-PROOF temporal split: train on the first
     60% of transitions, EARLY-STOP on the next 10% (validation), report one-step
     RMSE on the last 30% (test) ONCE.
  6. Models: Full RD-GKAN (eq.11), Self-only (D_theta=0), Shuffled-graph
     (permuted adjacency), Diffusion-only (psi=0), Persistence (c_hat(t+1)=c(t)),
     AR(1). Plus the Corollary stability margin for the learned Full model.
  7. Aggregate: Full-beats-self / Full-beats-shuffled win rates, two-sided
     Wilcoxon, persistence comparison, mean one-step RMSE, learned D_theta.

Author: Liang Dong
"""
import os, sys, json, zipfile, warnings, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
warnings.filterwarnings('ignore')
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, 'results'); sys.path.insert(0, HERE)
from run_synthetic_rd import RD_GKAN
from run_new_datasets import build_knn_graph
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ZIP = os.path.join(ROOT, 'data', 'StaphQS_BIAD1046', 'mm_savesV1.zip')
EXPS = ['s06','s07','s08','s10','s14','s15','s17','s24','s26','s27','s28','s34','s36','s37','s38']
COVER, MAX_LINK = 0.6, 12.0
N_SEEDS = 3


def load_csv(sid):
    with zipfile.ZipFile(ZIP) as z:
        names = [n for n in z.namelist() if ('/%s/' % sid) in n and 'stardistdata' in n and n.endswith('Pos01.csv')]
        if not names:
            names = [n for n in z.namelist() if ('/%s/' % sid) in n and 'stardistdata' in n and n.endswith('.csv')]
        with z.open(sorted(names)[0]) as f:
            return pd.read_csv(f)


def track(df):
    """Greedy nearest-centroid tracking on the busiest chamber -> (traj [T,N], pos [N,2])."""
    ch = df.chamber.value_counts().idxmax()
    df = df[df.chamber == ch]
    frames = sorted(df.frame.unique()); T = len(frames)
    fidx = {f: i for i, f in enumerate(frames)}
    # per-frame detection arrays
    det = {f: df[df.frame == f][['centroid-0', 'centroid-1', 'intensity_mean']].values for f in frames}
    tracks = []  # each: {'pos':(x,y),'cells':{t_idx:(x,y,I)}}
    for f in frames:
        ti = fidx[f]; D = det[f]
        used = set()
        for tr in tracks:
            if D.shape[0] == 0: break
            px, py = tr['pos']
            d = np.hypot(D[:, 0] - px, D[:, 1] - py)
            for j in np.argsort(d):
                if j in used: continue
                if d[j] <= MAX_LINK:
                    tr['cells'][ti] = (D[j, 0], D[j, 1], D[j, 2]); tr['pos'] = (D[j, 0], D[j, 1]); used.add(j)
                break
        for j in range(D.shape[0]):
            if j not in used:
                tracks.append({'pos': (D[j, 0], D[j, 1]), 'cells': {ti: (D[j, 0], D[j, 1], D[j, 2])}})
    keep = [tr for tr in tracks if len(tr['cells']) >= COVER * T]
    if len(keep) < 4:
        return None, None
    N = len(keep)
    traj = np.full((T, N), np.nan, dtype=np.float32); pos = np.zeros((N, 2), dtype=np.float32)
    for n, tr in enumerate(keep):
        ts = sorted(tr['cells']); xs = [tr['cells'][t][2] for t in ts]
        traj[ts, n] = xs
        # interpolate gaps
        col = traj[:, n]; nans = np.isnan(col)
        if nans.any():
            col[nans] = np.interp(np.where(nans)[0], np.where(~nans)[0], col[~nans]); traj[:, n] = col
        pos[n] = np.mean([tr['cells'][t][:2] for t in ts], axis=0)
    traj = (traj - np.nanmean(traj)) / (np.nanstd(traj) + 1e-9)
    return traj, pos


class DiffOnlyT(nn.Module):
    def __init__(self): super().__init__(); self.D = nn.Parameter(torch.tensor(0.1)); self.is_diff = True


def step_pred(model, C, L):
    """Vectorized one-step prediction for ALL transitions at once.
    C: (N, n) states (each column a timestep); returns pred (N, n)."""
    if getattr(model, 'is_diff', False):
        return C - torch.abs(model.D) * (L @ C)
    N, n = C.shape
    react = model.reaction_kans[0](C.reshape(-1)).reshape(N, n)     # feature-wise (M=1) reaction
    return C + react - torch.abs(model.D) * (L @ C)


def _cols(X, idx):
    return X[idx].T.contiguous()           # (N, len(idx))


def train_onestep(model, traj, L, tr_e, va_e, te_e, epochs=1500, lr=5e-3, patience=80, l1=1e-4):
    model = model.to(DEVICE); X = torch.tensor(traj, dtype=torch.float32).to(DEVICE); Lt = L.to(DEVICE)
    Ctr, Ytr = _cols(X, tr_e), _cols(X, [t+1 for t in tr_e])
    Cva, Yva = _cols(X, va_e), _cols(X, [t+1 for t in va_e])
    Cte, Yte = _cols(X, te_e), _cols(X, [t+1 for t in te_e])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    best, bs, wait = 1e9, None, 0
    for ep in range(epochs):
        model.train()
        l = F.mse_loss(step_pred(model, Ctr, Lt), Ytr)
        if l1 > 0: l = l + l1 * sum(p.abs().sum() for n, p in model.named_parameters() if 'coeff' in n)
        opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad(): vl = F.mse_loss(step_pred(model, Cva, Lt), Yva).item()
        if vl < best - 1e-7: best, bs, wait = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience: break
    if bs: model.load_state_dict(bs)
    model.eval()
    with torch.no_grad():
        pred = step_pred(model, Cte, Lt)
        per_t = torch.sqrt(((pred - Yte) ** 2).mean(0))     # per-transition RMSE over N cells
    return float(per_t.mean().item())


def persistence(traj, te_e):
    return float(np.mean([np.sqrt(np.mean((traj[t+1]-traj[t])**2)) for t in te_e]))

def ar1(traj, tr_e, te_e):
    X = np.concatenate([traj[t] for t in tr_e]); Y = np.concatenate([traj[t+1] for t in tr_e])
    a, b = np.linalg.lstsq(np.vstack([X, np.ones_like(X)]).T, Y, rcond=None)[0]
    return float(np.mean([np.sqrt(np.mean((traj[t+1]-(a*traj[t]+b))**2)) for t in te_e]))


def run_exp(sid):
    df = load_csv(sid); traj, pos = track(df)
    if traj is None: return None
    T, N = traj.shape
    L_norm, W, deg = build_knn_graph(pos, k=min(6, N-1))
    L = torch.tensor(L_norm, dtype=torch.float32)
    n_tr, n_va = int(0.6*T), int(0.7*T)
    tr_e = list(range(0, n_tr-1)); va_e = list(range(n_tr, n_va-1)); te_e = list(range(n_va, T-1))
    if len(tr_e) < 5 or len(va_e) < 2 or len(te_e) < 3: return None
    out = {'N': int(N), 'T': int(T), 'avg_degree': round(float(deg), 2), 'one_step_rmse': {}, 'D_theta': []}
    out['one_step_rmse']['Persistence'] = persistence(traj, te_e)
    out['one_step_rmse']['AR(1)'] = ar1(traj, tr_e, te_e)
    full_v, self_v, shuf_v, diff_v, Ds = [], [], [], [], []
    for s in range(N_SEEDS):
        torch.manual_seed(s); np.random.seed(s)
        m = RD_GKAN(1, G=8, k=3, x_range=(-4, 4)); full_v.append(train_onestep(m, traj, L, tr_e, va_e, te_e)); Ds.append(abs(m.D.item()))
        m = RD_GKAN(1, G=8, k=3, x_range=(-4, 4)); m.D.data.zero_(); m.D.requires_grad_(False); self_v.append(train_onestep(m, traj, L, tr_e, va_e, te_e))
        perm = np.random.permutation(N); Ws = W[perm][:, perm]; Ls = torch.tensor((np.diag(Ws.sum(1))-Ws)/N, dtype=torch.float32)
        m = RD_GKAN(1, G=8, k=3, x_range=(-4, 4)); shuf_v.append(train_onestep(m, traj, Ls, tr_e, va_e, te_e))
        diff_v.append(train_onestep(DiffOnlyT(), traj, L, tr_e, va_e, te_e, l1=0))
    out['one_step_rmse']['Full'] = {'mean': float(np.mean(full_v)), 'std': float(np.std(full_v))}
    out['one_step_rmse']['Self-only'] = {'mean': float(np.mean(self_v)), 'std': float(np.std(self_v))}
    out['one_step_rmse']['Shuffled'] = {'mean': float(np.mean(shuf_v)), 'std': float(np.std(shuf_v))}
    out['one_step_rmse']['Diff-only'] = {'mean': float(np.mean(diff_v)), 'std': float(np.std(diff_v))}
    out['D_theta'] = float(np.mean(Ds))
    return out


def main():
    print(f"Device: {DEVICE}")
    res = {}
    for sid in EXPS:
        try:
            r = run_exp(sid)
        except Exception as e:
            r = None; print(f"  {sid}: ERROR {e}")
        if r: res[sid] = r; print(f"  {sid}: N={r['N']} T={r['T']} | Full={r['one_step_rmse']['Full']['mean']:.3f} "
                                  f"Self={r['one_step_rmse']['Self-only']['mean']:.3f} Shuf={r['one_step_rmse']['Shuffled']['mean']:.3f} "
                                  f"Diff={r['one_step_rmse']['Diff-only']['mean']:.3f} Pers={r['one_step_rmse']['Persistence']:.3f} D={r['D_theta']:.3f}")
    # aggregate
    ids = list(res.keys())
    full = np.array([res[k]['one_step_rmse']['Full']['mean'] for k in ids])
    self_ = np.array([res[k]['one_step_rmse']['Self-only']['mean'] for k in ids])
    shuf = np.array([res[k]['one_step_rmse']['Shuffled']['mean'] for k in ids])
    diff = np.array([res[k]['one_step_rmse']['Diff-only']['mean'] for k in ids])
    pers = np.array([res[k]['one_step_rmse']['Persistence'] for k in ids])
    n = len(ids)
    agg = {'n_experiments': n, 'experiments': ids,
           'full_beats_self': int((full < self_).sum()), 'full_beats_shuffled': int((full < shuf).sum()),
           'full_beats_persistence': int((full < pers).sum()),
           'mean': {'Full': float(full.mean()), 'Self-only': float(self_.mean()), 'Shuffled': float(shuf.mean()),
                    'Diff-only': float(diff.mean()), 'Persistence': float(pers.mean())},
           'D_theta_mean': float(np.mean([res[k]['D_theta'] for k in ids]))}
    try:
        from scipy.stats import wilcoxon
        agg['wilcoxon_full_vs_self_two_sided'] = float(wilcoxon(full, self_, alternative='two-sided').pvalue)
    except Exception as e:
        agg['wilcoxon_full_vs_self_two_sided'] = None
    res['_aggregate'] = agg
    json.dump(res, open(os.path.join(RES, 'staph_rigorous.json'), 'w'), indent=2)
    print("\n=== AGGREGATE (%d experiments) ===" % n)
    print(f"Full beats Self-only:  {agg['full_beats_self']}/{n}")
    print(f"Full beats Shuffled:   {agg['full_beats_shuffled']}/{n}")
    print(f"Full beats Persistence:{agg['full_beats_persistence']}/{n}")
    print(f"mean RMSE: Full {agg['mean']['Full']:.3f}, Self {agg['mean']['Self-only']:.3f}, "
          f"Shuf {agg['mean']['Shuffled']:.3f}, Diff {agg['mean']['Diff-only']:.3f}, Pers {agg['mean']['Persistence']:.3f}")
    print(f"Wilcoxon Full vs Self (two-sided): p={agg['wilcoxon_full_vs_self_two_sided']}")
    print(f"mean D_theta: {agg['D_theta_mean']:.3f}")
    print("saved results/staph_rigorous.json")


if __name__ == '__main__':
    main()
