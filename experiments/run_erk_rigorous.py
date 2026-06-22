#!/usr/bin/env python3
"""
RIGOROUS ERK MDCK-waves temporal re-analysis (negative control), using the
IDENTICAL leakage-proof protocol as the S. aureus re-analysis
(run_staph_rigorous): chronological 60% train / 10% validation (early stopping)
/ 30% test, test read once. Controls: Full RD-GKAN (eq.11), Self-only
(D_theta=0), Shuffled-graph (permuted adjacency), Diffusion-only (psi=0),
Persistence (c_hat(t+1)=c(t)), AR(1).

The ERK trajectory construction is copied verbatim from
run_new_datasets.experiment_erk_waves / run_temporal_baselines.load_erk_trajectory
so the preprocessing matches the published ERK result; only the train/val/test
split and selection rule are upgraded to the leakage-proof protocol shared with
the S. aureus analysis. The previous ERK numbers used a 70/30 split with no
held-out validation segment.

Author: Liang Dong
"""
import os, sys, json, warnings, numpy as np, pandas as pd, torch
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, 'results'); sys.path.insert(0, HERE)
from run_synthetic_rd import RD_GKAN, DEVICE
from run_new_datasets import build_knn_graph, DATA_DIR
from run_staph_rigorous import train_onestep, persistence, ar1, DiffOnlyT, step_pred

N_SEEDS = 5


def load_erk():
    """Verbatim ERK trajectory construction (see experiment_erk_waves); returns
    traj as (T, N), normalized graph Laplacian L_norm, adjacency W, N, T, deg."""
    mdck_dir = os.path.join(DATA_DIR, 'ERK_Collective', 'MDCK-waves-extracted', 'MDCK-waves')
    csv_file = None
    for root, dirs, files in os.walk(mdck_dir):
        for f in files:
            if f.endswith('.csv.gz') and 'clean_tracks' in f:
                csv_file = os.path.join(root, f); break
        if csv_file: break
    if csv_file is None:
        for root, dirs, files in os.walk(os.path.join(DATA_DIR, 'ERK_Collective')):
            for f in files:
                if f.endswith('.csv.gz'): csv_file = os.path.join(root, f); break
            if csv_file: break
    df = pd.read_csv(csv_file, compression='gzip')
    time_col = [c for c in df.columns if 'Metadata_T' in c or 'time' in c.lower()][0]
    x_col = [c for c in df.columns if 'Location_Center_X' in c or 'center_x' in c.lower()][0]
    y_col = [c for c in df.columns if 'Location_Center_Y' in c or 'center_y' in c.lower()][0]
    track_col = [c for c in df.columns if 'track_id' in c.lower()][0]
    ratio_cols = [c for c in df.columns if 'MeanIntensity_imRATIO' in c or 'ratio' in c.lower()]
    if not ratio_cols:
        ratio_cols = [c for c in df.columns if 'FRET' in c or 'Intensity' in c]
    signal_col = ratio_cols[0]
    timepoints = sorted(df[time_col].unique())
    tracks = sorted(df[track_col].unique())
    tc = df.groupby(track_col)[time_col].nunique()
    complete = tc[tc >= len(timepoints) * 0.8].index.tolist()
    if len(complete) < 10:
        complete = tc.nlargest(min(50, len(tracks))).index.tolist()
    N = len(complete); T_total = min(len(timepoints), 100)
    tmap = {t: i for i, t in enumerate(complete)}
    traj = np.full((T_total, N, 1), np.nan, dtype=np.float32)
    pos = np.zeros((N, 2), dtype=np.float32)
    for _, row in df.iterrows():
        tid = row[track_col]; t = row[time_col]
        if tid in tmap and t < T_total:
            i = tmap[tid]; traj[int(t), i, 0] = row[signal_col]
            if t == 0: pos[i] = [row[x_col], row[y_col]]
    for i in range(N):
        vals = traj[:, i, 0]; nans = np.isnan(vals)
        if nans.all(): traj[:, i, 0] = 0
        elif nans.any():
            traj[:, i, 0] = np.interp(np.arange(T_total), np.where(~nans)[0], vals[~nans])
    traj = (traj - np.nanmean(traj)) / (np.nanstd(traj) + 1e-10)
    L_norm, W, avg_deg = build_knn_graph(pos, k=min(6, N - 1))
    return traj[..., 0], L_norm, W, N, T_total, float(avg_deg)


def rollout_rmse(model, traj, L, te_idx):
    """Autoregressive multi-step rollout from the first test timepoint;
    mean RMSE over the test horizon (teacher-free)."""
    Lt = L.to(DEVICE); model.eval()
    with torch.no_grad():
        c = torch.tensor(traj[te_idx[0]], dtype=torch.float32).reshape(-1, 1).to(DEVICE)
        errs = []
        for k in range(1, len(te_idx)):
            c = step_pred(model, c, Lt)
            tgt = torch.tensor(traj[te_idx[k]], dtype=torch.float32).reshape(-1, 1).to(DEVICE)
            errs.append(torch.sqrt(((c - tgt) ** 2).mean()).item())
    return float(np.mean(errs))


def persistence_rollout(traj, te_idx):
    """Rollout for persistence = predict the (constant) first test frame."""
    c0 = traj[te_idx[0]]
    return float(np.mean([np.sqrt(np.mean((c0 - traj[te_idx[k]]) ** 2)) for k in range(1, len(te_idx))]))


def main():
    print(f"Device: {DEVICE}")
    traj, L_norm, W, N, T, deg = load_erk()
    L = torch.tensor(L_norm, dtype=torch.float32)
    n_tr, n_va = int(0.6 * T), int(0.7 * T)
    tr_e = list(range(0, n_tr - 1)); va_e = list(range(n_tr, n_va - 1)); te_e = list(range(n_va, T - 1))
    print(f"ERK: N={N} T={T} deg={deg:.2f} | train={len(tr_e)} val={len(va_e)} test={len(te_e)} transitions")

    out = {'dataset': 'ERK MDCK waves', 'N': int(N), 'T_total': int(T), 'avg_degree': round(deg, 2),
           'n_seeds': N_SEEDS, 'split': '60/10/30 chronological (validation-based early stopping)',
           'one_step_rmse': {}}
    out['one_step_rmse']['Persistence'] = persistence(traj, te_e)
    out['one_step_rmse']['AR(1)'] = ar1(traj, tr_e, te_e)
    te_idx = list(range(n_va, T))
    full, self_, shuf, diff, Ds = [], [], [], [], []
    full_ro, self_ro = [], []
    for s in range(N_SEEDS):
        torch.manual_seed(s); np.random.seed(s)
        m = RD_GKAN(1, G=8, k=3, x_range=(-4, 4)); full.append(train_onestep(m, traj, L, tr_e, va_e, te_e)); Ds.append(abs(m.D.item())); full_ro.append(rollout_rmse(m, traj, L, te_idx))
        m = RD_GKAN(1, G=8, k=3, x_range=(-4, 4)); m.D.data.zero_(); m.D.requires_grad_(False); self_.append(train_onestep(m, traj, L, tr_e, va_e, te_e)); self_ro.append(rollout_rmse(m, traj, L, te_idx))
        perm = np.random.permutation(N); Ws = W[perm][:, perm]; Ls = torch.tensor((np.diag(Ws.sum(1)) - Ws) / N, dtype=torch.float32)
        m = RD_GKAN(1, G=8, k=3, x_range=(-4, 4)); shuf.append(train_onestep(m, traj, Ls, tr_e, va_e, te_e))
        diff.append(train_onestep(DiffOnlyT(), traj, L, tr_e, va_e, te_e, l1=0))
    for nm, v in [('Full', full), ('Self-only', self_), ('Shuffled', shuf), ('Diff-only', diff)]:
        out['one_step_rmse'][nm] = {'mean': float(np.mean(v)), 'std': float(np.std(v))}
    out['D_theta'] = float(np.mean(Ds))
    out['rollout_rmse'] = {'Persistence': persistence_rollout(traj, te_idx),
                           'Self-only': {'mean': float(np.mean(self_ro)), 'std': float(np.std(self_ro))},
                           'Full': {'mean': float(np.mean(full_ro)), 'std': float(np.std(full_ro))}}
    with open(os.path.join(RES, 'erk_rigorous.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print("\n=== ERK one-step RMSE (60/10/30, val-based; lower=better) ===")
    for k, v in out['one_step_rmse'].items():
        print(f"  {k:<12} {v if isinstance(v, float) else v['mean']:.4f}"
              + ("" if isinstance(v, float) else f" +/- {v['std']:.4f}"))
    print(f"  D_theta(mean) = {out['D_theta']:.4f}")
    print("=== rollout RMSE (autoregressive over test horizon) ===")
    rr = out['rollout_rmse']
    print(f"  Persistence {rr['Persistence']:.4f} | Self-only {rr['Self-only']['mean']:.4f} | Full {rr['Full']['mean']:.4f}")
    print("saved results/erk_rigorous.json")


if __name__ == '__main__':
    main()
