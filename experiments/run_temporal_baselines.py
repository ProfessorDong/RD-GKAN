#!/usr/bin/env python3
"""
Temporal baselines for the ERK negative-control claim (reviewer request #4):
adds PERSISTENCE (c_hat(t+1)=c(t)) and AR(1) (c_hat(t+1)=a*c(t)+b, fit on train)
one-step RMSE on the SAME ERK trajectory and 70/30 temporal split used by the
RD-GKAN one-step experiment, and re-confirms RD-GKAN / Self-only one-step under
the identical pipeline.

The ERK trajectory construction is copied verbatim from
run_new_datasets.experiment_erk_waves so the split and preprocessing match the
published ERK result exactly.

(The S. aureus temporal-controls generator is not in the repository, so its
trajectories cannot be reproduced without guessing the original preprocessing;
this script therefore covers only ERK, which is the negative control where the
persistence comparison is decisive.)

Author: Liang Dong
"""
import os, sys, json, warnings, numpy as np, pandas as pd, torch, torch.nn.functional as F
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, 'results'); sys.path.insert(0, HERE)
from run_new_datasets import (build_knn_graph, train_temporal_rdgkan, DATA_DIR)
from run_synthetic_rd import RD_GKAN, MLPReaction, DEVICE


def load_erk_trajectory():
    """Verbatim replication of experiment_erk_waves trajectory construction."""
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
    T_train = int(T_total * 0.7)
    return traj, L_norm, N, T_total, T_train, float(avg_deg)


def persistence_rmse(test_traj):
    e = [np.sqrt(np.mean((test_traj[t + 1] - test_traj[t]) ** 2)) for t in range(len(test_traj) - 1)]
    return float(np.mean(e))


def ar1_rmse(train_traj, test_traj):
    # global AR(1): c_{t+1} = a c_t + b, least squares over all train pairs/cells
    xs, ys = [], []
    for t in range(len(train_traj) - 1):
        xs.append(train_traj[t].ravel()); ys.append(train_traj[t + 1].ravel())
    X = np.concatenate(xs); Y = np.concatenate(ys)
    A = np.vstack([X, np.ones_like(X)]).T
    a, b = np.linalg.lstsq(A, Y, rcond=None)[0]
    e = [np.sqrt(np.mean((test_traj[t + 1] - (a * test_traj[t] + b)) ** 2))
         for t in range(len(test_traj) - 1)]
    return float(np.mean(e)), float(a), float(b)


def model_one_step(make_model, train_traj, test_traj, L_norm):
    L_t = torch.tensor(L_norm, dtype=torch.float32)
    model = make_model()
    train_temporal_rdgkan(model, train_traj, L_t, n_epochs=1500, patience=80)
    model.eval()
    with torch.no_grad():
        e = []
        for t in range(len(test_traj) - 1):
            c_t = torch.tensor(test_traj[t], dtype=torch.float32).to(DEVICE)
            c_n = torch.tensor(test_traj[t + 1], dtype=torch.float32).to(DEVICE)
            e.append(torch.sqrt(F.mse_loss(model.forward(c_t, L_t.to(DEVICE)), c_n)).item())
    return float(np.mean(e))


def main():
    print(f"Device: {DEVICE}")
    traj, L_norm, N, T_total, T_train, avg_deg = load_erk_trajectory()
    train_traj = traj[:T_train + 1]; test_traj = traj[T_train:]
    print(f"ERK: N={N}, T_total={T_total}, T_train={T_train}, deg={avg_deg:.2f}, "
          f"test steps={len(test_traj)-1}")

    out = {'dataset': 'ERK MDCK waves', 'N': N, 'T_total': T_total, 'T_train': T_train,
           'avg_degree': round(avg_deg, 2), 'split': '70/30 temporal', 'one_step_rmse': {}}
    out['one_step_rmse']['Persistence'] = persistence_rmse(test_traj)
    ar, a, b = ar1_rmse(train_traj, test_traj)
    out['one_step_rmse']['AR(1)'] = ar
    out['ar1_coef'] = {'a': a, 'b': b}
    # self-only = RD_GKAN with D=0; full = RD_GKAN (3 seeds each for stability)
    for name, mk in [('Self-only', lambda: _selfonly(RD_GKAN(1, G=8, k=3, x_range=(-4, 4)))),
                     ('RD-GKAN', lambda: RD_GKAN(1, G=8, k=3, x_range=(-4, 4))),
                     ('MLP-RD', lambda: MLPReaction(1, hidden=32))]:
        vals = []
        for s in range(3):
            torch.manual_seed(s); np.random.seed(s)
            vals.append(model_one_step(mk, train_traj, test_traj, L_norm))
        out['one_step_rmse'][name] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
    with open(os.path.join(RES, 'erk_temporal_baselines.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print("\n=== ERK one-step RMSE (lower=better) ===")
    for k, v in out['one_step_rmse'].items():
        print(f"  {k:<12} {v if isinstance(v, float) else v['mean']:.4f}"
              + ("" if isinstance(v, float) else f" +/- {v['std']:.4f}"))
    print(f"AR(1): a={a:.3f}, b={b:.3f}")
    print("saved results/erk_temporal_baselines.json")


def _selfonly(m):
    m.D.data.zero_(); m.D.requires_grad_(False); return m


if __name__ == '__main__':
    main()
