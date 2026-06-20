#!/usr/bin/env python3
"""Re-run the wound-healing spatial RMSE with the CONSTRAINED RD-GKAN (eq. 11),
making the wound table consistent with the paper's architecture, and fixing two
rigor issues in the original run:
  (i)  use REAL Visium coordinates (matched by barcode), not pseudo-positions;
  (ii) report mean +/- s.d. over 5 seeds with region-based (leakage-resistant)
       splits, as in the breast/intestine spatial table.

Both models are trained under an IDENTICAL protocol (same split, same noisy
input, same epochs) via the shared train_model harness: the constrained RD-GKAN
(feature-wise B-spline reaction + Laplacian diffusion, steady-state
reconstruction) and the GNN baseline. Self-only (D_theta=0) is included as a
control showing the value of graph coupling.
"""
import os, sys, json, warnings, numpy as np
import torch, torch.nn as nn
from sklearn.cluster import KMeans
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_synthetic_rd import RD_GKAN, DEVICE, RESULTS_DIR
from run_new_datasets import build_knn_graph
from run_revised_experiments import GNNBaseline, train_model
from run_wound_splines import load_condition, COND, M, N_MAX, SIGMA

N_SEEDS = 5


class RDGKANRecon(nn.Module):
    """Constrained RD-GKAN wrapped in the (x, edge_index, edge_weight) predict
    interface so it trains under the same harness as the baselines. The fixed
    normalized Laplacian is stored as a buffer; edges are ignored at predict."""
    def __init__(self, M, L_norm, G=5, k=3, freeze_diffusion=False):
        super().__init__()
        self.rd = RD_GKAN(M, G=G, k=k, x_range=(-4.0, 4.0))
        if freeze_diffusion:                      # self-only control: D_theta = 0
            self.rd.D.data.zero_(); self.rd.D.requires_grad_(False)
        self.register_buffer('L_norm', torch.tensor(L_norm, dtype=torch.float32))

    def predict(self, x, edge_index=None, edge_weight=None):
        return self.rd.forward(x, self.L_norm)

    def forward(self, x, edge_index=None, edge_weight=None):
        return self.predict(x)


def edges_from_W(W):
    N = W.shape[0]
    src, dst, w = [], [], []
    nz = np.argwhere(W > 0)
    for i, j in nz:
        src.append(int(i)); dst.append(int(j)); w.append(float(W[i, j]))
    ei = torch.tensor([src, dst], dtype=torch.long)
    ew = torch.tensor(w, dtype=torch.float32)
    return ei, ew


def region_split(positions, seed, n_clusters=10):
    N = len(positions)
    nc = max(2, min(n_clusters, N // 10))
    labels = KMeans(n_clusters=nc, random_state=seed, n_init=10).fit_predict(positions)
    test_cluster = np.random.RandomState(seed).choice(nc)
    test_mask = labels == test_cluster
    return np.where(~test_mask)[0], np.where(test_mask)[0]


def main():
    out = {'M': M, 'n_seeds': N_SEEDS, 'noise_sigma': SIGMA,
           'note': 'constrained RD-GKAN; real Visium coords; region-based splits',
           'conditions': {}}
    for label, prefix in COND:
        expr, genes, pos = load_condition(prefix)
        N = expr.shape[0]
        rng = np.random.RandomState(42)
        if N > N_MAX:
            idx = rng.choice(N, N_MAX, replace=False)
            expr, pos = expr[idx], pos[idx]
            N = N_MAX
        # per-condition top-M highly variable genes (Fano), as in the original protocol
        mean = expr.mean(0) + 1e-10
        fano = expr.var(0) / mean
        fano[mean < 0.5] = 0.0
        top = np.argsort(fano)[-M:][::-1]
        x = np.log1p(expr[:, top].astype(np.float32))
        x = (x - x.mean(0)) / (x.std(0) + 1e-10)
        L_norm, W, avg_deg = build_knn_graph(pos, k=min(6, N - 1))
        ei, ew = edges_from_W(W)

        res = {m: [] for m in ['RD-GKAN', 'Self-only', 'GNN']}
        for s in range(N_SEEDS):
            torch.manual_seed(s); np.random.seed(s)
            train_idx, test_idx = region_split(pos, seed=s)
            noise = np.random.RandomState(1000 + s).randn(*x.shape).astype(np.float32) * SIGMA
            x_noisy = x + noise
            models = {
                'RD-GKAN':   RDGKANRecon(M, L_norm, G=5, k=3),
                'Self-only': RDGKANRecon(M, L_norm, G=5, k=3, freeze_diffusion=True),
                'GNN':       GNNBaseline(M, hidden=64, n_layers=2),
            }
            for name, model in models.items():
                rmse, _ = train_model(model, x_noisy, x, ei, ew, train_idx, test_idx,
                                      n_epochs=600, lr=3e-3, l1_lambda=1e-4, patience=50)
                res[name].append(float(rmse))
        cond = {'N': int(N), 'avg_degree': round(float(avg_deg), 2)}
        for m in res:
            cond[m] = {'mean': round(float(np.mean(res[m])), 4),
                       'std': round(float(np.std(res[m])), 4),
                       'all': [round(r, 4) for r in res[m]]}
        out['conditions'][label] = cond
        print(f"{label}: N={N} deg={avg_deg:.2f} | "
              + " | ".join(f"{m} {cond[m]['mean']:.3f}±{cond[m]['std']:.3f}" for m in res))

    with open(os.path.join(RESULTS_DIR, 'wound_rmse_constrained.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print("\nsaved results/wound_rmse_constrained.json")


if __name__ == '__main__':
    main()
