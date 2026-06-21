#!/usr/bin/env python3
"""
RIGOROUS re-run of (A) the wound-healing spatial RMSE table and (B) the breast
component-ablation table, under the SAME leakage-proof protocol as
run_spatial_rigorous.py:
  - 3-way region split (train/val/test, whole k-means regions);
  - inductive message passing (edges among the active node set only, /N_full);
  - early stopping / checkpoint on VALIDATION; test computed ONCE;
  - constrained eq.(11) RD-GKAN (feature-wise B-spline reaction + Laplacian
    diffusion) for all RD-GKAN variants.

Ablation variants (all constrained, same harness):
  Full              : RD_GKAN (spline reaction + D*L diffusion)
  Self-only         : RD_GKAN with D_theta = 0 (no graph coupling)
  Diffusion-only    : reaction removed (x + D*L@x)
  Linear reaction   : per-feature linear reaction a*x+b + D*L diffusion
  GNN               : GNN baseline

Author: Liang Dong
"""
import os, sys, json, warnings, numpy as np, torch, torch.nn as nn
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, 'results'); sys.path.insert(0, HERE)

from run_spatial_rigorous import (train_rigorous, region_split_3way, W_from_edges,
                                  DEVICE, N_SEEDS, SIGMA)
from run_synthetic_rd import RD_GKAN
from run_revised_experiments import GNNBaseline, construct_graph, load_breast_cancer_data
from run_new_datasets import build_knn_graph
from run_wound_splines import load_condition, COND, M, N_MAX, SIGMA as WSIGMA


# ---------- ablation variant modules (constrained, forward(x, L)) ----------
class DiffOnly(nn.Module):
    """x + (-|D| L x): diffusion only, no reaction."""
    def __init__(self):
        super().__init__(); self.D = nn.Parameter(torch.tensor(0.1))
    def forward(self, x, L):
        return x - torch.abs(self.D) * (L @ x)

class LinearRD(nn.Module):
    """x + (a*x + b) + (-|D| L x): per-feature linear reaction + diffusion."""
    def __init__(self, M):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(M)); self.b = nn.Parameter(torch.zeros(M))
        self.D = nn.Parameter(torch.tensor(0.1))
    def forward(self, x, L):
        return x + (self.a * x + self.b) - torch.abs(self.D) * (L @ x)


def edges_from_W(W):
    N = W.shape[0]; nz = np.argwhere(W > 0)
    src = [int(i) for i, j in nz]; dst = [int(j) for i, j in nz]
    w = [float(W[i, j]) for i, j in nz]
    return torch.tensor([src, dst], dtype=torch.long), torch.tensor(w, dtype=torch.float32)


def run_one(model, is_rd, x_in, x_tgt, W, ei, ew, pos, N, l1, epochs, lr):
    """5 seeds, rigorous protocol; returns mean/std/all test RMSE."""
    vals = []
    for s in range(N_SEEDS):
        torch.manual_seed(s); np.random.seed(s)
        tr, va, te = region_split_3way(pos, seed=s + 200)
        noise = np.random.RandomState(s + 100).randn(*x_tgt.shape).astype(np.float32) * SIGMA
        xn = x_tgt + noise if x_in is None else x_in
        # fresh model per seed
        m, isrd = model()
        tr_rmse, _ = train_rigorous(m, isrd, xn, x_tgt, W, ei, ew, tr, va, te, N,
                                    n_epochs=epochs, lr=lr, l1=l1, patience=60)
        vals.append(tr_rmse)
    return {'mean': float(np.mean(vals)), 'std': float(np.std(vals)), 'all': [float(v) for v in vals]}


# ===================== (A) WOUND =====================
def wound():
    print("\n##### WOUND (rigorous) #####")
    out = {'M': M, 'n_seeds': N_SEEDS, 'sigma': SIGMA,
           'protocol': '3-way region split; inductive; val early-stop; constrained RD-GKAN',
           'conditions': {}}
    for label, prefix in COND:
        expr, genes, pos = load_condition(prefix)
        N = expr.shape[0]
        rng = np.random.RandomState(42)
        if N > N_MAX:
            idx = rng.choice(N, N_MAX, replace=False); expr, pos = expr[idx], pos[idx]; N = N_MAX
        mean = expr.mean(0) + 1e-10; fano = expr.var(0) / mean; fano[mean < 0.5] = 0.0
        top = np.argsort(fano)[-M:][::-1]
        x = np.log1p(expr[:, top].astype(np.float32)); x = (x - x.mean(0)) / (x.std(0) + 1e-10)
        L_norm, W, avg_deg = build_knn_graph(pos, k=min(6, N - 1))
        ei, ew = edges_from_W(W)
        defs = {
            'RD-GKAN':   (lambda: (RD_GKAN(M, G=5, k=3, x_range=(-4., 4.)), True), 1e-4),
            'Self-only': (lambda: (s(RD_GKAN(M, G=5, k=3, x_range=(-4., 4.))), True), 1e-4),
            'GNN':       (lambda: (GNNBaseline(M, 64, 2), False), 0.0),
        }
        cond = {'N': int(N), 'avg_degree': round(float(avg_deg), 2)}
        for name, (mk, l1) in defs.items():
            cond[name] = run_one(mk, None, None, x, W, ei, ew, pos, N, l1, epochs=600, lr=3e-3)
        out['conditions'][label] = cond
        print(f"{label}: N={N} deg={avg_deg:.2f} | " +
              " | ".join(f"{m} {cond[m]['mean']:.3f}±{cond[m]['std']:.3f}" for m in ['RD-GKAN','Self-only','GNN']))
    return out


def s(model):
    """freeze diffusion (self-only) helper for RD_GKAN instance."""
    model.D.data.zero_(); model.D.requires_grad_(False); return model


# ===================== (B) ABLATION (breast) =====================
def ablation():
    print("\n##### ABLATION breast (rigorous) #####")
    pos, expr, genes = load_breast_cancer_data(n_spots=500, seed=42)
    N, Mb = expr.shape
    ei, ew, L_mat, eig, stats = construct_graph(pos)
    W = W_from_edges(ei, ew, N)
    variants = {
        'Full':           (lambda: (RD_GKAN(Mb, G=5, k=3, x_range=(-4., 4.)), True), 1e-4),
        'Self-only':      (lambda: (s(RD_GKAN(Mb, G=5, k=3, x_range=(-4., 4.))), True), 1e-4),
        'Diffusion-only': (lambda: (DiffOnly(), True), 0.0),
        'Linear':         (lambda: (LinearRD(Mb), True), 0.0),
        'GNN':            (lambda: (GNNBaseline(Mb, 64, 2), False), 0.0),
    }
    out = {'N': N, 'M': Mb, 'n_seeds': N_SEEDS, 'graph': stats}
    for name, (mk, l1) in variants.items():
        out[name] = run_one(mk, None, None, expr, W, ei, ew, pos, N, l1, epochs=1000, lr=1e-3)
        print(f"  {name:<16} {out[name]['mean']:.4f} +/- {out[name]['std']:.4f}")
    base = out['Full']['mean']
    out['_degradation_pct'] = {k: round(100*(out[k]['mean']-base)/base, 1)
                               for k in variants if k != 'Full'}
    print("  degradation vs Full:", out['_degradation_pct'])
    return out


def main():
    print(f"Device: {DEVICE}")
    results = {'wound': wound(), 'ablation': ablation()}
    with open(os.path.join(RES, 'wound_ablation_rigorous.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("\nsaved results/wound_ablation_rigorous.json")


if __name__ == '__main__':
    main()
