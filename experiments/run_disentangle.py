#!/usr/bin/env python3
"""
Disentangling control: separate the effect of (i) the test-leakage protocol fix
and (ii) the model-identity fix on the static-spatial RMSE, using the ORIGINAL
loaders, the ORIGINAL graph construction, and the ORIGINAL model/training code.

2x2 design on each tissue (breast, intestine), 5 seeds:
  model in {CrossFeatureGraphKAN (original spatial-table model),
            constrained eq.(11) RD_GKAN (paper's described model)}
  protocol in {LEAKY  : original train_model (early-stop/checkpoint on TEST) +
                        2-way region_based_split + full (transductive) graph,
               RIGOROUS: train_rigorous (validation-based) + 3-way region split +
                        inductive subgraph}

Validation target: CrossFeatureGraphKAN + LEAKY must reproduce the published
breast number (~0.262), confirming the pipeline/data are the original ones.
"""
import os, sys, json, warnings, numpy as np, torch, torch.nn as nn
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, 'results'); sys.path.insert(0, HERE)
from run_revised_experiments import (CrossFeatureGraphKAN, construct_graph,
                                     load_breast_cancer_data, region_based_split, train_model)
from run_option_b import load_intestinal_spatial
from run_synthetic_rd import RD_GKAN
from run_spatial_rigorous import train_rigorous, region_split_3way, W_from_edges, N_SEEDS, SIGMA, DEVICE


class ConstrainedWrap(nn.Module):
    """Constrained eq.(11) RD_GKAN with a fixed full-graph Laplacian, exposing the
    (x, edge_index, edge_weight) predict interface so it runs under the ORIGINAL
    leaky train_model (which ignores the matrix and passes edges)."""
    def __init__(self, M, L_norm):
        super().__init__()
        self.rd = RD_GKAN(M, G=5, k=3, x_range=(-4., 4.))
        self.register_buffer('L', torch.tensor(L_norm, dtype=torch.float32))
    def predict(self, x, ei=None, ew=None):
        return self.rd.forward(x, self.L)
    def forward(self, x, ei=None, ew=None):
        return self.predict(x)


def run(tissue, pos, expr, genes):
    N, M = expr.shape
    ei, ew, L_mat, eig, stats = construct_graph(pos)
    W = W_from_edges(ei, ew, N)
    L_full = (np.diag(W.sum(1)) - W) / N
    print(f"\n=== {tissue}: N={N} M={M} |E|={stats['n_edges_undirected']} "
          f"deg={stats['avg_degree']:.2f} genes={list(genes)[:4]}... ===")
    cells = {('CrossFeature', 'leaky'): [], ('CrossFeature', 'rigorous'): [],
             ('Constrained', 'leaky'): [], ('Constrained', 'rigorous'): []}
    for s in range(N_SEEDS):
        rng = np.random.RandomState(s + 100)
        noise = rng.randn(*expr.shape).astype(np.float32) * SIGMA
        x_noisy = expr + noise
        # ---- LEAKY (original): 2-way split, original train_model (test-selected) ----
        torch.manual_seed(s); np.random.seed(s)
        tr2, te2 = region_based_split(pos, test_fraction=0.2, seed=s + 200)
        cf = CrossFeatureGraphKAN(M, n_layers=2, G=5, k=3)
        r, _ = train_model(cf, x_noisy, expr, ei, ew, tr2, te2, n_epochs=1000, lr=1e-3, l1_lambda=1e-4, patience=50)
        cells[('CrossFeature', 'leaky')].append(r)
        cw = ConstrainedWrap(M, L_full)
        r, _ = train_model(cw, x_noisy, expr, ei, ew, tr2, te2, n_epochs=1000, lr=1e-3, l1_lambda=1e-4, patience=50)
        cells[('Constrained', 'leaky')].append(r)
        # ---- RIGOROUS: 3-way inductive, validation-selected ----
        torch.manual_seed(s); np.random.seed(s)
        tr3, va3, te3 = region_split_3way(pos, seed=s + 200)
        cf2 = CrossFeatureGraphKAN(M, n_layers=2, G=5, k=3)
        r, _ = train_rigorous(cf2, False, x_noisy, expr, W, ei, ew, tr3, va3, te3, N,
                              n_epochs=1000, lr=1e-3, l1=1e-4, patience=60)
        cells[('CrossFeature', 'rigorous')].append(r)
        rd = RD_GKAN(M, G=5, k=3, x_range=(-4., 4.))
        r, _ = train_rigorous(rd, True, x_noisy, expr, W, ei, ew, tr3, va3, te3, N,
                              n_epochs=1000, lr=1e-3, l1=1e-4, patience=60)
        cells[('Constrained', 'rigorous')].append(r)
    out = {}
    for k, v in cells.items():
        out['%s|%s' % k] = {'mean': float(np.mean(v)), 'std': float(np.std(v))}
    return out


def main():
    print(f"Device: {DEVICE}")
    results = {}
    pb, eb, gb = load_breast_cancer_data(n_spots=500, seed=42)
    results['breast'] = run('breast', pb, eb, gb)
    pi, ei_, gi = load_intestinal_spatial()
    results['intestine'] = run('intestine', pi, ei_, gi)
    with open(os.path.join(RES, 'disentangle.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("\n" + "=" * 60)
    print("2x2  (rows=model, cols=protocol)   mean RMSE +/- std")
    print("=" * 60)
    for t in results:
        d = results[t]
        print(f"\n{t}:")
        print(f"{'':<14}{'LEAKY (orig)':<20}{'RIGOROUS':<20}")
        for m in ['CrossFeature', 'Constrained']:
            lk = d[f'{m}|leaky']; rg = d[f'{m}|rigorous']
            print(f"{m:<14}{lk['mean']:.3f} +/- {lk['std']:.3f}     {rg['mean']:.3f} +/- {rg['std']:.3f}")
    print("\nVALIDATION: CrossFeature|leaky on breast should reproduce published ~0.262")
    print("saved results/disentangle.json")


if __name__ == '__main__':
    main()
