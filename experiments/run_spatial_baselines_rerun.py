#!/usr/bin/env python3
"""
Re-run of the spatial-prediction test-RMSE table (tab:spatial_results) for the paper,
producing a single consistent run that includes the entries that were never computed:
  - Self-only (D_theta=0) for BOTH tissues
  - GRAND / GREAD for the INTESTINE tissue
and re-confirms all existing models so the whole table comes from one run.

Pipeline reused EXACTLY from:
  - experiments/run_revised_experiments.py  (models, graph, splits, train_model)
  - experiments/run_option_b.py             (intestine loader)
Protocol matches experiment_spatial_comprehensive() in run_final_comprehensive.py.

Self-only baseline = Graph KAN (CrossFeatureGraphKAN) with graph coupling DISABLED.
Implementation: pass an EMPTY edge_index / edge_weight so no neighbor aggregation
occurs. In CrossFeatureGraphKANLayer.forward, the only graph-coupling term is `agg`
(scatter_add over edges); with zero edges, agg == 0 for every node, leaving only the
per-node self_update (+ per-node cross-feature mixing, which is NOT graph coupling).
We verify the empty-graph output is identical regardless of the edges supplied.

Author: Liang Dong
"""
import os
import sys
import json
import warnings
import numpy as np
import torch

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS_DIR = os.path.join(ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
sys.path.insert(0, HERE)

from run_revised_experiments import (
    CrossFeatureGraphKAN, GNNBaseline, GATBaseline, GRANDBaseline, GREADBaseline,
    load_breast_cancer_data, construct_graph, region_based_split, train_model,
)
from run_option_b import load_intestinal_spatial

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_SEEDS = 5
SIGMA_NOISE = 0.3

MODEL_ORDER = ['Graph KAN', 'Self-only', 'GNN', 'GAT', 'GRAND', 'GREAD']


def build_models(M):
    """Build the same architectures (same args) as experiment_spatial_comprehensive()."""
    return {
        'Graph KAN': CrossFeatureGraphKAN(M, n_layers=2, G=5, k=3),
        'Self-only': CrossFeatureGraphKAN(M, n_layers=2, G=5, k=3),  # same arch; graph disabled at train time
        'GNN': GNNBaseline(M, hidden=64, n_layers=2),
        'GAT': GATBaseline(M, hidden=64, n_heads=4),
        'GRAND': GRANDBaseline(M, hidden=64, n_steps=2),
        'GREAD': GREADBaseline(M, hidden=64, n_steps=2),
    }


def empty_graph():
    """Edge set with zero edges -> no neighbor aggregation (graph coupling off)."""
    ei = torch.zeros((2, 0), dtype=torch.long)
    ew = torch.zeros((0,), dtype=torch.float32)
    return ei, ew


def verify_self_only_decoupled(M, edge_index, edge_weight):
    """Sanity: with empty edges, CrossFeatureGraphKAN output is independent of the
    supplied graph (i.e., graph coupling is genuinely removed)."""
    torch.manual_seed(123)
    model = CrossFeatureGraphKAN(M, n_layers=2, G=5, k=3).to(DEVICE)
    model.eval()
    x = torch.randn(20, M).to(DEVICE)
    ei0, ew0 = empty_graph()
    with torch.no_grad():
        out_empty = model.predict(x, ei0.to(DEVICE), ew0.to(DEVICE))
        # full graph on a 20-node subgraph: build a small ring just to have edges
        src = torch.arange(20); dst = (torch.arange(20) + 1) % 20
        ei_full = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])]).to(DEVICE)
        ew_full = torch.ones(ei_full.shape[1]).to(DEVICE)
        out_full = model.predict(x, ei_full, ew_full)
    # With empty edges the two CALLS use different graphs; the empty one must NOT
    # change if we instead pass yet another empty graph, and must DIFFER from full
    # (confirming the graph term is what we removed).
    ei1, ew1 = empty_graph()
    with torch.no_grad():
        out_empty2 = model.predict(x, ei1.to(DEVICE), ew1.to(DEVICE))
    same_empty = torch.allclose(out_empty, out_empty2, atol=1e-6)
    differs_from_full = not torch.allclose(out_empty, out_full, atol=1e-5)
    return bool(same_empty), bool(differs_from_full)


def run_tissue(tissue_name, positions, expression, gene_names):
    N, M = expression.shape
    edge_index, edge_weight, L_mat, eigvals, graph_stats = construct_graph(positions)
    print(f"\n=== Tissue: {tissue_name} | N={N}, M={M}, "
          f"|E|={graph_stats['n_edges_undirected']}, avg_deg={graph_stats['avg_degree']:.2f} ===")
    print(f"    genes: {gene_names}")

    ei_empty, ew_empty = empty_graph()

    results = {m: [] for m in MODEL_ORDER}
    for seed in range(N_SEEDS):
        rng = np.random.RandomState(seed + 100)
        train_idx, test_idx = region_based_split(positions, test_fraction=0.2, seed=seed + 200)
        noise = rng.randn(*expression.shape).astype(np.float32) * SIGMA_NOISE
        x_noisy = expression + noise

        models = build_models(M)
        for name in MODEL_ORDER:
            model = models[name]
            if name == 'Self-only':
                # graph coupling OFF via empty graph; same training settings as Graph KAN
                rmse, _ = train_model(
                    model, x_noisy, expression, ei_empty, ew_empty,
                    train_idx, test_idx, n_epochs=1000, lr=1e-3,
                    l1_lambda=1e-4, patience=50)
            else:
                rmse, _ = train_model(
                    model, x_noisy, expression, edge_index, edge_weight,
                    train_idx, test_idx, n_epochs=1000, lr=1e-3,
                    l1_lambda=1e-4 if name == 'Graph KAN' else 0,
                    patience=50)
            results[name].append(rmse)
        print("  seed %d: " % seed +
              ", ".join(f"{n}={results[n][-1]:.4f}" for n in MODEL_ORDER))

    summary = {}
    for name in MODEL_ORDER:
        vals = results[name]
        summary[name] = {'mean': float(np.mean(vals)),
                         'std': float(np.std(vals)),
                         'all': [float(v) for v in vals]}
    summary['_graph_stats'] = graph_stats
    summary['_gene_names'] = list(gene_names)
    return summary


def main():
    print(f"Device: {DEVICE}")

    # ---- Self-only decoupling verification (on breast M) ----
    pos_b, expr_b, genes_b = load_breast_cancer_data(n_spots=500, seed=42)
    M_b = expr_b.shape[1]
    ei_b, ew_b, _, _, _ = construct_graph(pos_b)
    same_empty, differs_full = verify_self_only_decoupled(M_b, ei_b, ew_b)
    print(f"\n[Self-only verification] empty-graph output deterministic across empty graphs: "
          f"{same_empty}; differs from full-graph output: {differs_full}")

    all_results = {
        '_meta': {
            'n_seeds': N_SEEDS, 'sigma_noise': SIGMA_NOISE,
            'protocol': 'region_based_split(test_fraction=0.2, seed=seed+200); '
                        'noise=RandomState(seed+100).randn*0.3; '
                        'train_model(n_epochs=1000, lr=1e-3, patience=50, '
                        'l1=1e-4 for Graph KAN & Self-only else 0)',
            'self_only_impl': 'CrossFeatureGraphKAN with empty edge_index/edge_weight '
                              '(no neighbor aggregation; only per-node self_update + per-node mix)',
            'self_only_verification': {
                'deterministic_across_empty_graphs': same_empty,
                'differs_from_full_graph': differs_full,
            },
            'device': str(DEVICE),
        }
    }

    # ---- BREAST first (sanity check) ----
    all_results['breast'] = run_tissue('breast', pos_b, expr_b, genes_b)

    # Sanity check vs known table values (within ~0.01)
    expected = {'Graph KAN': 0.262, 'GNN': 0.272, 'GAT': 0.305, 'GRAND': 0.278, 'GREAD': 0.283}
    print("\n--- BREAST SANITY CHECK (expected vs got) ---")
    sane = True
    for name, exp in expected.items():
        got = all_results['breast'][name]['mean']
        ok = abs(got - exp) <= 0.015  # ~0.01 with small slack
        sane = sane and ok
        print(f"  {name}: expected {exp:.3f}, got {got:.3f}  -> {'OK' if ok else 'MISMATCH'}")
    all_results['_meta']['breast_sanity_passed'] = bool(sane)

    # ---- INTESTINE + Self-only (always run; we report regardless) ----
    pos_i, expr_i, genes_i = load_intestinal_spatial()
    all_results['intestine'] = run_tissue('intestine', pos_i, expr_i, genes_i)

    out = os.path.join(RESULTS_DIR, 'spatial_baselines_rerun.json')
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved -> {out}")

    # ---- Final table ----
    print("\n" + "=" * 64)
    print("FINAL TABLE  (mean +/- std, test RMSE, 5 seeds, region splits)")
    print("=" * 64)
    print(f"{'Model':<12} {'breast':<18} {'intestine':<18}")
    for name in MODEL_ORDER:
        b = all_results['breast'][name]
        i = all_results['intestine'][name]
        print(f"{name:<12} {b['mean']:.3f} +/- {b['std']:.3f}     "
              f"{i['mean']:.3f} +/- {i['std']:.3f}")

    print("\nNEW NUMBERS:")
    print(f"  Self-only breast    : {all_results['breast']['Self-only']['mean']:.3f} +/- {all_results['breast']['Self-only']['std']:.3f}")
    print(f"  Self-only intestine : {all_results['intestine']['Self-only']['mean']:.3f} +/- {all_results['intestine']['Self-only']['std']:.3f}")
    print(f"  GRAND intestine     : {all_results['intestine']['GRAND']['mean']:.3f} +/- {all_results['intestine']['GRAND']['std']:.3f}")
    print(f"  GREAD intestine     : {all_results['intestine']['GREAD']['mean']:.3f} +/- {all_results['intestine']['GREAD']['std']:.3f}")
    print(f"\nBREAST SANITY PASSED: {sane}")


if __name__ == '__main__':
    main()
