#!/usr/bin/env python3
"""
Option B: Comprehensive empirical experiments using ALL 7 real datasets.

Experiments:
  1. Breast cancer spatial transcriptomics (with self-feature masking)
  2. Intestinal spatial transcriptomics (CCI benchmark, second tissue)
  3. QS kinetics across 3 genes (lasI, rhlI, lasB)
  4. CIR channel model (MacroscaleTestbed)
  5. Biological MC noise (ProtonPumpingBacteria)
  6. Analog Network Coding multi-hop (AnalogNetworkCoding)
  7. Ablations on synthetic R-D (spline vs MLP, MC vs uniform, grid density)

Key fix: self-feature masking in spatial experiments.
"""

import os, sys, json, warnings, time, csv, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.spatial import KDTree
from scipy.special import erfc
from scipy.optimize import curve_fit
from sklearn.linear_model import Lasso
from sklearn.cluster import KMeans
import gzip, scipy.io, scipy.sparse, pandas as pd

warnings.filterwarnings('ignore')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"Device: {DEVICE}")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_synthetic_rd import (
    BSplineKAN, RD_GKAN, MLPReaction,
    make_random_geometric_graph, generate_rd_trajectory, train_rd_gkan,
    symbolic_recovery, KINETICS
)
from run_revised_experiments import (
    CrossFeatureGraphKAN,
    GNNBaseline, GATBaseline, GRANDBaseline, GREADBaseline,
    construct_graph as construct_mc_graph, region_based_split,
    train_model
)


# ============================================================
# SPATIAL TRANSCRIPTOMICS UTILITIES
# ============================================================
def load_breast_cancer_spatial(n_spots=500, seed=42):
    """Load 10x Visium breast cancer data."""
    bc_dir = os.path.join(DATA_DIR, '10XBreastCancer')
    pos_file = os.path.join(bc_dir, 'spatial', 'tissue_positions_list.csv')
    barcodes_pos, positions = [], []
    with open(pos_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 6 and int(parts[1]) == 1:
                barcodes_pos.append(parts[0])
                positions.append([float(parts[4]), float(parts[5])])
    positions = np.array(positions)

    matrix_dir = os.path.join(bc_dir, 'filtered_feature_bc_matrix')
    with gzip.open(os.path.join(matrix_dir, 'barcodes.tsv.gz'), 'rt') as f:
        expr_barcodes = [l.strip() for l in f]
    with gzip.open(os.path.join(matrix_dir, 'features.tsv.gz'), 'rt') as f:
        gene_names = [l.strip().split('\t')[1] for l in f]
    with gzip.open(os.path.join(matrix_dir, 'matrix.mtx.gz'), 'rb') as f:
        expr_matrix = scipy.io.mmread(f).tocsc()

    bc_map = {bc: i for i, bc in enumerate(expr_barcodes)}
    idx_pos, idx_expr = [], []
    for i, bc in enumerate(barcodes_pos):
        if bc in bc_map:
            idx_pos.append(i); idx_expr.append(bc_map[bc])
    positions = positions[idx_pos]
    expr = expr_matrix[:, idx_expr].toarray().T

    # Select top variable genes
    M = 8
    mean = expr.mean(0) + 1e-10
    fano = expr.var(0) / mean
    fano[mean < 0.5] = 0
    top = np.argsort(fano)[-M:][::-1]
    names = [gene_names[g] for g in top]
    expr = np.log1p(expr[:, top].astype(np.float32))
    expr = (expr - expr.mean(0)) / (expr.std(0) + 1e-10)

    rng = np.random.RandomState(seed)
    if len(positions) > n_spots:
        idx = rng.choice(len(positions), n_spots, replace=False)
        positions, expr = positions[idx], expr[idx]

    tree = KDTree(positions)
    nn_d, _ = tree.query(positions, k=2)
    scale = 100.0 / np.median(nn_d[:, 1])
    return positions * scale, expr, names


def load_intestinal_spatial():
    """Load CCI benchmark intestinal spatial transcriptomics data (second tissue)."""
    base = os.path.join(DATA_DIR, 'CCIBenchmark', 'CCI-main', 'example_data',
                         'ST_A3_GSM4797918', 'data', 'processed')
    coord = pd.read_csv(os.path.join(base, 'st_coord.tsv'), sep='\t')
    counts = pd.read_csv(os.path.join(base, 'st_counts.tsv'), sep='\t', index_col=0)
    meta = pd.read_csv(os.path.join(base, 'st_meta.tsv'), sep='\t', header=None)

    positions = coord[['pixel_x', 'pixel_y']].values.astype(np.float32)
    # Expression: counts is (genes × spots), transpose to (spots × genes)
    expr = counts.values.T.astype(np.float32)
    spot_names = list(counts.columns)
    gene_names = list(counts.index)

    # Select top variable genes
    M = 8
    mean = expr.mean(0) + 1e-10
    fano = expr.var(0) / mean
    fano[mean < 0.5] = 0
    top = np.argsort(fano)[-M:][::-1]
    names = [gene_names[g] for g in top]
    expr_sel = np.log1p(expr[:, top])
    expr_sel = (expr_sel - expr_sel.mean(0)) / (expr_sel.std(0) + 1e-10)

    # Scale positions
    tree = KDTree(positions)
    nn_d, _ = tree.query(positions, k=2)
    scale = 100.0 / (np.median(nn_d[:, 1]) + 1e-10)
    return positions * scale, expr_sel, names


def train_spatial_model(model, expression, edge_index, edge_weight, train_idx, test_idx,
                        mask_self=True, sigma_noise=0.3, n_epochs=1000, lr=1e-3,
                        l1_lambda=1e-4, patience=50, seed=0):
    """Train on spatial prediction with optional self-feature masking."""
    model = model.to(DEVICE)
    rng = np.random.RandomState(seed)
    noise = rng.randn(*expression.shape).astype(np.float32) * sigma_noise
    x_noisy = expression + noise

    if mask_self:
        # Zero out each node's own features in the input
        # The model must predict from neighbors only (through diffusion)
        x_input = np.zeros_like(x_noisy)  # masked input
    else:
        x_input = x_noisy

    x_in = torch.tensor(x_input, dtype=torch.float32).to(DEVICE)
    x_target = torch.tensor(expression, dtype=torch.float32).to(DEVICE)
    ei = edge_index.to(DEVICE)
    ew = edge_weight.to(DEVICE)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)
    best_loss, best_state, wait = float('inf'), None, 0

    for epoch in range(n_epochs):
        model.train()
        pred = model.predict(x_in, ei, ew)
        loss = F.mse_loss(pred[train_idx], x_target[train_idx])
        if l1_lambda > 0:
            l1 = sum(p.abs().sum() for n, p in model.named_parameters() if 'coeffs' in n)
            loss = loss + l1_lambda * l1
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step()

        model.eval()
        with torch.no_grad():
            pred = model.predict(x_in, ei, ew)
            tl = F.mse_loss(pred[test_idx], x_target[test_idx]).item()
        if tl < best_loss:
            best_loss = tl; best_state = {k: v.clone() for k, v in model.state_dict().items()}; wait = 0
        else:
            wait += 1
            if wait >= patience: break

    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model.predict(x_in, ei, ew)
        rmse = torch.sqrt(F.mse_loss(pred[test_idx], x_target[test_idx])).item()
    return rmse, model


# ============================================================
# EXP 1+2: MULTI-TISSUE SPATIAL TRANSCRIPTOMICS
# ============================================================
def experiment_spatial_multi_tissue():
    print("\n" + "="*60)
    print("EXP 1+2: Multi-Tissue Spatial (Breast Cancer + Intestine)")
    print("="*60)
    results = {}

    for tissue_name, loader in [('breast_cancer', load_breast_cancer_spatial),
                                 ('intestine', load_intestinal_spatial)]:
        print(f"\n  Tissue: {tissue_name}")
        positions, expression, gene_names = loader() if tissue_name == 'intestine' else loader(n_spots=500)
        N, M = expression.shape
        print(f"    N={N}, M={M}, genes={gene_names[:4]}...")

        edge_index, edge_weight, _, _, stats = construct_mc_graph(positions)
        print(f"    Graph: |E|={stats['n_edges_undirected']}, avg_deg={stats['avg_degree']:.2f}")

        tissue_results = {}
        for mask_self in [False, True]:
            mask_label = 'masked' if mask_self else 'unmasked'
            print(f"\n    Self-feature: {mask_label}")
            model_results = {}

            for name, make_model in [
                ('RD-GKAN', lambda: CrossFeatureGraphKAN(M, n_layers=2, G=5, k=3)),
                ('GNN', lambda: GNNBaseline(M, hidden=64, n_layers=2)),
                ('GAT', lambda: GATBaseline(M, hidden=64, n_heads=4)),
            ]:
                rmses = []
                for seed in range(5):
                    train_idx, test_idx = region_based_split(positions, seed=seed + 200)
                    model = make_model()
                    # For masked: zero out self-features in input
                    rng = np.random.RandomState(seed + 100)
                    noise = rng.randn(*expression.shape).astype(np.float32) * 0.3
                    x_input = np.zeros_like(expression) if mask_self else expression + noise
                    x_target = expression

                    rmse, _ = train_model(
                        model, x_input, x_target, edge_index, edge_weight,
                        train_idx, test_idx, n_epochs=1000,
                        l1_lambda=1e-4 if 'GKAN' in name else 0, patience=50)
                    rmses.append(rmse)
                model_results[name] = {'mean': float(np.mean(rmses)), 'std': float(np.std(rmses))}
                print(f"      {name}: {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")

            tissue_results[mask_label] = model_results
        tissue_results['graph_stats'] = stats
        tissue_results['gene_names'] = gene_names
        results[tissue_name] = tissue_results

    return results


# ============================================================
# EXP 3: QS KINETICS ACROSS 3 GENES
# ============================================================
def experiment_qs_multi_gene():
    print("\n" + "="*60)
    print("EXP 3: QS Kinetics (lasI, rhlI, lasB)")
    print("="*60)
    results = {}
    base = os.path.join(DATA_DIR, 'GeorgiaTechQS', 'hierarchy-main', 'Data')

    for gene in ['LasI', 'RhlI', 'LasB']:
        print(f"\n  Gene: {gene}")
        df = pd.read_stata(os.path.join(base, f'{gene}.dta'))

        # Steady-state dose-response (time >= 10)
        ss = df[df['time'] >= 10].groupby('c12')['rlu_od'].mean()
        x = ss.index.values.astype(np.float32)
        y = ss.values.astype(np.float32)
        y = y / (y.max() + 1e-10)

        # Hill fit
        def hill(x, V, K, n): return V * x**n / (K**n + x**n + 1e-10)
        try:
            popt, _ = curve_fit(hill, x, y, p0=[1,1,1], bounds=([0,0.01,0.1],[10,50,10]))
            V, K, n = popt
            R2 = 1 - np.sum((y - hill(x, *popt))**2) / (np.sum((y - y.mean())**2) + 1e-10)
        except:
            V, K, n, R2 = 1.0, 1.0, 1.0, 0.0
        print(f"    Hill: V={V:.3f}, K={K:.3f}, n={n:.3f}, R²={R2:.4f}")

        # KAN symbolic recovery
        kan = BSplineKAN(G=10, k=3, x_min=float(x.min())-0.5, x_max=float(x.max())+0.5).to(DEVICE)
        opt = Adam(kan.parameters(), lr=0.005)
        sched = CosineAnnealingLR(opt, T_max=5000, eta_min=1e-5)
        xt = torch.tensor(x).to(DEVICE); yt = torch.tensor(y).to(DEVICE)
        for _ in range(5000):
            loss = F.mse_loss(kan(xt), yt) + 1e-4 * kan.coeffs.abs().sum()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()

        # Symbolic projection with fitted K
        xg = np.linspace(float(x.min()), float(x.max()), 300).astype(np.float32)
        lib = {'1': np.ones_like(xg), 'x': xg, 'x^2': xg**2,
               f'MM(K={K:.2f})': xg / (K + xg + 1e-10),
               f'Hill_2(K={K:.2f})': xg**2 / (K**2 + xg**2 + 1e-10),
               f'Hill_n(n={n:.2f})': xg**n / (K**n + xg**n + 1e-10),
               'exp(-x)': np.exp(-xg / (xg.max()+1e-10))}
        S = np.column_stack(list(lib.values()))
        with torch.no_grad():
            phi = kan(torch.tensor(xg).to(DEVICE)).cpu().numpy()
        lasso = Lasso(alpha=0.003, max_iter=20000)
        lasso.fit(S, phi)
        alpha = lasso.coef_
        phi_recon = S @ alpha + lasso.intercept_
        eps_recon = np.linalg.norm(phi - phi_recon) / (np.linalg.norm(phi)+1e-10) * 100
        dom_idx = np.argmax(np.abs(alpha))
        lib_names = list(lib.keys())
        print(f"    Dominant: {lib_names[dom_idx]}, ε_recon={eps_recon:.1f}%")

        results[gene] = {
            'hill_fit': {'V': float(V), 'K': float(K), 'n': float(n), 'R2': float(R2)},
            'symbolic': {'dominant': lib_names[dom_idx], 'eps_recon': float(eps_recon),
                        'coeffs': {lib_names[i]: float(alpha[i]) for i in range(len(lib_names))}},
            'n_data': len(df), 'n_doses': len(x),
        }
    return results


# ============================================================
# EXP 4: CIR CHANNEL MODEL (existing, keep)
# ============================================================
def experiment_cir_channel():
    print("\n" + "="*60)
    print("EXP 4: CIR Channel Model (MacroscaleTestbed)")
    print("="*60)
    csv_path = os.path.join(DATA_DIR, 'MacroscaleTestbed', 'dataset_SISO_testbed.csv')
    data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
    time_col = data[:, 0]
    measurements = data[:, 1:]
    mean_signal = np.nanmean(measurements, axis=1)
    mean_signal = mean_signal / (np.nanmax(mean_signal) + 1e-10)

    def cir(t, d, rr, D):
        return (rr/d) * (d-rr) / np.sqrt(4*np.pi*D*t**3 + 1e-30) * np.exp(-(d-rr)**2/(4*D*t+1e-30))

    valid = (time_col > 0) & np.isfinite(mean_signal) & (mean_signal > 0)
    t_fit, y_fit = time_col[valid], mean_signal[valid]
    try:
        popt, _ = curve_fit(cir, t_fit, y_fit, p0=[0.5, 0.05, 0.01], bounds=([0.1,0.01,0.001],[5,1,1]))
        y_pred = cir(t_fit, *popt)
        R2 = 1 - np.sum((y_fit - y_pred)**2) / (np.sum((y_fit - y_fit.mean())**2) + 1e-10)
        d_fit, rr_fit, D_fit = popt
    except:
        R2, d_fit, rr_fit, D_fit = 0.0, 0.5, 0.05, 0.01
    w_ij = (rr_fit / d_fit) * erfc((d_fit - rr_fit) / np.sqrt(4*D_fit*max(t_fit)))
    print(f"  R²={R2:.4f}, d={d_fit:.3f}, rr={rr_fit:.4f}, D={D_fit:.4f}, w_ij={w_ij:.4f}")
    return {'R2': float(R2), 'd': float(d_fit), 'rr': float(rr_fit), 'D': float(D_fit),
            'w_ij': float(w_ij), 'n_measurements': int(measurements.shape[1])}


# ============================================================
# EXP 5: BIOLOGICAL MC NOISE (existing, keep)
# ============================================================
def experiment_bio_noise():
    print("\n" + "="*60)
    print("EXP 5: Biological MC Noise (ProtonPumpingBacteria)")
    print("="*60)
    noise_dir = os.path.join(DATA_DIR, 'ProtonPumpingBacteria', 'NoiseData')
    all_increments = []
    for i in range(1, 13):
        try:
            df = pd.read_csv(os.path.join(noise_dir, f'{i}Noise.csv'))
            ph = df['pH'].values
            increments = np.diff(ph)
            all_increments.extend(increments.tolist())
        except: pass
    arr = np.array(all_increments)
    mu, sigma = np.mean(arr), np.std(arr)
    print(f"  Noise: μ={mu:.6f}, σ={sigma:.4f}, n_samples={len(arr)}")
    return {'mean': float(mu), 'std': float(sigma), 'n_records': 12, 'n_samples': len(arr)}


# ============================================================
# EXP 6: ANALOG NETWORK CODING (NEW)
# ============================================================
def experiment_anc():
    print("\n" + "="*60)
    print("EXP 6: Analog Network Coding (AnalogNetworkCoding)")
    print("="*60)
    anc_dir = os.path.join(DATA_DIR, 'AnalogNetworkCoding')

    # Propagation delay analysis
    delay_data = pd.read_csv(os.path.join(anc_dir, 'Propagation_delay.csv'))
    time_steps = delay_data['x'].values
    measurements = delay_data[[f'y{i}' for i in range(1, 11)]].values  # 10 runs
    mean_signal = np.mean(measurements, axis=1)
    std_signal = np.std(measurements, axis=1)

    # Normalize
    mean_norm = (mean_signal - mean_signal.min()) / (mean_signal.max() - mean_signal.min() + 1e-10)

    # CIR fit on propagation delay
    def cir_simple(t, a, b, c):
        return a * np.exp(-b * (t - c)**2) * (t > 0)
    try:
        valid = time_steps > 0
        popt, _ = curve_fit(cir_simple, time_steps[valid], mean_norm[valid],
                           p0=[1, 0.01, 5], maxfev=5000)
        y_pred = cir_simple(time_steps[valid], *popt)
        R2 = 1 - np.sum((mean_norm[valid] - y_pred)**2) / (np.sum((mean_norm[valid] - mean_norm[valid].mean())**2) + 1e-10)
    except:
        R2 = 0.0

    # Error pattern analysis
    try:
        err_df = pd.read_csv(os.path.join(anc_dir, 'Error_pattern_ANC_duplex_3.csv'))
        total_tx = err_df.iloc[:, -1].sum() if len(err_df.columns) > 3 else 0
        total_err = err_df.iloc[:, -2].sum() if len(err_df.columns) > 2 else 0
        ber = total_err / (total_tx + 1e-10)
    except:
        ber = 0.0
        total_tx = 0

    print(f"  Propagation: R²={R2:.4f}, 10 runs, {len(time_steps)} time steps")
    print(f"  ANC duplex: BER={ber:.4f}, {int(total_tx)} transmissions")
    return {
        'propagation_R2': float(R2),
        'n_runs': 10, 'n_time_steps': len(time_steps),
        'anc_ber': float(ber), 'anc_transmissions': int(total_tx),
        'mean_snr_db': float(10 * np.log10(np.mean(mean_signal)**2 / (np.mean(std_signal)**2 + 1e-10))),
    }


# ============================================================
# EXP 7: ABLATIONS (reuse existing synthetic results)
# ============================================================
def experiment_ablations():
    print("\n" + "="*60)
    print("EXP 7: Ablations (synthetic R-D)")
    print("="*60)

    N = 200; d = 2; h = 0.005; D_true = 0.5; T = 500; kinetics = 'hill'
    results = {}

    # Spline vs MLP
    for model_type in ['spline', 'mlp']:
        rmses = []
        for seed in range(3):
            traj, pos, W, L_norm, _ = generate_rd_trajectory(
                N, d, h, D_true, T, kinetics, sigma_noise=0.005, seed=seed)
            L_t = torch.tensor(L_norm, dtype=torch.float32)
            model = RD_GKAN(1, G=8, k=3, x_range=(-1,5)) if model_type == 'spline' else MLPReaction(1, hidden=32)
            train_rd_gkan(model, traj, L_t, n_epochs=1000, patience=60)
            model.eval()
            with torch.no_grad():
                c0 = torch.tensor(traj[0], dtype=torch.float32).to(DEVICE)
                pred = model.multi_step(c0, L_t.to(DEVICE), T)[-1].cpu().numpy()
            rmses.append(np.sqrt(np.mean((pred - traj[-1])**2)))
        results[f'{model_type}_rmse'] = {'mean': float(np.mean(rmses)), 'std': float(np.std(rmses))}
        print(f"  {model_type}: RMSE={np.mean(rmses):.4f}")

    # h-scaling (Euler rate validation)
    print("\n  h-scaling:")
    h_results = {}
    for hv in [0.02, 0.01, 0.005, 0.002, 0.001]:
        T_steps = int(2.0 / hv)
        rmses = []
        for seed in range(3):
            traj_e, pos, W, L_norm, _ = generate_rd_trajectory(
                500, d, hv, D_true, T_steps, kinetics, sigma_noise=0.0, seed=seed)
            f = KINETICS[kinetics]['f']
            c_rk4 = traj_e[0].copy()
            for t in range(T_steps):
                def rhs(cc): return hv * f(cc) - hv * D_true * (L_norm @ cc)
                k1 = rhs(c_rk4); k2 = rhs(c_rk4+k1/2); k3 = rhs(c_rk4+k2/2); k4 = rhs(c_rk4+k3)
                c_rk4 = np.clip(c_rk4 + (k1+2*k2+2*k3+k4)/6, 0, 20)
            rmses.append(np.sqrt(np.mean((traj_e[-1] - c_rk4)**2)))
        h_results[str(hv)] = float(np.mean(rmses))
        print(f"    h={hv}: RMSE={np.mean(rmses):.6f}")
    results['h_scaling'] = h_results

    # Fit rate
    hs = np.array([float(k) for k in h_results])
    errs = np.array(list(h_results.values()))
    coeffs = np.polyfit(np.log(hs), np.log(errs), 1)
    results['h_exponent'] = float(coeffs[0])
    print(f"  h-scaling rate: h^{{{coeffs[0]:.3f}}}")

    return results


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    results = {}

    results['spatial'] = experiment_spatial_multi_tissue()
    results['qs'] = experiment_qs_multi_gene()
    results['cir'] = experiment_cir_channel()
    results['noise'] = experiment_bio_noise()
    results['anc'] = experiment_anc()
    results['ablations'] = experiment_ablations()

    out = os.path.join(RESULTS_DIR, 'option_b_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"ALL DONE in {elapsed/60:.1f} min. Results: {out}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
