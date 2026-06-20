#!/usr/bin/env python3
"""
============================================================================
Graph KAN v2: Improved Architecture + Rigorous Multi-Seed Experiments
============================================================================
Key improvements over v1:
  1. RESIDUAL Graph KAN: h^{l+1} = h^{l} + Psi(h^{l}) + sum w_ij Phi(h^{l}_j)
     This mirrors the reaction-diffusion physics (Eq. 7) where the update
     is identity + small correction.
  2. Proper training: cosine LR schedule, 500 epochs, patience=80
  3. Multi-seed evaluation: 5 seeds, report mean ± std
  4. Sample complexity: 10 seeds per T value
============================================================================
"""

import os, sys, json, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.special import erfc
from scipy.sparse import csr_matrix
from scipy.io import mmread
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gzip

warnings.filterwarnings('ignore')

BASE = '/home/dong/Workspace/BioComm'
DATA = os.path.join(BASE, 'data')
RESULTS = os.path.join(BASE, 'results')
os.makedirs(RESULTS, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================================
# IMPROVED KAN FUNCTION
# ============================================================================
class KANFunction(nn.Module):
    """Learnable univariate spline phi: R -> R using RBF basis."""
    def __init__(self, G=5, k=3, x_min=-2.0, x_max=2.0):
        super().__init__()
        self.G = G
        self.grid = nn.Parameter(torch.linspace(x_min, x_max, G + k),
                                 requires_grad=False)
        self.coeffs = nn.Parameter(torch.randn(G + k) * 0.05)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        dists = (x.unsqueeze(-1) - self.grid.unsqueeze(0)) ** 2
        sigma = (self.grid[-1] - self.grid[0]) / (2.0 * self.G)
        bases = torch.exp(-dists / (2 * sigma**2 + 1e-8))
        return self.scale * (bases * self.coeffs).sum(-1)

    def get_spline_values(self, x_grid):
        with torch.no_grad():
            return self.forward(x_grid).cpu().numpy()


# ============================================================================
# KEY FIX: RESIDUAL GRAPH KAN LAYER
# ============================================================================
class ResidualGraphKANLayer(nn.Module):
    """Graph KAN layer WITH residual connection:
       h_i^{l+1} = h_i^{l} + Psi(h_i^{l}) + sum_j w_ij * Phi(h_j^{l})

    This mirrors the reaction-diffusion physics (Eq. 7):
       c(t+1) = c(t) + tau*f(c(t)) + tau*D*sum w_ij*(c_j - c_i)

    The splines learn only the CORRECTION (tau*f), not the identity.
    """
    def __init__(self, features, G=5, k=3):
        super().__init__()
        self.features = features
        self.msg_kans = nn.ModuleList([KANFunction(G, k) for _ in range(features)])
        self.self_kans = nn.ModuleList([KANFunction(G, k) for _ in range(features)])
        # Learnable residual gate (initialized near 1 for strong skip connection)
        self.gate = nn.Parameter(torch.ones(1) * 0.9)

    def forward(self, h, edge_index, edge_weight):
        N, P = h.shape
        src, dst = edge_index

        # Message passing with KAN splines
        msg = torch.zeros_like(h)
        for p in range(P):
            m_p = self.msg_kans[p](h[src, p])
            weighted_m = m_p * edge_weight
            msg[:, p].scatter_add_(0, dst, weighted_m)

        # Self-update with KAN splines
        self_update = torch.zeros_like(h)
        for p in range(P):
            self_update[:, p] = self.self_kans[p](h[:, p])

        # RESIDUAL connection (the key improvement)
        return self.gate * h + (1 - self.gate) * (self_update + msg)


class ResidualGraphKAN(nn.Module):
    """Full Residual Graph KAN model."""
    def __init__(self, features, n_layers=2, G=5, k=3):
        super().__init__()
        self.layers = nn.ModuleList([
            ResidualGraphKANLayer(features, G, k) for _ in range(n_layers)
        ])

    def forward(self, h, edge_index, edge_weight):
        for layer in self.layers:
            h = layer(h, edge_index, edge_weight)
        return h


# ============================================================================
# BASELINES (same as before)
# ============================================================================
class GNNLayer(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.W = nn.Linear(in_f, out_f, bias=False)
        self.U = nn.Linear(in_f, out_f, bias=False)

    def forward(self, h, edge_index, edge_weight):
        src, dst = edge_index
        N = h.shape[0]
        agg = torch.zeros(N, h.shape[1], device=h.device)
        msg = self.U(h[src]) * edge_weight.unsqueeze(-1)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
        return torch.relu(self.W(h) + agg)


class GNNBaseline(nn.Module):
    def __init__(self, in_f, hid_f, out_f, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        dims = [in_f] + [hid_f]*(n_layers-1) + [out_f]
        for i in range(n_layers):
            self.layers.append(GNNLayer(dims[i], dims[i+1]))

    def forward(self, h, edge_index, edge_weight):
        for layer in self.layers:
            h = layer(h, edge_index, edge_weight)
        return h


class GATLayer(nn.Module):
    def __init__(self, in_f, out_f, heads=4):
        super().__init__()
        self.W = nn.Linear(in_f, out_f, bias=False)
        self.attn_src = nn.Linear(out_f, 1, bias=False)
        self.attn_dst = nn.Linear(out_f, 1, bias=False)
        self.out_f = out_f

    def forward(self, h, edge_index, edge_weight):
        src, dst = edge_index
        N = h.shape[0]
        Wh = self.W(h)
        e = self.attn_src(Wh[src]) + self.attn_dst(Wh[dst])
        e = torch.relu(e).squeeze(-1)
        # Softmax per destination node
        e_max = torch.zeros(N, device=h.device).scatter_reduce_(
            0, dst, e, reduce='amax', include_self=False)
        e_exp = torch.exp(e - e_max[dst]) * edge_weight
        e_sum = torch.zeros(N, device=h.device).scatter_add_(0, dst, e_exp)
        alpha = e_exp / (e_sum[dst] + 1e-8)
        agg = torch.zeros(N, self.out_f, device=h.device)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand(-1, self.out_f),
                         Wh[src] * alpha.unsqueeze(-1))
        return torch.relu(agg)


class GATBaseline(nn.Module):
    def __init__(self, in_f, hid_f, out_f, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        dims = [in_f] + [hid_f]*(n_layers-1) + [out_f]
        for i in range(n_layers):
            self.layers.append(GATLayer(dims[i], dims[i+1]))

    def forward(self, h, edge_index, edge_weight):
        for layer in self.layers:
            h = layer(h, edge_index, edge_weight)
        return h


# ============================================================================
# DATA LOADING (breast cancer spatial graph)
# ============================================================================
def load_breast_cancer_graph():
    bc_dir = os.path.join(DATA, '10XBreastCancer')
    pos_df = pd.read_csv(os.path.join(bc_dir, 'spatial', 'tissue_positions_list.csv'),
                         header=None,
                         names=['barcode', 'in_tissue', 'row', 'col', 'px_row', 'px_col'])
    pos_df = pos_df[pos_df['in_tissue'] == 1]

    coords = pos_df[['px_row', 'px_col']].values.astype(np.float32)
    nn_scale = NearestNeighbors(n_neighbors=2).fit(coords)
    nn_dists, _ = nn_scale.kneighbors(coords)
    median_nn_px = np.median(nn_dists[:, 1])
    scale = 100.0 / median_nn_px
    coords_um = coords * scale

    matrix_dir = os.path.join(bc_dir, 'filtered_feature_bc_matrix')
    with gzip.open(os.path.join(matrix_dir, 'barcodes.tsv.gz'), 'rt') as f:
        barcodes = [line.strip() for line in f]
    with gzip.open(os.path.join(matrix_dir, 'features.tsv.gz'), 'rt') as f:
        features = [line.strip().split('\t') for line in f]
    gene_names = [f[1] if len(f) > 1 else f[0] for f in features]
    mat = mmread(os.path.join(matrix_dir, 'matrix.mtx.gz')).tocsc()

    pos_barcode_to_idx = {b: i for i, b in enumerate(pos_df['barcode'].values)}
    matched_bc_col, matched_pos_row = [], []
    for col_i, b in enumerate(barcodes):
        if b in pos_barcode_to_idx:
            matched_bc_col.append(col_i)
            matched_pos_row.append(pos_barcode_to_idx[b])

    expr_sub = mat[:, matched_bc_col].toarray().T
    coords_sub = coords_um[matched_pos_row]

    M = 8
    gene_var = np.var(expr_sub, axis=0)
    gene_mean = np.mean(expr_sub, axis=0)
    fano = gene_var / (gene_mean + 1e-8)
    expressed = (expr_sub > 0).mean(axis=0) > 0.1
    fano[~expressed] = 0
    top_genes = np.argsort(fano)[-M:][::-1]

    X = np.log1p(expr_sub[:, top_genes].astype(np.float32))
    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True) + 1e-8
    X = (X - X_mean) / X_std
    selected_names = [gene_names[g] for g in top_genes]

    N_max = 500
    if X.shape[0] > N_max:
        idx = np.random.RandomState(42).choice(X.shape[0], N_max, replace=False)
        X = X[idx]
        coords_sub = coords_sub[idx]

    N = X.shape[0]

    # k-NN graph with edge weights from Eq. (2)
    r_r = 27.5
    D_free = 100.0
    tau = 60.0
    k_nn = 6
    nn_model = NearestNeighbors(n_neighbors=k_nn+1, metric='euclidean')
    nn_model.fit(coords_sub)
    dists_knn, indices_knn = nn_model.kneighbors(coords_sub)

    w_matrix = np.zeros((N, N))
    for i in range(N):
        for k_idx in range(1, k_nn+1):
            j = indices_knn[i, k_idx]
            d_ij = dists_knn[i, k_idx]
            if d_ij > r_r * 0.1:
                w = (r_r / max(d_ij, r_r)) * erfc(
                    max(d_ij - r_r, 0) / np.sqrt(4 * D_free * tau + 1e-8))
                w = min(w, 1.0)
                if w > 0.01:
                    w_matrix[i, j] = max(w_matrix[i, j], w)
                    w_matrix[j, i] = max(w_matrix[j, i], w)

    edges = np.array(np.nonzero(w_matrix))
    weights = w_matrix[edges[0], edges[1]]

    degree = w_matrix.sum(axis=1)
    L = np.diag(degree) - w_matrix
    eigenvalues = np.linalg.eigvalsh(L)
    lambda2 = eigenvalues[1] if len(eigenvalues) > 1 else 0

    print(f"  Graph: N={N}, M={M}, |E|={edges.shape[1]}, "
          f"avg_deg={edges.shape[1]/N:.1f}, lambda2={lambda2:.4f}")
    print(f"  Genes: {selected_names}")

    return X, coords_sub, edges, weights, L, eigenvalues, lambda2, selected_names


def generate_rd_dynamics(X_init, edges, weights, N, M, T=500,
                         tau=0.1, D_rate=0.05):
    """Hill-function RD dynamics on the real graph."""
    V_max, K_D, n_hill, beta = 0.5, 1.0, 2, 0.1
    def f_reaction(c):
        return V_max * c**n_hill / (K_D**n_hill + c**n_hill) - beta * c

    W = np.zeros((N, N))
    W[edges[0], edges[1]] = weights

    c = np.zeros((T, N, M), dtype=np.float32)
    c[0] = X_init[:N, :M]
    sigma_noise = 0.01

    for t in range(T - 1):
        reaction = f_reaction(c[t])
        diffusion = np.zeros_like(c[t])
        for i in range(N):
            neighbors = np.nonzero(W[i])[0]
            for j in neighbors:
                diffusion[i] += W[i, j] * (c[t, j] - c[t, i])
        noise = np.random.randn(N, M).astype(np.float32) * sigma_noise
        c[t+1] = c[t] + tau * reaction + tau * D_rate * diffusion + noise
        c[t+1] = np.clip(c[t+1], -5, 5)

    return c, f_reaction


# ============================================================================
# TRAINING FUNCTION (shared by all models)
# ============================================================================
def train_model(model, c_tensor, edge_index, edge_weight,
                T_train, T_val, epochs=500, lr=1e-3, l1_lambda=1e-4,
                patience=80, verbose=True):
    """Train with cosine LR, proper early stopping, L1 regularization."""
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/100)

    def spline_l1(model):
        l1 = 0
        for m in model.modules():
            if isinstance(m, KANFunction):
                l1 += m.coeffs.abs().sum()
        return l1_lambda * l1

    best_val = float('inf')
    best_state = None
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        # Use more samples per epoch (128 instead of 64)
        perm = torch.randperm(T_train)[:128]
        for t_idx in perm:
            h_pred = model(c_tensor[t_idx], edge_index, edge_weight)
            loss = F.mse_loss(h_pred, c_tensor[t_idx + 1])
            if l1_lambda > 0:
                loss = loss + spline_l1(model)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        train_losses.append(epoch_loss / len(perm))

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for t_idx in range(T_train, T_train + T_val):
                h_pred = model(c_tensor[t_idx], edge_index, edge_weight)
                val_loss += F.mse_loss(h_pred, c_tensor[t_idx+1]).item()
            val_loss /= T_val
            val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and epoch % 100 == 0:
            print(f"    Epoch {epoch}: train={train_losses[-1]:.6f}, "
                  f"val={val_loss:.6f}, lr={scheduler.get_last_lr()[0]:.6f}")

        if patience_counter > patience:
            if verbose:
                print(f"    Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return train_losses, val_losses


def evaluate_model(model, c_tensor, edge_index, edge_weight,
                   test_start, T_test, T_total):
    """Evaluate 1-step, 10-step, 50-step RMSE."""
    model.eval()
    results = {}
    with torch.no_grad():
        # 1-step
        rmse_1 = []
        for t in range(test_start, min(test_start + T_test, T_total - 1)):
            pred = model(c_tensor[t], edge_index, edge_weight)
            rmse_1.append(torch.sqrt(F.mse_loss(pred, c_tensor[t+1])).item())
        results['1step_rmse'] = float(np.mean(rmse_1))

        # Multi-step rollouts
        for horizon in [10, 50]:
            if test_start + horizon < T_total:
                h = c_tensor[test_start].clone()
                for step in range(horizon):
                    h = model(h, edge_index, edge_weight)
                target = c_tensor[test_start + horizon]
                results[f'{horizon}step_rmse'] = float(
                    torch.sqrt(F.mse_loss(h, target)).item())
            else:
                results[f'{horizon}step_rmse'] = float('nan')

    return results


# ============================================================================
# MAIN EXPERIMENT: MULTI-SEED EVALUATION
# ============================================================================
def main():
    print("="*70)
    print("GRAPH KAN v2: IMPROVED ARCHITECTURE + MULTI-SEED EXPERIMENTS")
    print("="*70)
    start_time = time.time()

    # Load data
    print("\n  Loading breast cancer spatial graph...")
    np.random.seed(42)
    X, coords, edges, weights, L, eigenvalues, lambda2, gene_names = \
        load_breast_cancer_graph()

    N, M = X.shape
    T_total = 500
    T_train, T_val = 350, 75
    T_test = T_total - T_train - T_val - 1
    test_start = T_train + T_val

    # Generate dynamics
    print("  Generating R-D dynamics (Hill kinetics) on real graph...")
    c_data, f_true = generate_rd_dynamics(X, edges, weights, N, M,
                                          T=T_total, tau=0.1, D_rate=0.05)

    edge_index = torch.tensor(edges, dtype=torch.long).to(DEVICE)
    edge_weight = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    c_tensor = torch.tensor(c_data, dtype=torch.float32).to(DEVICE)

    # ================================================================
    # MULTI-SEED EXPERIMENT
    # ================================================================
    N_SEEDS = 5
    all_results = {name: [] for name in
                   ['Residual Graph KAN', 'GNN', 'GAT']}

    for seed in range(N_SEEDS):
        print(f"\n{'='*50}")
        print(f"  SEED {seed+1}/{N_SEEDS}")
        print(f"{'='*50}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        # --- Residual Graph KAN ---
        print("  Training Residual Graph KAN...")
        model_kan = ResidualGraphKAN(M, n_layers=2, G=5, k=3).to(DEVICE)
        train_model(model_kan, c_tensor, edge_index, edge_weight,
                    T_train, T_val, epochs=500, lr=2e-3, l1_lambda=1e-4,
                    patience=80, verbose=(seed == 0))
        r = evaluate_model(model_kan, c_tensor, edge_index, edge_weight,
                          test_start, T_test, T_total)
        all_results['Residual Graph KAN'].append(r)
        print(f"    1-step RMSE: {r['1step_rmse']:.4f}, "
              f"50-step: {r['50step_rmse']:.4f}")

        # --- GNN ---
        print("  Training GNN...")
        model_gnn = GNNBaseline(M, M, M, n_layers=2).to(DEVICE)
        train_model(model_gnn, c_tensor, edge_index, edge_weight,
                    T_train, T_val, epochs=500, lr=1e-3, l1_lambda=0,
                    patience=80, verbose=False)
        r = evaluate_model(model_gnn, c_tensor, edge_index, edge_weight,
                          test_start, T_test, T_total)
        all_results['GNN'].append(r)
        print(f"    1-step RMSE: {r['1step_rmse']:.4f}, "
              f"50-step: {r['50step_rmse']:.4f}")

        # --- GAT ---
        print("  Training GAT...")
        model_gat = GATBaseline(M, M, M, n_layers=2).to(DEVICE)
        train_model(model_gat, c_tensor, edge_index, edge_weight,
                    T_train, T_val, epochs=500, lr=1e-3, l1_lambda=0,
                    patience=80, verbose=False)
        r = evaluate_model(model_gat, c_tensor, edge_index, edge_weight,
                          test_start, T_test, T_total)
        all_results['GAT'].append(r)
        print(f"    1-step RMSE: {r['1step_rmse']:.4f}, "
              f"50-step: {r['50step_rmse']:.4f}")

    # ================================================================
    # AGGREGATE RESULTS
    # ================================================================
    print("\n" + "="*70)
    print("AGGREGATED RESULTS (mean ± std over 5 seeds)")
    print("="*70)

    summary = {}
    for name in all_results:
        metrics = {}
        for key in ['1step_rmse', '10step_rmse', '50step_rmse']:
            vals = [r[key] for r in all_results[name] if not np.isnan(r[key])]
            metrics[key] = {'mean': float(np.mean(vals)),
                            'std': float(np.std(vals))}
        summary[name] = metrics
        print(f"\n  {name}:")
        for key in metrics:
            m, s = metrics[key]['mean'], metrics[key]['std']
            print(f"    {key}: {m:.4f} ± {s:.4f}")

    # ================================================================
    # SYMBOLIC RECOVERY (on best seed)
    # ================================================================
    print("\n" + "="*70)
    print("SYMBOLIC RECOVERY")
    print("="*70)

    torch.manual_seed(0)
    model_kan_best = ResidualGraphKAN(M, n_layers=2, G=5, k=3).to(DEVICE)
    train_model(model_kan_best, c_tensor, edge_index, edge_weight,
                T_train, T_val, epochs=500, lr=2e-3, l1_lambda=1e-4,
                patience=80, verbose=True)

    # Extract spline
    phi = model_kan_best.layers[0].self_kans[0]
    x_grid = torch.linspace(-2, 2, 200).to(DEVICE)
    phi_vals = phi.get_spline_values(x_grid)
    x_np = x_grid.cpu().numpy()

    # Symbolic library
    library = {
        '1': np.ones_like(x_np),
        'x': x_np,
        'x^2': x_np**2,
        'x^3': x_np**3,
        'x/(1+|x|)': np.abs(x_np)/(1+np.abs(x_np)),
        'x^2/(1+x^2)': x_np**2/(1+x_np**2),
        'exp(-|x|)': np.exp(-np.abs(x_np)),
        'ln(1+|x|)': np.log(1+np.abs(x_np)),
    }
    S_matrix = np.column_stack(list(library.values()))
    lib_names = list(library.keys())

    lasso = Lasso(alpha=0.01, max_iter=10000, fit_intercept=True)
    lasso.fit(S_matrix, phi_vals)
    alpha_hat = lasso.coef_
    phi_symbolic = S_matrix @ alpha_hat + lasso.intercept_

    residual = np.sum(np.abs(phi_vals - phi_symbolic))
    total = np.sum(np.abs(phi_vals)) + 1e-8
    eps_sym = residual / total * 100

    dominant_idx = np.argmax(np.abs(alpha_hat))
    print(f"  Symbolic accuracy: eps_sym = {eps_sym:.2f}%")
    print(f"  Dominant kinetics: '{lib_names[dominant_idx]}' "
          f"(alpha = {alpha_hat[dominant_idx]:.4f})")
    for i in np.argsort(np.abs(alpha_hat))[::-1][:3]:
        if np.abs(alpha_hat[i]) > 0.01:
            print(f"    {lib_names[i]}: {alpha_hat[i]:.4f}")

    # ================================================================
    # STABILITY CERTIFICATE
    # ================================================================
    print("\n" + "="*70)
    print("STABILITY CERTIFICATE (Theorem 2)")
    print("="*70)

    model_kan_best.eval()
    c_mean = c_tensor[:T_train].mean(dim=0).mean(dim=0)
    c_mean_expanded = c_mean.unsqueeze(0).repeat(N, 1)
    eps_fd = 1e-4

    J_f = np.zeros((M, M))
    with torch.no_grad():
        f0 = model_kan_best(c_mean_expanded, edge_index, edge_weight)[0].cpu().numpy()
        for m in range(M):
            c_pert = c_mean_expanded.clone()
            c_pert[:, m] += eps_fd
            f_pert = model_kan_best(c_pert, edge_index, edge_weight)[0].cpu().numpy()
            J_f[:, m] = (f_pert - f0) / eps_fd

    L_np = L[:N, :N]
    I_N, I_M = np.eye(N), np.eye(M)
    A = np.kron(I_N, I_M) + 0.1 * np.kron(I_N, J_f) - 0.1 * 0.05 * np.kron(L_np, I_M)
    spectral_radius = np.max(np.abs(np.linalg.eigvals(A)))
    print(f"  rho(A) = {spectral_radius:.6f}")
    print(f"  Stability: {'PASS (contractive)' if spectral_radius < 1 else 'Non-contractive'}")

    # ================================================================
    # IT BOUND (Theorem 3)
    # ================================================================
    print("\n" + "="*70)
    print("IT BOUND (Theorem 3)")
    print("="*70)

    n_bar, X_bar = 100, 10
    w_mean = np.mean(weights)
    signal_power = n_bar**2 * w_mean**2 * X_bar
    noise_power = n_bar * w_mean + 0.1
    C_bound = 0.5 * np.log2(1 + signal_power / noise_power)

    model_kan_best.eval()
    with torch.no_grad():
        pred = model_kan_best(c_tensor[test_start], edge_index, edge_weight)
        mse_pred = F.mse_loss(pred, c_tensor[test_start+1]).item()
        signal_var = c_tensor[test_start+1].var().item()
        snr_pred = signal_var / (mse_pred + 1e-8)
        MI_achieved = 0.5 * np.log2(1 + snr_pred)

    gap_dB = 10 * np.log10(max(C_bound / (MI_achieved + 1e-8), 1))
    print(f"  C_bound = {C_bound:.3f} bits/use")
    print(f"  MI_achieved = {MI_achieved:.3f} bits/use")
    print(f"  Gap = {gap_dB:.2f} dB")

    # ================================================================
    # SAMPLE COMPLEXITY (Theorem 4) with 10 seeds per T
    # ================================================================
    print("\n" + "="*70)
    print("SAMPLE COMPLEXITY (Theorem 4) - 10 seeds per T")
    print("="*70)

    T_values = [50, 100, 150, 200, 300, 400]
    sample_results = {}
    for T_sub in T_values:
        accs = []
        for sc_seed in range(10):
            torch.manual_seed(sc_seed * 100 + T_sub)
            model_tmp = ResidualGraphKAN(M, n_layers=2, G=5, k=3).to(DEVICE)
            opt_tmp = Adam(model_tmp.parameters(), lr=3e-3)
            for ep in range(150):
                model_tmp.train()
                perm = torch.randperm(min(T_sub, T_total-1))[:64]
                for t_idx in perm:
                    pred = model_tmp(c_tensor[t_idx], edge_index, edge_weight)
                    loss = F.mse_loss(pred, c_tensor[t_idx+1])
                    l1 = sum(m.coeffs.abs().sum() for m in model_tmp.modules()
                             if isinstance(m, KANFunction))
                    loss = loss + 1e-4 * l1
                    opt_tmp.zero_grad(); loss.backward(); opt_tmp.step()

            model_tmp.eval()
            phi_tmp = model_tmp.layers[0].self_kans[0]
            phi_v = phi_tmp.get_spline_values(x_grid)
            lasso_tmp = Lasso(alpha=0.01, max_iter=5000, fit_intercept=True)
            lasso_tmp.fit(S_matrix, phi_v)
            phi_sym = S_matrix @ lasso_tmp.coef_ + lasso_tmp.intercept_
            res = np.sum(np.abs(phi_v - phi_sym))
            tot = np.sum(np.abs(phi_v)) + 1e-8
            accs.append(res / tot * 100)

        mean_eps = float(np.mean(accs))
        std_eps = float(np.std(accs))
        sample_results[T_sub] = {'mean': mean_eps, 'std': std_eps}
        print(f"  T={T_sub}: eps_sym = {mean_eps:.2f} ± {std_eps:.2f}%")

    # ================================================================
    # SAVE ALL RESULTS
    # ================================================================
    elapsed = time.time() - start_time
    final_results = {
        'multi_seed_comparison': summary,
        'symbolic_accuracy_pct': float(eps_sym),
        'dominant_kinetics': lib_names[dominant_idx],
        'spectral_radius': float(spectral_radius),
        'IT_bound_bits': float(C_bound),
        'MI_achieved_bits': float(MI_achieved),
        'IT_gap_dB': float(gap_dB),
        'sample_complexity': sample_results,
        'elapsed_minutes': elapsed / 60,
    }

    with open(os.path.join(RESULTS, 'exp_v2_results.json'), 'w') as f:
        json.dump(final_results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"COMPLETE in {elapsed/60:.1f} minutes")
    print(f"Results saved to {RESULTS}/exp_v2_results.json")
    print(f"{'='*70}")

    return final_results


if __name__ == '__main__':
    main()
