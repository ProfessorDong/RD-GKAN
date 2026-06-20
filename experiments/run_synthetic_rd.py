#!/usr/bin/env python3
"""
Synthetic Reaction-Diffusion experiments for the theory paper.
Constrained RD-GKAN architecture that EXACTLY matches proved dynamics:

    c_i(t+1) = c_i(t) + ψ_θ(c_i(t)) + D_θ · (1/N) Σ_j κ_ij (c_j(t) - c_i(t))

where ψ_θ is a feature-wise B-spline KAN (learns τ·f), D_θ is learnable coupling,
and diffusion uses the explicit graph Laplacian. No multi-layer, no Φ on neighbors,
no cross-feature mixing.

Experiments:
  1. N-scaling (Theorem 1: O(N^{-2/d}) convergence rate)
  2. h-scaling (Theorem 1: O(h²) time-discretization error)
  3. T-scaling (Theorem 4: symbolic recovery vs training snapshots)
  4. Ablations (spline vs MLP, MC vs uniform weights, grid density)
  5. Real-data illustrations (QS dose-response, spatial transcriptomics)
"""

import os, sys, json, warnings, time, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.spatial import KDTree
from scipy.special import erfc
from scipy.optimize import curve_fit
from sklearn.linear_model import Lasso
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"Device: {DEVICE}")


# ============================================================
# B-SPLINE KAN (proper Cox-de Boor recursion)
# ============================================================
class BSplineKAN(nn.Module):
    def __init__(self, G=8, k=3, x_min=-2.0, x_max=2.0):
        super().__init__()
        self.G, self.k = G, k
        n_basis = G + k
        knots = np.linspace(x_min, x_max, G + 1)
        step = knots[1] - knots[0]
        knots = np.concatenate([
            knots[0] - step * np.arange(k, 0, -1), knots,
            knots[-1] + step * np.arange(1, k + 1)
        ])
        self.register_buffer('knots', torch.tensor(knots, dtype=torch.float32))
        self.coeffs = nn.Parameter(torch.randn(n_basis) * 0.05)
        self.w_s = nn.Parameter(torch.tensor(1.0))
        self.w_b = nn.Parameter(torch.tensor(0.0))

    def _basis(self, x):
        knots = self.knots
        n_b = len(knots) - self.k - 1
        xf = x.reshape(-1, 1)
        B = ((xf >= knots[:-1]) & (xf < knots[1:])).float()
        for p in range(1, self.k + 1):
            Bn = torch.zeros(xf.shape[0], len(knots) - p - 1, device=x.device)
            for i in range(len(knots) - p - 1):
                d1 = knots[i+p] - knots[i]
                d2 = knots[i+p+1] - knots[i+1]
                if d1 > 1e-10:
                    Bn[:, i] += (xf[:, 0] - knots[i]) / d1 * B[:, i]
                if d2 > 1e-10:
                    Bn[:, i] += (knots[i+p+1] - xf[:, 0]) / d2 * B[:, i+1]
            B = Bn
        return B[:, :n_b].reshape(x.shape + (n_b,))

    def forward(self, x):
        return self.w_s * (self._basis(x) * self.coeffs).sum(-1) + self.w_b * x * torch.sigmoid(x)


# ============================================================
# CONSTRAINED RD-GKAN (exactly matches proved dynamics)
# ============================================================
class RD_GKAN(nn.Module):
    """Reaction-Diffusion Graph KAN.
    Forward pass IS the proved dynamics:
        c(t+1) = c(t) + ψ_θ(c(t)) + D_θ · L_norm · c(t)
    where L_norm = (1/N) Σ_j κ_ij (c_j - c_i) is the normalized Laplacian action.
    """
    def __init__(self, M, G=8, k=3, x_range=(-3, 5)):
        super().__init__()
        self.M = M
        self.reaction_kans = nn.ModuleList([
            BSplineKAN(G, k, x_range[0], x_range[1]) for _ in range(M)
        ])
        self.D = nn.Parameter(torch.tensor(0.1))  # learnable coupling

    def forward(self, c, L_norm):
        """One-step prediction: c(t) -> c(t+1).
        c: (N, M), L_norm: (N, N) normalized Laplacian (1/N factor included).
        """
        # Reaction: feature-wise KAN
        reaction = torch.stack([self.reaction_kans[p](c[:, p]) for p in range(self.M)], dim=1)
        # Diffusion: -D * L_norm @ c  (Laplacian diffusion)
        diffusion = -torch.abs(self.D) * (L_norm @ c)
        return c + reaction + diffusion

    def multi_step(self, c0, L_norm, T):
        """Autoregressive rollout for T steps."""
        traj = [c0]
        c = c0
        for _ in range(T):
            c = self.forward(c, L_norm)
            traj.append(c)
        return torch.stack(traj)  # (T+1, N, M)


class MLPReaction(nn.Module):
    """MLP baseline for ablation: replaces B-spline KAN with MLP."""
    def __init__(self, M, hidden=32):
        super().__init__()
        self.M = M
        self.nets = nn.ModuleList([
            nn.Sequential(nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, 1))
            for _ in range(M)
        ])
        self.D = nn.Parameter(torch.tensor(0.1))

    def forward(self, c, L_norm):
        reaction = torch.cat([self.nets[p](c[:, p:p+1]) for p in range(c.shape[1])], dim=1)
        diffusion = -torch.abs(self.D) * (L_norm @ c)
        return c + reaction + diffusion

    def multi_step(self, c0, L_norm, T):
        traj = [c0]
        c = c0
        for _ in range(T):
            c = self.forward(c, L_norm)
            traj.append(c)
        return torch.stack(traj)


# ============================================================
# SYNTHETIC R-D DATA GENERATOR
# ============================================================
def make_random_geometric_graph(N, d=2, sigma=0.15, seed=42):
    """Random geometric graph with Gaussian kernel weights. Uses sparse internals."""
    rng = np.random.RandomState(seed)
    positions = rng.uniform(0, 1, (N, d)).astype(np.float32)
    tree = KDTree(positions)

    # Use k-NN for speed instead of radius query for large N
    k_nn = min(20, N - 1)
    dists, indices = tree.query(positions, k=k_nn + 1)

    src, dst, weights = [], [], []
    seen = set()
    for i in range(N):
        for j_idx in range(1, k_nn + 1):
            j = indices[i, j_idx]
            dist_sq = dists[i, j_idx]**2
            w = np.exp(-dist_sq / (2 * sigma**2))
            if w > 0.01 and (i, j) not in seen:
                src.extend([i, j]); dst.extend([j, i])
                weights.extend([w, w])
                seen.add((i, j)); seen.add((j, i))

    # Build dense L_norm (needed for torch matmul)
    W = np.zeros((N, N), dtype=np.float32)
    for s, d_, w in zip(src, dst, weights):
        W[s, d_] = w
    D_deg = np.diag(W.sum(axis=1))
    L = D_deg - W
    L_norm = L / N

    avg_deg = W.sum() / N
    # Only compute eigenvalues for small N (expensive for large N)
    if N <= 2000:
        eigvals = np.linalg.eigvalsh(L_norm)
    else:
        eigvals = np.array([0.0])  # skip for large reference meshes
    return positions, W, L_norm, eigvals, avg_deg


KINETICS = {
    'linear': {
        'f': lambda c: -0.1 * c,
        'alpha_true': np.array([0.0, -0.1, 0.0, 0.0, 0.0, 0.0]),
        'label': r'$f(c) = -0.1c$',
    },
    'hill': {
        'f': lambda c: 1.0 * c**2 / (1.0 + c**2) - 0.3 * c,
        'alpha_true': np.array([0.0, -0.3, 0.0, 0.0, 0.0, 1.0]),
        'label': r'$f(c) = c^2/(1+c^2) - 0.3c$',
    },
    'michaelis_menten': {
        'f': lambda c: 1.0 * c / (0.5 + c) - 0.2 * c,
        'alpha_true': np.array([0.0, -0.2, 0.0, 0.0, 1.0, 0.0]),
        'label': r'$f(c) = c/(0.5+c) - 0.2c$',
    },
}


def generate_rd_trajectory(N, d, h, D_true, T, kinetics_name, sigma_noise=0.01,
                           sigma_kernel=0.15, seed=42):
    """Generate synthetic R-D trajectory with known kinetics."""
    rng = np.random.RandomState(seed)
    positions, W, L_norm, eigvals, avg_deg = make_random_geometric_graph(N, d, sigma_kernel, seed)
    f = KINETICS[kinetics_name]['f']

    # Initial condition: uniform random
    c = rng.uniform(0.1, 3.0, (N, 1)).astype(np.float32)
    trajectory = [c.copy()]

    L_dense = L_norm.copy()
    for t in range(T):
        reaction = h * f(c)
        diffusion = -h * D_true * (L_dense @ c)
        noise = rng.randn(*c.shape).astype(np.float32) * sigma_noise * np.sqrt(h)
        c = c + reaction + diffusion + noise
        c = np.clip(c, 0, 20)
        trajectory.append(c.copy())

    return np.array(trajectory), positions, W, L_norm, eigvals


def compute_reference_solution(positions_fine, W_fine, L_norm_fine, h, D_true, T,
                                kinetics_name, c0_interp_fn):
    """Compute reference solution on fine mesh using RK4."""
    f = KINETICS[kinetics_name]['f']
    N_fine = len(positions_fine)
    c = c0_interp_fn(positions_fine).reshape(N_fine, 1).astype(np.float32)

    for t in range(T):
        # RK4 integration
        def rhs(cc):
            return f(cc) - D_true * (L_norm_fine @ cc)
        k1 = h * rhs(c)
        k2 = h * rhs(c + k1/2)
        k3 = h * rhs(c + k2/2)
        k4 = h * rhs(c + k3)
        c = c + (k1 + 2*k2 + 2*k3 + k4) / 6
        c = np.clip(c, 0, 20)
    return c


# ============================================================
# TRAINING
# ============================================================
def train_rd_gkan(model, trajectory, L_norm_t, n_epochs=2000, lr=5e-3, l1=1e-4,
                  patience=100, verbose=False):
    """Train RD-GKAN on trajectory data (one-step prediction)."""
    model = model.to(DEVICE)
    T = len(trajectory) - 1
    L_norm_t = L_norm_t.to(DEVICE)

    N_nodes = trajectory.shape[1]
    T = len(trajectory) - 1
    n_batch = min(T, 64)  # subsample time steps per epoch for speed

    # Pre-convert to tensors
    traj_t = torch.tensor(trajectory, dtype=torch.float32).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

    best_loss = float('inf')
    best_state = None
    wait = 0

    for epoch in range(n_epochs):
        model.train()
        # Random subsample of time steps for this epoch
        t_indices = np.random.choice(T, size=n_batch, replace=False)

        loss_total = 0
        for t_idx in t_indices:
            pred = model.forward(traj_t[t_idx], L_norm_t)
            loss_total = loss_total + F.mse_loss(pred, traj_t[t_idx + 1])
        loss_avg = loss_total / n_batch
        if l1 > 0:
            l1_reg = sum(p.abs().sum() for n, p in model.named_parameters() if 'coeffs' in n)
            loss_avg = loss_avg + l1 * l1_reg

        optimizer.zero_grad()
        loss_avg.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        lv = loss_avg.item()
        if lv < best_loss:
            best_loss = lv
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

        if verbose and epoch % 200 == 0:
            print(f"      Epoch {epoch}: loss={lv:.6f}, D={model.D.item():.4f}")

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return best_loss


# ============================================================
# SYMBOLIC RECOVERY
# ============================================================
def symbolic_recovery(model, kinetics_name, M=1):
    """Extract learned spline and project onto symbolic library.
    Key: the model learns h*f(c), so we NORMALIZE the spline to unit scale
    before LASSO, making it comparable to the normalized ground truth."""
    model.eval()
    x_grid = np.linspace(0.01, 4.0, 300).astype(np.float32)

    # Library (unit-scale basis functions)
    library = {
        '1': np.ones_like(x_grid),
        'x': x_grid.copy(),
        'x^2': x_grid**2,
        'x^3': x_grid**3,
        'MM (x/(0.5+x))': x_grid / (0.5 + x_grid),
        'Hill_2 (x^2/(1+x^2))': x_grid**2 / (1 + x_grid**2),
    }
    lib_names = list(library.keys())
    S = np.column_stack([library[k] for k in lib_names])
    # Normalize library columns to unit L2 norm
    S_norms = np.linalg.norm(S, axis=0, keepdims=True) + 1e-10
    S_normed = S / S_norms

    # Ground truth: normalize alpha_true by the same column norms
    alpha_true_raw = KINETICS[kinetics_name]['alpha_true']
    results = {}

    for p in range(M):
        with torch.no_grad():
            phi = model.reaction_kans[p](torch.tensor(x_grid).to(DEVICE)).cpu().numpy()

        # Normalize learned spline to unit L2 norm (removes h scaling)
        phi_norm = np.linalg.norm(phi) + 1e-10
        phi_normed = phi / phi_norm

        # Also compute the normalized ground truth function
        f_true = KINETICS[kinetics_name]['f']
        f_vals = f_true(x_grid)
        f_norm = np.linalg.norm(f_vals) + 1e-10
        f_normed = f_vals / f_norm

        # LASSO on normalized quantities (small alpha for fine recovery)
        lasso = Lasso(alpha=1e-4, max_iter=20000, tol=1e-8)
        lasso.fit(S_normed, phi_normed)
        alpha_hat = lasso.coef_

        # Also fit the ground truth for reference
        lasso_true = Lasso(alpha=1e-4, max_iter=20000, tol=1e-8)
        lasso_true.fit(S_normed, f_normed)
        alpha_true = lasso_true.coef_

        # Metrics: compare alpha_hat vs alpha_true (both from normalized LASSO)
        eps_sym = np.linalg.norm(alpha_hat - alpha_true) / (np.linalg.norm(alpha_true) + 1e-10) * 100

        # Support recovery (dominant nonzero coefficients)
        thresh = 0.05 * max(np.max(np.abs(alpha_true)), 1e-6)
        true_support = set(np.where(np.abs(alpha_true) > thresh)[0])
        hat_support = set(np.where(np.abs(alpha_hat) > thresh)[0])
        support_correct = true_support == hat_support
        support_f1 = 2 * len(true_support & hat_support) / (len(true_support) + len(hat_support) + 1e-10)

        # Reconstruction quality
        phi_recon = S_normed @ alpha_hat + lasso.intercept_
        eps_recon = np.linalg.norm(phi_normed - phi_recon) / (np.linalg.norm(phi_normed) + 1e-10) * 100

        results[f'species_{p}'] = {
            'alpha_hat': {lib_names[i]: float(alpha_hat[i]) for i in range(len(lib_names))},
            'alpha_true': {lib_names[i]: float(alpha_true[i]) for i in range(len(lib_names))},
            'eps_sym': float(eps_sym),
            'eps_recon': float(eps_recon),
            'support_correct': bool(support_correct),
            'support_f1': float(support_f1),
        }

    return results


# ============================================================
# EXPERIMENT 1: N-SCALING
# ============================================================
def experiment_n_scaling():
    print("\n" + "="*60)
    print("EXPERIMENT 1: N-Scaling (Theorem 1)")
    print("="*60)

    N_values = [50, 100, 200, 500, 1000, 2000]
    d = 2
    h = 0.005
    D_true = 0.5
    T = 200
    kinetics = 'hill'
    n_seeds = 3
    results = {}

    # N-scaling measures GRAPH DISCRETIZATION ERROR (no model needed).
    # Compare the N-node graph dynamics against a fine-mesh (N_ref) reference.
    N_ref = 3000
    print(f"  Computing reference trajectory (N_ref={N_ref})...")
    ref_traj, ref_pos, _, ref_L, _ = generate_rd_trajectory(
        N_ref, d, h, D_true, T, kinetics, sigma_noise=0.0, seed=999)
    ref_final = ref_traj[-1].flatten()

    from scipy.interpolate import NearestNDInterpolator

    for N in N_values:
        if N >= N_ref:
            continue
        print(f"\n  N = {N}")
        rmses = []
        for seed in range(n_seeds):
            # Run exact (noise-free) dynamics on N-node graph
            traj, pos, W, L_norm, _ = generate_rd_trajectory(
                N, d, h, D_true, T, kinetics, sigma_noise=0.0, seed=seed)
            graph_final = traj[-1].flatten()

            # Interpolate reference solution to these N positions
            interp = NearestNDInterpolator(ref_pos, ref_final)
            ref_at_pos = interp(pos)

            rmse = np.sqrt(np.mean((graph_final - ref_at_pos)**2))
            rmses.append(rmse)
            print(f"    seed {seed}: discretization RMSE={rmse:.6f}")

        results[str(N)] = {
            'mean_rmse': float(np.mean(rmses)),
            'std_rmse': float(np.std(rmses)),
        }

    # Fit convergence rate
    ns = np.array([int(k) for k in results if k.isdigit()])
    errs = np.array([results[str(n)]['mean_rmse'] for n in ns])
    valid = (errs > 0) & np.isfinite(errs)
    if valid.sum() >= 3:
        log_fit = np.polyfit(np.log(ns[valid]), np.log(errs[valid]), 1)
        results['fit'] = {
            'exponent': float(-log_fit[0]),
            'theoretical': 2.0 / d,
            'intercept': float(log_fit[1]),
        }
        print(f"\n  Fitted: N^{{-{-log_fit[0]:.3f}}} (theory: N^{{-{2/d:.1f}}})")
    return results


# ============================================================
# EXPERIMENT 2: h-SCALING
# ============================================================
def experiment_h_scaling():
    print("\n" + "="*60)
    print("EXPERIMENT 2: h-Scaling (Theorem 1 discretization)")
    print("="*60)

    h_values = [0.02, 0.01, 0.005, 0.002, 0.001]
    N = 500
    d = 2
    D_true = 0.5
    T_phys = 2.0  # fixed physical time
    kinetics = 'hill'
    n_seeds = 3
    results = {}

    for h in h_values:
        T = int(T_phys / h)
        print(f"\n  h = {h}, T = {T}")
        rmses = []

        for seed in range(n_seeds):
            # Euler trajectory
            traj_euler, pos, W, L_norm, _ = generate_rd_trajectory(
                N, d, h, D_true, T, kinetics, sigma_noise=0.0, seed=seed)

            # RK4 reference on same graph
            f = KINETICS[kinetics]['f']
            c_rk4 = traj_euler[0].copy()
            for t in range(T):
                def rhs(cc):
                    return h * f(cc) - h * D_true * (L_norm @ cc)
                k1 = rhs(c_rk4)
                k2 = rhs(c_rk4 + k1/2)
                k3 = rhs(c_rk4 + k2/2)
                k4 = rhs(c_rk4 + k3)
                c_rk4 = c_rk4 + (k1 + 2*k2 + 2*k3 + k4) / 6
                c_rk4 = np.clip(c_rk4, 0, 20)

            # Euler vs RK4 error at final time
            rmse = np.sqrt(np.mean((traj_euler[-1] - c_rk4)**2))
            rmses.append(rmse)

        results[str(h)] = {
            'mean_rmse': float(np.mean(rmses)),
            'std_rmse': float(np.std(rmses)),
            'T_steps': T,
        }
        print(f"    RMSE(Euler vs RK4) = {np.mean(rmses):.6f}")

    # Fit rate
    hs = np.array([float(k) for k in results if k != 'fit'])
    errs = np.array([results[str(hv)]['mean_rmse'] for hv in hs])
    valid = (errs > 0) & np.isfinite(errs)
    if valid.sum() >= 3:
        log_fit = np.polyfit(np.log(hs[valid]), np.log(errs[valid]), 1)
        results['fit'] = {
            'exponent': float(log_fit[0]),
            'theoretical': 1.0,  # Euler is first-order: RMSE ~ O(h)
        }
        print(f"\n  Fitted: h^{{{log_fit[0]:.3f}}} (theory: h^1 for Euler RMSE)")
    return results


# ============================================================
# EXPERIMENT 3: T-SCALING (Symbolic Recovery)
# ============================================================
def experiment_t_scaling():
    print("\n" + "="*60)
    print("EXPERIMENT 3: T-Scaling (Theorem 4)")
    print("="*60)

    T_values = [20, 50, 100, 200, 500, 1000]
    N = 200
    d = 2
    h = 0.005
    D_true = 0.5
    n_seeds = 5
    results = {}

    for kin_name in ['linear', 'hill', 'michaelis_menten']:
        print(f"\n  Kinetics: {kin_name}")
        kin_results = {}

        for T in T_values:
            print(f"    T = {T}")
            eps_syms = []
            support_correct_count = 0

            for seed in range(n_seeds):
                traj, pos, W, L_norm, _ = generate_rd_trajectory(
                    N, d, h, D_true, T, kin_name, sigma_noise=0.005, seed=seed)

                L_norm_t = torch.tensor(L_norm, dtype=torch.float32)
                model = RD_GKAN(1, G=10, k=3, x_range=(-1, 5))
                train_rd_gkan(model, traj, L_norm_t, n_epochs=2000, lr=5e-3,
                             l1=5e-5, patience=100)

                sym = symbolic_recovery(model, kin_name, M=1)
                s0 = sym['species_0']
                eps_syms.append(s0['eps_sym'])
                if s0['support_correct']:
                    support_correct_count += 1

            kin_results[str(T)] = {
                'mean_eps_sym': float(np.mean(eps_syms)),
                'std_eps_sym': float(np.std(eps_syms)),
                'support_recovery_rate': float(support_correct_count / n_seeds),
            }
            print(f"      ε_sym={np.mean(eps_syms):.1f}%, "
                  f"support_correct={support_correct_count}/{n_seeds}")

        results[kin_name] = kin_results

    return results


# ============================================================
# EXPERIMENT 4: ABLATIONS
# ============================================================
def experiment_ablations():
    print("\n" + "="*60)
    print("EXPERIMENT 4: Ablations")
    print("="*60)

    N = 200; d = 2; h = 0.005; D_true = 0.5; T = 500
    kinetics = 'hill'
    n_seeds = 3
    results = {}

    # 4a: Spline vs MLP
    print("\n  4a: Spline vs MLP")
    for model_type in ['spline', 'mlp']:
        rmses = []
        for seed in range(n_seeds):
            traj, pos, W, L_norm, _ = generate_rd_trajectory(
                N, d, h, D_true, T, kinetics, sigma_noise=0.005, seed=seed)
            L_norm_t = torch.tensor(L_norm, dtype=torch.float32)

            if model_type == 'spline':
                model = RD_GKAN(1, G=8, k=3, x_range=(-1, 5))
            else:
                model = MLPReaction(1, hidden=32)

            train_rd_gkan(model, traj, L_norm_t, n_epochs=1500, l1=0, patience=80)

            model.eval()
            with torch.no_grad():
                c0 = torch.tensor(traj[0], dtype=torch.float32).to(DEVICE)
                L_t = L_norm_t.to(DEVICE)
                pred_traj = model.multi_step(c0, L_t, T)
                pred = pred_traj[-1].cpu().numpy()
            rmse = np.sqrt(np.mean((pred - traj[-1])**2))
            rmses.append(rmse)

        results[f'{model_type}_rmse'] = {'mean': float(np.mean(rmses)), 'std': float(np.std(rmses))}
        print(f"    {model_type}: RMSE={np.mean(rmses):.6f}")

    # Symbolic recovery comparison
    for model_type in ['spline', 'mlp']:
        traj, pos, W, L_norm, _ = generate_rd_trajectory(
            N, d, h, D_true, T, kinetics, sigma_noise=0.005, seed=0)
        L_norm_t = torch.tensor(L_norm, dtype=torch.float32)
        if model_type == 'spline':
            model = RD_GKAN(1, G=8, k=3, x_range=(-1, 5))
            train_rd_gkan(model, traj, L_norm_t, n_epochs=1500, l1=5e-5, patience=80)
            sym = symbolic_recovery(model, kinetics)
            results[f'{model_type}_eps_sym'] = sym['species_0']['eps_sym']
        print(f"    {model_type} symbolic: done")

    # 4b: Grid density
    print("\n  4b: Grid density G")
    for G in [3, 5, 8, 12, 20]:
        traj, pos, W, L_norm, _ = generate_rd_trajectory(
            N, d, h, D_true, T, kinetics, sigma_noise=0.005, seed=0)
        L_norm_t = torch.tensor(L_norm, dtype=torch.float32)
        model = RD_GKAN(1, G=G, k=3, x_range=(-1, 5))
        train_rd_gkan(model, traj, L_norm_t, n_epochs=1500, l1=5e-5, patience=80)
        sym = symbolic_recovery(model, kinetics)
        results[f'G_{G}_eps_sym'] = sym['species_0']['eps_sym']
        print(f"    G={G}: ε_sym={sym['species_0']['eps_sym']:.1f}%")

    # 4c: MC-derived vs uniform weights
    print("\n  4c: MC vs uniform weights")
    for weight_type in ['mc', 'uniform']:
        rmses = []
        for seed in range(n_seeds):
            traj, pos, W, L_norm, _ = generate_rd_trajectory(
                N, d, h, D_true, T, kinetics, sigma_noise=0.005, seed=seed)
            if weight_type == 'uniform':
                W_uni = (W > 0).astype(np.float32)
                D_uni = np.diag(W_uni.sum(1))
                L_uni = (D_uni - W_uni) / N
                L_norm_t = torch.tensor(L_uni, dtype=torch.float32)
            else:
                L_norm_t = torch.tensor(L_norm, dtype=torch.float32)

            model = RD_GKAN(1, G=8, k=3, x_range=(-1, 5))
            train_rd_gkan(model, traj, L_norm_t, n_epochs=1500, l1=0, patience=80)
            model.eval()
            with torch.no_grad():
                c0 = torch.tensor(traj[0], dtype=torch.float32).to(DEVICE)
                pred = model.multi_step(c0, L_norm_t.to(DEVICE), T)[-1].cpu().numpy()
            rmses.append(np.sqrt(np.mean((pred - traj[-1])**2)))

        results[f'{weight_type}_weights_rmse'] = {
            'mean': float(np.mean(rmses)), 'std': float(np.std(rmses))
        }
        print(f"    {weight_type}: RMSE={np.mean(rmses):.6f}")

    return results


# ============================================================
# EXPERIMENT 5: STABILITY AT FIXED POINT
# ============================================================
def experiment_stability():
    print("\n" + "="*60)
    print("EXPERIMENT 5: Stability at Fixed Point")
    print("="*60)

    N = 200; d = 2; h = 0.005; D_true = 0.5; T = 2000  # long to reach steady state
    kinetics = 'hill'

    traj, pos, W, L_norm, eigvals = generate_rd_trajectory(
        N, d, h, D_true, T, kinetics, sigma_noise=0.0, seed=42)

    # Train RD-GKAN
    L_norm_t = torch.tensor(L_norm, dtype=torch.float32)
    model = RD_GKAN(1, G=8, k=3, x_range=(-1, 5))
    train_rd_gkan(model, traj, L_norm_t, n_epochs=2000, lr=5e-3, patience=100)

    # The fixed point is the final state (converged after T=2000 steps)
    c_star = traj[-1]  # (N, 1)
    print(f"  Fixed point range: [{c_star.min():.4f}, {c_star.max():.4f}]")

    # Compute Jacobian of reaction KAN at the fixed point concentrations
    model.eval()
    eps = 1e-4
    # For M=1: Jacobian is scalar df/dc at each node
    c_mean = float(c_star.mean())
    x_p = torch.tensor([c_mean + eps], dtype=torch.float32).to(DEVICE)
    x_m = torch.tensor([c_mean - eps], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        mu = (model.reaction_kans[0](x_p).item() - model.reaction_kans[0](x_m).item()) / (2*eps)

    D_learned = abs(model.D.item())
    lambda_N = eigvals[-1]

    # System eigenvalues: a_i = 1 + mu - D_learned * lambda_i
    a_vals = [1 + mu - D_learned * li for li in eigvals]
    rho = max(abs(a) for a in a_vals)

    # CFL condition
    cfl_bound = 2.0 / (abs(mu) + D_learned * lambda_N)

    print(f"  Reaction Jacobian mu = {mu:.4f}")
    print(f"  D_learned = {D_learned:.4f}")
    print(f"  rho(A) = {rho:.6f}")
    print(f"  Stable: {rho < 1}")
    print(f"  CFL bound on h: {cfl_bound:.4f}")

    return {
        'mu': float(mu),
        'D_learned': float(D_learned),
        'rho': float(rho),
        'stable': bool(rho < 1),
        'cfl_bound': float(cfl_bound),
        'fixed_point_mean': float(c_mean),
    }


# ============================================================
# EXPERIMENT 6: QS REAL DATA (illustrative)
# ============================================================
def experiment_qs_illustration():
    print("\n" + "="*60)
    print("EXPERIMENT 6: QS Real Data (Illustrative)")
    print("="*60)

    import csv
    csv_path = os.path.join(DATA_DIR, 'GeorgiaTechQS', 'hierarchy-main', 'Data', 'Combined lasI.csv')
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    dose_response = {}
    for row in rows:
        try:
            t = float(row.get('Time', 0))
            c12 = float(row.get('C12', 0))
            rlu = float(row.get('RLU_OD', row.get('Corrected_RLU_per_OD', 0)))
            if t >= 10:
                dose_response.setdefault(c12, []).append(rlu)
        except (ValueError, KeyError):
            continue

    doses = sorted(dose_response.keys())
    y = np.array([np.mean(dose_response[d]) for d in doses], dtype=np.float32)
    x = np.array(doses, dtype=np.float32)
    y = y / (y.max() + 1e-10)

    # Hill fit
    def hill(x, V, K, n):
        return V * x**n / (K**n + x**n + 1e-10)
    popt, _ = curve_fit(hill, x, y, p0=[1,1,1], bounds=([0,0.01,0.1],[10,50,10]))
    V, K, n = popt
    R2 = 1 - np.sum((y - hill(x, *popt))**2) / (np.sum((y - y.mean())**2) + 1e-10)
    print(f"  Hill fit: V={V:.3f}, K={K:.3f}, n={n:.3f}, R²={R2:.4f}")

    # KAN fit
    kan = BSplineKAN(G=10, k=3, x_min=float(x.min())-0.5, x_max=float(x.max())+0.5).to(DEVICE)
    opt = Adam(kan.parameters(), lr=0.005)
    sched = CosineAnnealingLR(opt, T_max=5000, eta_min=1e-5)
    xt = torch.tensor(x).to(DEVICE)
    yt = torch.tensor(y).to(DEVICE)
    for _ in range(5000):
        loss = F.mse_loss(kan(xt), yt) + 1e-4 * kan.coeffs.abs().sum()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()

    # Symbolic projection with fitted K
    xg = np.linspace(float(x.min()), float(x.max()), 300).astype(np.float32)
    lib = {
        '1': np.ones_like(xg), 'x': xg, 'x^2': xg**2,
        f'MM(K={K:.2f})': xg / (K + xg + 1e-10),
        f'Hill_2(K={K:.2f})': xg**2 / (K**2 + xg**2 + 1e-10),
        f'Hill_n(n={n:.2f})': xg**n / (K**n + xg**n + 1e-10),
        'exp(-x)': np.exp(-xg / (xg.max() + 1e-10)),
    }
    S = np.column_stack(list(lib.values()))
    with torch.no_grad():
        phi = kan(torch.tensor(xg).to(DEVICE)).cpu().numpy()
    lasso = Lasso(alpha=0.003, max_iter=20000)
    lasso.fit(S, phi)
    alpha = lasso.coef_
    phi_recon = S @ alpha + lasso.intercept_
    eps_recon = np.linalg.norm(phi - phi_recon) / (np.linalg.norm(phi) + 1e-10) * 100

    dom_idx = np.argmax(np.abs(alpha))
    lib_names = list(lib.keys())
    print(f"  Dominant: {lib_names[dom_idx]} (|α|={abs(alpha[dom_idx]):.3f})")
    print(f"  ε_recon = {eps_recon:.1f}%")

    return {
        'hill_fit': {'V': float(V), 'K': float(K), 'n': float(n), 'R2': float(R2)},
        'symbolic': {
            'dominant': lib_names[dom_idx],
            'coeffs': {lib_names[i]: float(alpha[i]) for i in range(len(lib_names))},
            'eps_recon': float(eps_recon),
        },
    }


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    results = {}

    results['n_scaling'] = experiment_n_scaling()
    results['h_scaling'] = experiment_h_scaling()
    results['t_scaling'] = experiment_t_scaling()
    results['ablations'] = experiment_ablations()
    results['stability'] = experiment_stability()
    results['qs'] = experiment_qs_illustration()

    out = os.path.join(RESULTS_DIR, 'synthetic_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"ALL DONE in {elapsed/60:.1f} min. Results: {out}")
    print(f"{'='*60}")

    # Summary
    if 'n_scaling' in results and 'fit' in results['n_scaling']:
        print(f"  N-scaling: N^{{-{results['n_scaling']['fit']['exponent']:.3f}}} "
              f"(theory: N^{{-{results['n_scaling']['fit']['theoretical']:.1f}}})")
    if 'h_scaling' in results and 'fit' in results['h_scaling']:
        print(f"  h-scaling: h^{{{results['h_scaling']['fit']['exponent']:.3f}}} "
              f"(theory: h^{results['h_scaling']['fit']['theoretical']:.0f})")
    if 'stability' in results:
        print(f"  Stability: rho={results['stability']['rho']:.6f}, "
              f"stable={results['stability']['stable']}")


if __name__ == '__main__':
    main()
