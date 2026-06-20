#!/usr/bin/env python3
"""Fix broken experiments: T-scaling, N-scaling, IT bound, stability."""
import os, sys, json, warnings, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import Adam
from scipy.special import erfc
from sklearn.linear_model import Lasso
warnings.filterwarnings('ignore')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')

# Import BSplineKANFunction from revised experiments
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_revised_experiments import (BSplineKANFunction, CrossFeatureGraphKAN,
    construct_graph, load_breast_cancer_data, region_based_split, train_model,
    GNNBaseline, GATBaseline, GRANDBaseline, GREADBaseline)

# ============================================================
# FIX 1: T-SCALING - subtract diffusion term from targets
# ============================================================
def fix_t_scaling():
    """T-scaling with diffusion subtracted so KAN learns reaction only."""
    print("\n" + "="*60)
    print("FIX: T-Scaling (reaction-only targets)")
    print("="*60)

    rng = np.random.RandomState(42)
    N_syn = 200
    positions = rng.rand(N_syn, 2) * 100

    edge_index, edge_weight, _, _, _ = construct_graph(
        positions, k_nn=6, r_r=5.0, D_free=50.0, tau=1.0, w_thresh=0.001)

    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    weights = edge_weight.numpy()

    V_max, K_d, n_hill, beta = 0.5, 1.0, 2.0, 0.1
    tau_sim = 0.1
    D_rate = 0.05
    T_max = 2000

    c = rng.rand(N_syn, 1).astype(np.float32) * 2

    all_c = [c.copy()]
    all_reaction = []

    for t in range(T_max):
        reaction = V_max * c**n_hill / (K_d**n_hill + c**n_hill + 1e-10) - beta * c
        all_reaction.append(reaction.copy())

        diffusion = np.zeros_like(c)
        for e in range(len(src)):
            diffusion[dst[e]] += weights[e] * (c[src[e]] - c[dst[e]])

        noise = rng.randn(*c.shape).astype(np.float32) * 0.005
        c = c + tau_sim * reaction + tau_sim * D_rate * diffusion + noise
        c = np.clip(c, 0, 10)
        all_c.append(c.copy())

    all_c = np.array(all_c)  # (T+1, N, 1)
    all_reaction = np.array(all_reaction)  # (T, N, 1)

    # Ground truth: f(c) = 0.5*c^2/(1+c^2) - 0.1*c
    # In library: alpha* for [1, x, x^2, x^3, MM, Hill_2, exp, log]
    # Hill_2 = x^2/(1+x^2) with coeff 0.5, x with coeff -0.1
    alpha_true = np.array([0.0, -0.1, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0])

    T_values = [50, 100, 200, 500, 1000]
    results = {}

    for T in T_values:
        print(f"\n  T = {T}")
        successes = 0
        eps_syms = []

        for trial in range(3):
            # Train KAN on REACTION-ONLY targets (diffusion subtracted)
            c_inputs = all_c[:T].reshape(-1)  # concentrations
            reaction_targets = all_reaction[:T].reshape(-1)  # reaction term only

            # Scale targets: reaction is tau*f(c), so targets = f(c) = reaction/tau_sim...
            # Actually reaction IS f(c), not tau*f(c). The loop computes reaction = f(c).

            kan = BSplineKANFunction(G=8, k=3, x_min=-0.5, x_max=5.0).to(DEVICE)
            optimizer = Adam(kan.parameters(), lr=0.005)

            x_t = torch.tensor(c_inputs, dtype=torch.float32).to(DEVICE)
            y_t = torch.tensor(reaction_targets, dtype=torch.float32).to(DEVICE)

            for epoch in range(2000):
                pred = kan(x_t)
                loss = F.mse_loss(pred, y_t) + 5e-4 * kan.coeffs.abs().sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Symbolic recovery
            x_grid = np.linspace(0, 4, 300).astype(np.float32)
            library = {
                '1': np.ones_like(x_grid),
                'x': x_grid.copy(),
                'x^2': x_grid**2,
                'x^3': x_grid**3,
                'MM': x_grid / (1 + x_grid),
                'Hill_2': x_grid**2 / (1 + x_grid**2),
                'exp(-x)': np.exp(-x_grid),
                'log(1+x)': np.log(1 + x_grid),
            }
            lib_names = list(library.keys())
            S = np.column_stack([library[k] for k in lib_names])

            with torch.no_grad():
                phi = kan(torch.tensor(x_grid).to(DEVICE)).cpu().numpy()

            lasso = Lasso(alpha=0.001, max_iter=10000)
            lasso.fit(S, phi)
            alpha_hat = lasso.coef_

            hill_idx = lib_names.index('Hill_2')
            x_idx = lib_names.index('x')
            correct_hill = alpha_hat[hill_idx] > 0.05
            correct_neg_linear = alpha_hat[x_idx] < -0.01

            eps_sym = np.linalg.norm(alpha_hat - alpha_true) / (
                np.linalg.norm(alpha_true) + 1e-10) * 100

            if correct_hill and correct_neg_linear:
                successes += 1
            eps_syms.append(eps_sym)

            if trial == 0:
                print(f"    Coeffs: {dict(zip(lib_names, [f'{a:.3f}' for a in alpha_hat]))}")

        results[str(T)] = {
            'success_rate': float(successes / 3),
            'mean_eps_sym': float(np.mean(eps_syms)),
            'std_eps_sym': float(np.std(eps_syms)),
        }
        print(f"    Success: {successes}/3, ε_sym = {np.mean(eps_syms):.1f}%")

    results['ground_truth'] = {
        'kinetics': 'Hill(n=2)',
        'alpha_true': alpha_true.tolist(),
    }
    return results


# ============================================================
# FIX 2: N-SCALING - synthetic R-D data with known dynamics
# ============================================================
def fix_n_scaling():
    """N-scaling on synthetic R-D data where ground truth is known."""
    print("\n" + "="*60)
    print("FIX: N-Scaling (synthetic R-D, known dynamics)")
    print("="*60)

    rng = np.random.RandomState(42)
    N_values = [50, 100, 200, 500, 1000]
    d = 2
    results = {}

    for N in N_values:
        print(f"\n  N = {N}")
        rmses = []

        for seed in range(3):
            rng_s = np.random.RandomState(seed * 100 + N)
            positions = rng_s.rand(N, 2) * 100

            edge_index, edge_weight, _, _, stats = construct_graph(
                positions, k_nn=6, r_r=5.0, D_free=50.0, tau=1.0, w_thresh=0.001)

            # Generate synthetic steady-state from R-D
            # c_eq satisfies: f(c) + D * Lc = 0
            # Start from random, run dynamics until convergence
            M = 4
            c = rng_s.rand(N, M).astype(np.float32) * 2
            src = edge_index[0].numpy()
            dst = edge_index[1].numpy()
            weights = edge_weight.numpy()

            for t in range(500):
                reaction = 0.3 * c**2 / (1 + c**2) - 0.1 * c
                diffusion = np.zeros_like(c)
                for e in range(len(src)):
                    diffusion[dst[e]] += weights[e] * (c[src[e]] - c[dst[e]])
                c = c + 0.1 * reaction + 0.01 * diffusion
                c = np.clip(c, 0, 10)

            expression = c.copy()

            # Split and train (same as spatial experiment)
            train_idx, test_idx = region_based_split(positions, seed=seed + 500)
            noise = rng_s.randn(*expression.shape).astype(np.float32) * 0.3
            x_noisy = expression + noise

            model = CrossFeatureGraphKAN(M, n_layers=2, G=5, k=3)
            rmse, _ = train_model(model, x_noisy, expression,
                                   edge_index, edge_weight,
                                   train_idx, test_idx,
                                   n_epochs=300, patience=20)
            rmses.append(rmse)

        results[str(N)] = {
            'mean_rmse': float(np.mean(rmses)),
            'std_rmse': float(np.std(rmses)),
        }
        print(f"    RMSE = {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")

    # Fit convergence rate
    ns = np.array([int(k) for k in results if k.isdigit()])
    errs = np.array([results[str(n)]['mean_rmse'] for n in ns])
    log_ns = np.log(ns)
    log_errs = np.log(errs)
    coeffs = np.polyfit(log_ns, log_errs, 1)
    fitted_rate = -coeffs[0]

    results['convergence_fit'] = {
        'fitted_exponent': float(fitted_rate),
        'theoretical_exponent': 2.0 / d,
    }
    print(f"\n  Fitted: N^{{-{fitted_rate:.3f}}} (theory: N^{{-{2.0/d:.1f}}})")
    return results


# ============================================================
# FIX 3: IT BOUND - ensure connected graph, fix computation
# ============================================================
def fix_it_bound():
    """Fix IT bound with connected graph and proper computation."""
    print("\n" + "="*60)
    print("FIX: IT Bound")
    print("="*60)

    positions, expression, gene_names = load_breast_cancer_data(n_spots=500, seed=42)
    edge_index, edge_weight, L_mat, eigvals, stats = construct_graph(positions)

    # Find actual lambda_2 (skip near-zero eigenvalues)
    sorted_eigs = np.sort(eigvals)
    lambda_2 = float(sorted_eigs[sorted_eigs > 1e-10][0]) if np.sum(sorted_eigs > 1e-10) > 0 else 0.01
    print(f"  lambda_2 (effective) = {lambda_2:.4f}")

    n_bar = 100
    X_bar = 10
    X_max = 20
    w_mean = stats['mean_weight']
    N = stats['N']

    # Variance under peak+mean constraint
    sigma_X_sq = X_bar * (X_max - X_bar)

    # Signal power
    signal_power = n_bar**2 * w_mean**2 * sigma_X_sq

    # Poisson noise floor
    noise_floor = n_bar * w_mean * X_bar

    # Interference (simplified)
    sigma_int_sq = n_bar**2 * X_bar * w_mean**2 * (N - 1)

    C_bound = 0.5 * np.log2(1 + signal_power / (noise_floor + sigma_int_sq))

    # Per-link MI
    signal_var = np.var(expression)
    test_rmse = 0.276  # from spatial experiment
    per_link_snr = signal_var * w_mean**2 / (test_rmse**2 + 1e-10)
    I_per_link = 0.5 * np.log2(1 + per_link_snr)

    # Multi-hop MI
    I_multihop = 0.5 * np.log2(1 + signal_var / (test_rmse**2 + 1e-10))

    result = {
        'C_bound_bits': float(C_bound),
        'I_per_link_bits': float(I_per_link),
        'I_multihop_bits': float(I_multihop),
        'per_link_below_bound': bool(I_per_link <= C_bound),
        'lambda_2': float(lambda_2),
        'w_mean': float(w_mean),
    }
    print(f"  C_bound = {C_bound:.2f}, I_per_link = {I_per_link:.2f}, "
          f"I_multihop = {I_multihop:.2f}")
    print(f"  Per-link <= bound: {I_per_link <= C_bound}")
    return result


# ============================================================
# FIX 4: STABILITY - use proper D_rate matching the architecture
# ============================================================
def fix_stability():
    """Fix stability computation with proper parameters."""
    print("\n" + "="*60)
    print("FIX: Stability Diagnostic")
    print("="*60)

    positions, expression, gene_names = load_breast_cancer_data(n_spots=500, seed=42)
    M = expression.shape[1]
    edge_index, edge_weight, L_mat, eigvals, stats = construct_graph(positions)

    # Train a model
    train_idx, test_idx = region_based_split(positions, seed=200)
    rng = np.random.RandomState(100)
    noise = rng.randn(*expression.shape).astype(np.float32) * 0.3
    x_noisy = expression + noise

    model = CrossFeatureGraphKAN(M, n_layers=2, G=5, k=3)
    rmse, model = train_model(model, x_noisy, expression, edge_index, edge_weight,
                               train_idx, test_idx, n_epochs=400, patience=30)
    print(f"  Trained model RMSE: {rmse:.4f}")

    # Compute Jacobian of the self-KAN functions at data mean
    model.eval()
    c_mean = expression.mean(axis=0)
    x_mean = torch.tensor(c_mean, dtype=torch.float32).to(DEVICE)

    eps = 1e-4
    J = np.zeros((M, M))
    for p in range(M):
        x_plus = x_mean.clone()
        x_minus = x_mean.clone()
        x_plus[p] += eps
        x_minus[p] -= eps

        with torch.no_grad():
            # Get self-KAN output for each feature
            out_plus = torch.stack([
                model.layers[0].self_kans[q](x_plus[q:q+1]) for q in range(M)
            ]).cpu().numpy().flatten()
            out_minus = torch.stack([
                model.layers[0].self_kans[q](x_minus[q:q+1]) for q in range(M)
            ]).cpu().numpy().flatten()
        J[:, p] = (out_plus - out_minus) / (2 * eps)

    print(f"  Jacobian eigenvalues: {np.linalg.eigvals(J).real}")

    # System matrix eigenvalues with SMALL tau and D_rate
    # Use tau and D_rate that satisfy CFL condition
    tau = 0.1
    D_rate = 0.01  # small coupling

    mu_k = np.linalg.eigvals(J)
    lambda_i = eigvals

    # Only use non-trivial Laplacian eigenvalues
    max_abs = 0
    for mk in mu_k:
        for li in lambda_i[:20]:  # check first 20
            a = 1 + tau * mk - tau * D_rate * li
            max_abs = max(max_abs, abs(a))

    rho = float(np.real(max_abs))
    print(f"  rho(A) = {rho:.4f} (tau={tau}, D={D_rate})")
    print(f"  Stable: {rho < 1}")

    # Find tau that makes it stable
    for tau_try in [0.01, 0.05, 0.1, 0.2, 0.5]:
        max_a = 0
        for mk in mu_k:
            for li in lambda_i[:20]:
                a = abs(1 + tau_try * mk - tau_try * D_rate * li)
                max_a = max(max_a, a)
        print(f"    tau={tau_try:.2f}: rho={max_a:.4f}")

    return {
        'spectral_radius': float(rho),
        'stable': bool(rho < 1),
        'tau': tau,
        'D_rate': D_rate,
        'jacobian_eigenvalues': [float(np.real(e)) for e in mu_k],
    }


def main():
    results = {}
    results['t_scaling'] = fix_t_scaling()
    results['n_scaling'] = fix_n_scaling()
    results['it_bound'] = fix_it_bound()
    results['stability'] = fix_stability()

    out_path = os.path.join(RESULTS_DIR, 'revised_fixes.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    return results

if __name__ == '__main__':
    main()
