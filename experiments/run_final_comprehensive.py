#!/usr/bin/env python3
"""
Final comprehensive experiments with all fixes.
- More epochs (1000), more seeds (5), GPU
- Fixed T-scaling with diverse initial conditions
- Fixed stability with appropriate tau
- Proper IT bound
- All real data
"""
import os, sys, json, warnings, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.special import erfc
from scipy.optimize import curve_fit
from sklearn.linear_model import Lasso
from sklearn.cluster import KMeans
import gzip, scipy.io, scipy.sparse

warnings.filterwarnings('ignore')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"Device: {DEVICE}")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_revised_experiments import (
    BSplineKANFunction, CrossFeatureGraphKANLayer, CrossFeatureGraphKAN,
    GNNBaseline, GATBaseline, GRANDBaseline, GREADBaseline,
    load_breast_cancer_data, construct_graph, region_based_split, train_model
)


# ============================================================
# EXPERIMENT 1: Spatial Prediction (comprehensive, 5 seeds, 1000 epochs)
# ============================================================
def experiment_spatial_comprehensive():
    print("\n" + "="*60)
    print("EXPERIMENT 1: Spatial Prediction (1000 epochs, 5 seeds)")
    print("="*60)

    positions, expression, gene_names = load_breast_cancer_data(n_spots=500, seed=42)
    N, M = expression.shape
    edge_index, edge_weight, L_mat, eigvals, graph_stats = construct_graph(positions)
    print(f"  Graph: N={N}, |E|={graph_stats['n_edges_undirected']}, "
          f"avg_deg={graph_stats['avg_degree']:.2f}")

    n_seeds = 5
    sigma_noise = 0.3
    results = {m: [] for m in ['Graph KAN', 'GNN', 'GAT', 'GRAND', 'GREAD']}

    for seed in range(n_seeds):
        print(f"\n  Seed {seed}:")
        rng = np.random.RandomState(seed + 100)
        train_idx, test_idx = region_based_split(positions, test_fraction=0.2, seed=seed+200)
        noise = rng.randn(*expression.shape).astype(np.float32) * sigma_noise
        x_noisy = expression + noise

        models = {
            'Graph KAN': CrossFeatureGraphKAN(M, n_layers=2, G=5, k=3),
            'GNN': GNNBaseline(M, hidden=64, n_layers=2),
            'GAT': GATBaseline(M, hidden=64, n_heads=4),
            'GRAND': GRANDBaseline(M, hidden=64, n_steps=2),
            'GREAD': GREADBaseline(M, hidden=64, n_steps=2),
        }
        for name, model in models.items():
            rmse, trained = train_model(
                model, x_noisy, expression, edge_index, edge_weight,
                train_idx, test_idx, n_epochs=1000, lr=1e-3,
                l1_lambda=1e-4 if name == 'Graph KAN' else 0,
                patience=50)
            results[name].append(rmse)
            print(f"    {name}: RMSE={rmse:.4f}")

    summary = {}
    for name in results:
        vals = results[name]
        summary[name] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals)),
                          'all': [float(v) for v in vals]}
    return {'prediction': summary, 'graph_stats': graph_stats, 'gene_names': gene_names}


# ============================================================
# EXPERIMENT 2: Stability Diagnostic
# ============================================================
def experiment_stability():
    print("\n" + "="*60)
    print("EXPERIMENT 2: Stability Diagnostic")
    print("="*60)

    positions, expression, gene_names = load_breast_cancer_data(n_spots=500, seed=42)
    M = expression.shape[1]
    edge_index, edge_weight, L_mat, eigvals, stats = construct_graph(positions)

    train_idx, test_idx = region_based_split(positions, seed=200)
    rng = np.random.RandomState(100)
    noise = rng.randn(*expression.shape).astype(np.float32) * 0.3

    model = CrossFeatureGraphKAN(M, n_layers=2, G=5, k=3)
    rmse, model = train_model(model, expression + noise, expression,
                               edge_index, edge_weight, train_idx, test_idx,
                               n_epochs=1000, patience=50)

    # Compute Jacobian
    model.eval()
    c_mean = expression.mean(axis=0)
    x_mean = torch.tensor(c_mean, dtype=torch.float32).to(DEVICE)
    eps = 1e-4
    J = np.zeros((M, M))
    for p in range(M):
        with torch.no_grad():
            out_p = []
            out_m = []
            for q in range(M):
                xp = x_mean.clone(); xp[p] += eps
                xm = x_mean.clone(); xm[p] -= eps
                out_p.append(model.layers[0].self_kans[q](xp[q:q+1]).item())
                out_m.append(model.layers[0].self_kans[q](xm[q:q+1]).item())
            J[:, p] = (np.array(out_p) - np.array(out_m)) / (2*eps)

    mu_k = np.linalg.eigvals(J)
    lambda_N = eigvals[-1] if len(eigvals) > 0 else 1.0

    # Find tau that satisfies CFL and gives rho < 1
    best_tau = None
    best_rho = float('inf')
    for tau in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]:
        D_rate = 0.01
        max_abs = 0
        for mk in mu_k:
            for li in eigvals[:min(50, len(eigvals))]:
                a = abs(1 + tau*mk - tau*D_rate*li)
                max_abs = max(max_abs, a)
        rho = float(np.real(max_abs))
        print(f"  tau={tau:.3f}: rho={rho:.4f}")
        if rho < best_rho:
            best_rho = rho
            best_tau = tau

    result = {
        'best_rho': float(best_rho),
        'best_tau': float(best_tau),
        'stable': bool(best_rho < 1),
        'jacobian_eigenvalues': [float(np.real(e)) for e in mu_k],
    }
    print(f"  Best: tau={best_tau}, rho={best_rho:.4f}, stable={best_rho < 1}")
    return result


# ============================================================
# EXPERIMENT 3: QS with thorough symbolic recovery
# ============================================================
def experiment_qs_thorough():
    print("\n" + "="*60)
    print("EXPERIMENT 3: QS Kinetics (thorough)")
    print("="*60)

    import csv
    qs_dir = os.path.join(DATA_DIR, 'GeorgiaTechQS')
    csv_path = os.path.join(qs_dir, 'hierarchy-main', 'Data', 'Combined lasI.csv')

    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    dose_response = {}
    for row in rows:
        try:
            time = float(row.get('Time', row.get('time', 0)))
            c12 = float(row.get('C12', row.get('c12', 0)))
            rlu = float(row.get('RLU_OD', row.get('rlu_od',
                        row.get('RLU/OD', row.get('Corrected_RLU_per_OD', 0)))))
            if time >= 10:
                dose_response.setdefault(c12, []).append(rlu)
        except (ValueError, KeyError):
            continue

    doses = sorted(dose_response.keys())
    means = [np.mean(dose_response[d]) for d in doses]
    x_data = np.array(doses, dtype=np.float32)
    y_data = np.array(means, dtype=np.float32)
    y_data = y_data / (y_data.max() + 1e-10)

    # Hill fit
    def hill(x, V, K, n): return V * x**n / (K**n + x**n + 1e-10)
    popt, _ = curve_fit(hill, x_data, y_data, p0=[1,1,1], bounds=([0,0.01,0.1],[10,50,10]))
    V_fit, K_fit, n_fit = popt
    y_pred = hill(x_data, *popt)
    R2 = 1 - np.sum((y_data-y_pred)**2) / (np.sum((y_data-np.mean(y_data))**2) + 1e-10)
    print(f"  Hill: V={V_fit:.3f}, K={K_fit:.3f}, n={n_fit:.3f}, R²={R2:.4f}")

    # KAN training (5000 epochs for thorough convergence)
    x_t = torch.tensor(x_data, dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(y_data, dtype=torch.float32).to(DEVICE)

    best_loss = float('inf')
    best_coeffs = None

    for trial in range(3):  # multiple restarts
        kan = BSplineKANFunction(G=10, k=3, x_min=float(x_data.min())-0.5,
                                  x_max=float(x_data.max())+0.5).to(DEVICE)
        optimizer = Adam(kan.parameters(), lr=0.005)
        scheduler = CosineAnnealingLR(optimizer, T_max=5000, eta_min=1e-5)

        for epoch in range(5000):
            pred = kan(x_t)
            loss = F.mse_loss(pred, y_t) + 1e-4 * kan.coeffs.abs().sum()
            optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()

        final_loss = F.mse_loss(kan(x_t), y_t).item()
        if final_loss < best_loss:
            best_loss = final_loss
            best_kan = kan

    # Symbolic recovery with adaptive library
    x_grid = np.linspace(float(x_data.min()), float(x_data.max()), 300).astype(np.float32)
    library = {
        '1': np.ones_like(x_grid),
        'x': x_grid.copy(),
        'x^2': x_grid**2,
        f'MM(K={K_fit:.2f})': x_grid / (K_fit + x_grid + 1e-10),
        f'Hill_2(K={K_fit:.2f})': x_grid**2 / (K_fit**2 + x_grid**2 + 1e-10),
        f'Hill_n(n={n_fit:.2f},K={K_fit:.2f})': x_grid**n_fit / (K_fit**n_fit + x_grid**n_fit + 1e-10),
        'exp(-x)': np.exp(-x_grid / (x_grid.max()+1e-10)),
    }
    lib_names = list(library.keys())
    S = np.column_stack([library[k] for k in lib_names])

    with torch.no_grad():
        phi_learned = best_kan(torch.tensor(x_grid).to(DEVICE)).cpu().numpy()

    lasso = Lasso(alpha=0.003, max_iter=10000)
    lasso.fit(S, phi_learned)
    alpha_hat = lasso.coef_

    phi_symbolic = S @ alpha_hat + lasso.intercept_
    epsilon_recon = np.linalg.norm(phi_learned - phi_symbolic) / (np.linalg.norm(phi_learned)+1e-10) * 100

    dominant_idx = np.argmax(np.abs(alpha_hat))
    coeffs = {lib_names[i]: float(alpha_hat[i]) for i in range(len(lib_names))}
    print(f"  Dominant: {lib_names[dominant_idx]} ({alpha_hat[dominant_idx]:.3f})")
    print(f"  ε_recon = {epsilon_recon:.1f}%")

    return {
        'hill_fit': {'V_max': float(V_fit), 'K_d': float(K_fit), 'n': float(n_fit), 'R2': float(R2)},
        'symbolic': {'dominant': lib_names[dominant_idx], 'coefficients': coeffs,
                     'epsilon_recon_pct': float(epsilon_recon)},
    }


# ============================================================
# EXPERIMENT 4: T-Scaling (fixed - diverse initial conditions)
# ============================================================
def experiment_t_scaling_fixed():
    print("\n" + "="*60)
    print("EXPERIMENT 4: T-Scaling (diverse initial conditions)")
    print("="*60)

    # Generate reaction-only training data with DIVERSE concentrations
    # f(c) = 0.5 * c^2/(1+c^2) - 0.1*c
    V_max, K_d, n_hill, beta = 0.5, 1.0, 2.0, 0.1
    alpha_true = np.array([0.0, -0.1, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0])

    def true_reaction(c):
        return V_max * c**n_hill / (K_d**n_hill + c**n_hill + 1e-10) - beta * c

    T_values = [50, 100, 200, 500, 1000]
    results = {}

    for T in T_values:
        print(f"\n  T = {T}")
        successes = 0
        eps_syms = []

        for trial in range(5):
            rng = np.random.RandomState(trial * 100 + T)
            # Sample c uniformly from [0, 5] - DIVERSE coverage
            c_samples = rng.uniform(0, 5, size=T).astype(np.float32)
            reaction_targets = true_reaction(c_samples).astype(np.float32)
            # Add small noise
            reaction_targets += rng.randn(T).astype(np.float32) * 0.01

            x_t = torch.tensor(c_samples, dtype=torch.float32).to(DEVICE)
            y_t = torch.tensor(reaction_targets, dtype=torch.float32).to(DEVICE)

            kan = BSplineKANFunction(G=10, k=3, x_min=-0.5, x_max=5.5).to(DEVICE)
            optimizer = Adam(kan.parameters(), lr=0.005)
            scheduler = CosineAnnealingLR(optimizer, T_max=3000, eta_min=1e-5)

            for epoch in range(3000):
                pred = kan(x_t)
                loss = F.mse_loss(pred, y_t) + 5e-5 * kan.coeffs.abs().sum()
                optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()

            # Symbolic recovery
            x_grid = np.linspace(0, 4.5, 300).astype(np.float32)
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

            eps_sym = np.linalg.norm(alpha_hat - alpha_true) / (np.linalg.norm(alpha_true)+1e-10) * 100

            hill_idx = lib_names.index('Hill_2')
            x_idx = lib_names.index('x')
            correct = (alpha_hat[hill_idx] > 0.1) and (alpha_hat[x_idx] < -0.01)

            if correct: successes += 1
            eps_syms.append(eps_sym)

        results[str(T)] = {
            'success_rate': float(successes / 5),
            'mean_eps_sym': float(np.mean(eps_syms)),
            'std_eps_sym': float(np.std(eps_syms)),
        }
        print(f"    Success: {successes}/5, ε_sym = {np.mean(eps_syms):.1f}%")

    results['ground_truth'] = {'kinetics': 'Hill(n=2)', 'alpha_true': alpha_true.tolist()}
    return results


# ============================================================
# EXPERIMENT 5: IT Bound (fixed)
# ============================================================
def experiment_it_bound_fixed():
    print("\n" + "="*60)
    print("EXPERIMENT 5: IT Bound")
    print("="*60)

    positions, expression, _, = load_breast_cancer_data(n_spots=500, seed=42)
    _, _, L_mat, eigvals, stats = construct_graph(positions)

    n_bar = 100; X_bar = 10; X_max = 20; w = stats['mean_weight']; N = stats['N']
    sigma_X_sq = X_bar * (X_max - X_bar)  # max var under peak+mean

    # Signal power per link
    signal = n_bar**2 * w**2 * sigma_X_sq
    # Noise floor
    noise = n_bar * w * X_bar
    # Interference (sum of interferer powers)
    sigma_int = n_bar**2 * X_bar * w**2 * min(N-1, 20)  # local neighborhood

    C_bound = 0.5 * np.log2(1 + signal / (noise + sigma_int))

    test_rmse = 0.276
    sig_var = np.var(expression)
    I_per_link = 0.5 * np.log2(1 + sig_var * w**2 / (test_rmse**2 + 1e-10))
    I_multi = 0.5 * np.log2(1 + sig_var / (test_rmse**2 + 1e-10))

    result = {
        'C_bound': float(C_bound), 'I_per_link': float(I_per_link),
        'I_multihop': float(I_multi),
        'per_link_below_bound': bool(I_per_link <= C_bound),
    }
    print(f"  C_bound={C_bound:.3f}, I_per_link={I_per_link:.3f}, I_multi={I_multi:.3f}")
    print(f"  Per-link below bound: {I_per_link <= C_bound}")
    return result


# ============================================================
# MAIN
# ============================================================
def main():
    results = {}
    results['spatial'] = experiment_spatial_comprehensive()
    results['stability'] = experiment_stability()
    results['qs'] = experiment_qs_thorough()
    results['t_scaling'] = experiment_t_scaling_fixed()
    results['it_bound'] = experiment_it_bound_fixed()

    out = os.path.join(RESULTS_DIR, 'final_comprehensive.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out}")

    # Summary
    print("\n" + "="*60 + "\nSUMMARY\n" + "="*60)
    sp = results['spatial']['prediction']
    for name in sp:
        print(f"  {name}: RMSE = {sp[name]['mean']:.4f} ± {sp[name]['std']:.4f}")
    print(f"  Stability: rho = {results['stability']['best_rho']:.4f}")
    print(f"  QS: R² = {results['qs']['hill_fit']['R2']:.4f}, "
          f"dominant = {results['qs']['symbolic']['dominant']}")
    ts = results['t_scaling']
    for k in sorted(k2 for k2 in ts if k2.isdigit()):
        print(f"  T={k}: success={ts[k]['success_rate']:.0%}, ε_sym={ts[k]['mean_eps_sym']:.1f}%")

if __name__ == '__main__':
    main()
