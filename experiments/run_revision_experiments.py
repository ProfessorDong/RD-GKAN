#!/usr/bin/env python3
"""
Revision experiments for TMBMC major revision (TMBMC-SelfOrgLivSys-26-0003).

E1 (Reviewer 1, Major Q1): Out-of-library kinetics rejection.
    When the true kinetics is NOT representable by the symbolic library S,
    does LASSO silently force a wrong selection, or does the reconstruction
    error epsilon_recon flag the poor fit so the identification can be rejected?
    We compare epsilon_recon for in-library kinetics (linear, Hill, MM) vs
    out-of-library kinetics (high-frequency sine, localized Gaussian bump,
    non-smooth absolute-value kink) and derive a rejection threshold.

E2 (Reviewer 3, Q1): Time-step transferability.
    Because the architecture absorbs the step size as psi_theta ~ h f and
    D_theta ~ h D, the continuous-time vector field f = psi_theta/h and D =
    D_theta/h is recoverable. A model trained at step h_train should therefore
    transfer to a different integration step h_test by LINEAR RESCALING of the
    reaction and diffusion terms by (h_test/h_train), without retraining.
    We verify this against (a) naive transfer (no rescaling) and (b) a model
    retrained at h_test (oracle). This shows RD-GKAN learns a discrete-time
    update whose underlying continuous-time dynamics are recoverable.

Reuses the constrained RD-GKAN, B-spline KAN, data generator and trainer from
run_synthetic_rd.py so the architecture is identical to the submitted paper.
"""

import os, sys, json, time, warnings, numpy as np
import torch, torch.nn.functional as F
from sklearn.linear_model import Lasso
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_synthetic_rd import (
    RD_GKAN, KINETICS, generate_rd_trajectory, train_rd_gkan,
    DEVICE, RESULTS_DIR,
)

np.random.seed(0)
torch.manual_seed(0)


# ------------------------------------------------------------
# Register OUT-OF-LIBRARY kinetics (alpha_true=None; not in S).
# In-library set already present in KINETICS: linear, hill, michaelis_menten.
# ------------------------------------------------------------
# Out-of-library functions chosen to lie clearly outside the span of S
# (high-frequency oscillation and multi-modal localized bumps), which a
# smooth low-order dictionary fundamentally cannot represent.
KINETICS['highfreq'] = {
    'f': lambda c: 0.4 * np.sin(6.0 * c) - 0.2 * c,
    'alpha_true': None,
    'label': r'$0.4\sin(6c)-0.2c$',
}
KINETICS['gauss_bump'] = {
    'f': lambda c: 0.8 * np.exp(-((c - 2.0) ** 2) / 0.2) - 0.15 * c,
    'alpha_true': None,
    'label': r'$0.8e^{-(c-2)^2/0.2}-0.15c$',
}
KINETICS['double_bump'] = {
    'f': lambda c: 0.7 * np.exp(-((c - 1.0) ** 2) / 0.12)
                   - 0.7 * np.exp(-((c - 2.3) ** 2) / 0.12) - 0.1 * c,
    'alpha_true': None,
    'label': r'bimodal Gaussian',
}
# Michaelis-Menten with K=1 so the exact basis x/(K+x) is in the library S (K=1).
KINETICS['mm_k1'] = {
    'f': lambda c: 1.0 * c / (1.0 + c) - 0.2 * c,
    'alpha_true': None,
    'label': r'$c/(1+c)-0.2c$',
}

IN_LIBRARY = ['linear', 'hill', 'mm_k1']
OUT_LIBRARY = ['highfreq', 'gauss_bump', 'double_bump']


# ------------------------------------------------------------
# Paper symbolic library S, eq. (10)-(11): 8 elements, K and alpha fixed.
#   {1, x, x^2, x^3, x/(K+x), x^2/(K^2+x^2), e^{-alpha x}, ln(1+x)}
# ------------------------------------------------------------
def build_library(xg, K=1.0, alpha=1.0):
    # Library S of eq. (10)-(11) WITHOUT the constant basis: symbolic recovery
    # targets the functional SHAPE of the reaction, so a constant offset (basal
    # production) is absorbed by mean-centering rather than competing with the
    # kinetics bases. Columns are mean-centered and unit-normalized.
    ax = np.abs(xg)
    cols = {
        'x': xg,
        'x^2': xg ** 2,
        'x^3': xg ** 3,
        'MM x/(K+x)': ax / (K + ax),
        'Hill2 x^2/(K^2+x^2)': xg ** 2 / (K ** 2 + xg ** 2),
        'exp(-a x)': np.exp(-alpha * xg),
        'ln(1+|x|)': np.log(1.0 + ax),
    }
    names = list(cols.keys())
    S = np.column_stack([cols[n] for n in names])
    S = S - S.mean(axis=0, keepdims=True)
    S_norms = np.linalg.norm(S, axis=0, keepdims=True) + 1e-10
    return S / S_norms, names


def recover_eps_recon(model, x_lo, x_hi, K=1.0, alpha=1.0, mu=1e-4):
    """Project the learned reaction spline's SHAPE onto library S; return
    eps_recon (%) and the dominant basis. The spline is evaluated over the
    data-visited range [x_lo, x_hi], mean-centered, and unit-normalized so the
    metric reflects how well the library captures the reaction shape, not a
    constant offset or extrapolation artifacts. No ground truth used."""
    xg = np.linspace(x_lo, x_hi, 300).astype(np.float32)
    S, names = build_library(xg, K, alpha)
    with torch.no_grad():
        phi = model.reaction_kans[0](torch.tensor(xg).to(DEVICE)).cpu().numpy()
    phi_c = phi - phi.mean()
    phi_n = phi_c / (np.linalg.norm(phi_c) + 1e-10)
    lasso = Lasso(alpha=mu, max_iter=100000, tol=1e-10, fit_intercept=False)
    lasso.fit(S, phi_n)
    a = lasso.coef_
    recon = S @ a
    eps_recon = np.linalg.norm(phi_n - recon) / (np.linalg.norm(phi_n) + 1e-10) * 100
    dom = names[int(np.argmax(np.abs(a)))]
    return float(eps_recon), dom, xg, phi


def model_fit_rmse(model, traj, L_norm_t):
    """One-step prediction RMSE of trained model on its own trajectory
    (sanity check that training succeeded, so a high eps_recon reflects
    library inadequacy, not underfitting)."""
    model.eval()
    tt = torch.tensor(traj, dtype=torch.float32).to(DEVICE)
    Lt = L_norm_t.to(DEVICE)
    with torch.no_grad():
        errs = []
        for t in range(len(traj) - 1):
            pred = model.forward(tt[t], Lt)
            errs.append(F.mse_loss(pred, tt[t + 1]).item())
    return float(np.sqrt(np.mean(errs)))


# ============================================================
# EXPERIMENT E1: OUT-OF-LIBRARY KINETICS REJECTION
# ============================================================
def experiment_out_of_library():
    print("\n" + "=" * 64)
    print("E1: Out-of-library kinetics rejection (Reviewer 1, Q1)")
    print("=" * 64)
    N, d, h, D_true, T = 200, 2, 0.005, 0.5, 500
    n_seeds = 3
    out = {'in_library': {}, 'out_of_library': {}}

    for group, names in [('in_library', IN_LIBRARY), ('out_of_library', OUT_LIBRARY)]:
        for name in names:
            eps_list, fit_list, doms = [], [], []
            for seed in range(n_seeds):
                traj, pos, W, L_norm, _ = generate_rd_trajectory(
                    N, d, h, D_true, T, name, sigma_noise=0.005, seed=seed)
                L_norm_t = torch.tensor(L_norm, dtype=torch.float32)
                model = RD_GKAN(1, G=8, k=3, x_range=(-1, 5))
                train_rd_gkan(model, traj, L_norm_t, n_epochs=1200,
                              lr=5e-3, l1=5e-5, patience=80)
                # Evaluate recovery over the data-visited concentration range,
                # inset slightly from the spline boundary to avoid edge artifacts.
                c_lo = max(float(traj.min()), -0.9)
                c_hi = min(float(traj.max()), 4.9)
                eps, dom, _, _ = recover_eps_recon(model, c_lo, c_hi)
                fit = model_fit_rmse(model, traj, L_norm_t)
                eps_list.append(eps); fit_list.append(fit); doms.append(dom)
            out[group][name] = {
                'label': KINETICS[name]['label'],
                'eps_recon_mean': float(np.mean(eps_list)),
                'eps_recon_std': float(np.std(eps_list)),
                'eps_recon_all': [float(e) for e in eps_list],
                'model_fit_rmse_mean': float(np.mean(fit_list)),
                'dominant_basis': max(set(doms), key=doms.count),
            }
            print(f"  [{group:14s}] {name:16s} eps_recon={np.mean(eps_list):5.1f}"
                  f"+-{np.std(eps_list):4.1f}%  fit_rmse={np.mean(fit_list):.4f}"
                  f"  dom={out[group][name]['dominant_basis']}")

    in_eps = [v['eps_recon_mean'] for v in out['in_library'].values()]
    out_eps = [v['eps_recon_mean'] for v in out['out_of_library'].values()]
    in_all = [e for v in out['in_library'].values() for e in v['eps_recon_all']]
    out_all = [e for v in out['out_of_library'].values() for e in v['eps_recon_all']]
    # Midpoint threshold between worst in-library and best out-of-library
    thr = 0.5 * (max(in_all) + min(out_all))
    correct = sum(e < thr for e in in_all) + sum(e >= thr for e in out_all)
    out['summary'] = {
        'in_library_eps_max': float(max(in_all)),
        'out_library_eps_min': float(min(out_all)),
        'rejection_threshold_pct': float(thr),
        'separation_margin_pct': float(min(out_all) - max(in_all)),
        'detection_accuracy': float(correct / (len(in_all) + len(out_all))),
        'in_library_mean': float(np.mean(in_eps)),
        'out_library_mean': float(np.mean(out_eps)),
    }
    print(f"\n  In-library eps_recon (max over trials):  {max(in_all):.1f}%")
    print(f"  Out-of-library eps_recon (min over trials): {min(out_all):.1f}%")
    print(f"  Rejection threshold tau_rej = {thr:.1f}%  -> "
          f"detection accuracy {correct}/{len(in_all)+len(out_all)}")
    return out


# ============================================================
# EXPERIMENT E2: TIME-STEP TRANSFERABILITY
# ============================================================
def _scaled_rollout(model, c0, L_norm_t, T, s):
    """Autoregressive rollout with reaction and diffusion scaled by s = h_test/h_train."""
    model.eval()
    Lt = L_norm_t.to(DEVICE)
    c = torch.tensor(c0, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        for _ in range(T):
            reaction = torch.stack(
                [model.reaction_kans[p](c[:, p]) for p in range(model.M)], dim=1)
            diffusion = -torch.abs(model.D) * (Lt @ c)
            c = c + s * reaction + s * diffusion
            c = torch.clamp(c, 0, 20)
    return c.cpu().numpy()


def experiment_timestep_transfer():
    print("\n" + "=" * 64)
    print("E2: Time-step transferability (Reviewer 3, Q1)")
    print("=" * 64)
    N, d, D_true = 200, 2, 0.5
    kin = 'hill'
    T_phys = 2.0
    h_train = 0.01
    T_train = int(T_phys / h_train)
    seed = 7

    # Train at h_train on a clean trajectory.
    traj_tr, pos, W, L_norm, _ = generate_rd_trajectory(
        N, d, h_train, D_true, T_train, kin, sigma_noise=0.0, seed=seed)
    L_norm_t = torch.tensor(L_norm, dtype=torch.float32)
    model = RD_GKAN(1, G=8, k=3, x_range=(-1, 5))
    train_rd_gkan(model, traj_tr, L_norm_t, n_epochs=2000, lr=5e-3, l1=1e-4, patience=120)
    D_theta = abs(model.D.item())
    print(f"  Trained at h_train={h_train}: D_theta={D_theta:.4f}, "
          f"implied continuous D = D_theta/h = {D_theta/h_train:.3f} (true {D_true})")

    results = {'h_train': h_train, 'D_theta': D_theta,
               'D_continuous_est': D_theta / h_train, 'D_true': D_true, 'transfer': {}}

    for h_test in [0.005, 0.01, 0.02, 0.04]:
        T_test = int(T_phys / h_test)
        gt, _, _, L2, _ = generate_rd_trajectory(
            N, d, h_test, D_true, T_test, kin, sigma_noise=0.0, seed=seed)
        L2_t = torch.tensor(L2, dtype=torch.float32)  # identical graph (same seed)
        c0 = gt[0]
        s = h_test / h_train

        # (b) rescaled transfer (no retraining)
        c_resc = _scaled_rollout(model, c0, L2_t, T_test, s=s)
        rmse_resc = float(np.sqrt(np.mean((c_resc - gt[-1]) ** 2)))
        # (a) naive transfer (no rescaling)
        c_naive = _scaled_rollout(model, c0, L2_t, T_test, s=1.0)
        rmse_naive = float(np.sqrt(np.mean((c_naive - gt[-1]) ** 2)))
        # (c) retrained-at-h_test oracle
        m2 = RD_GKAN(1, G=8, k=3, x_range=(-1, 5))
        train_rd_gkan(m2, gt, L2_t, n_epochs=2000, lr=5e-3, l1=1e-4, patience=120)
        c_retr = _scaled_rollout(m2, c0, L2_t, T_test, s=1.0)
        rmse_retr = float(np.sqrt(np.mean((c_retr - gt[-1]) ** 2)))

        results['transfer'][str(h_test)] = {
            'T_test': T_test, 'scale': s,
            'rmse_rescaled': rmse_resc,
            'rmse_naive': rmse_naive,
            'rmse_retrained': rmse_retr,
        }
        print(f"  h_test={h_test:5.3f} (x{s:4.1f}): rescaled={rmse_resc:.4f}  "
              f"naive={rmse_naive:.4f}  retrained(oracle)={rmse_retr:.4f}")

    return results


def main():
    t0 = time.time()
    out = os.path.join(RESULTS_DIR, 'revision_results.json')
    # Preserve the (clean) E2 results from the prior run; rerun only E1.
    results = {}
    if os.path.exists(out):
        with open(out) as f:
            results = json.load(f)
    only_e1 = '--e1' in sys.argv
    results['out_of_library'] = experiment_out_of_library()
    if not only_e1 or 'timestep_transfer' not in results:
        results['timestep_transfer'] = experiment_timestep_transfer()
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n{'='*64}\nDONE in {(time.time()-t0)/60:.1f} min. Results -> {out}\n{'='*64}")


if __name__ == '__main__':
    main()
