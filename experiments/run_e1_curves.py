#!/usr/bin/env python3
"""Dump learned-spline / true-kinetics / library-reconstruction curves for
representative E1 cases, to build the out-of-library mechanism figure.

For each case we save, on the data-visited range, the mean-centered unit-norm
SHAPES of (i) the true kinetics f, (ii) the learned reaction spline psi, and
(iii) the LASSO reconstruction of the learned spline from the library S.

Punchline cases:
  hill        (in-library)      : learned ~ true ~ reconstruction  -> low eps_recon
  gauss_bump  (resolved out)    : learned ~ true bump, recon cannot match -> high eps_recon
  highfreq    (sub-grid out)    : learned is SMOOTH, true oscillates,
                                  recon ~ learned -> low eps_recon despite f not in S
"""
import os, sys, json, warnings, numpy as np, torch
from sklearn.linear_model import Lasso
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_synthetic_rd import RD_GKAN, KINETICS, generate_rd_trajectory, train_rd_gkan, DEVICE, RESULTS_DIR
import run_revision_experiments as R  # registers highfreq, gauss_bump, etc.

np.random.seed(0); torch.manual_seed(0)
CASES = ['hill', 'gauss_bump', 'highfreq']
out = {}
for name in CASES:
    traj, pos, W, L, _ = generate_rd_trajectory(200, 2, 0.005, 0.5, 500, name, sigma_noise=0.005, seed=0)
    Lt = torch.tensor(L, dtype=torch.float32)
    m = RD_GKAN(1, G=8, k=3, x_range=(-1, 5))
    train_rd_gkan(m, traj, Lt, n_epochs=1200, lr=5e-3, l1=5e-5, patience=80)
    lo = max(float(traj.min()), -0.9); hi = min(float(traj.max()), 4.9)
    xg = np.linspace(lo, hi, 200).astype(np.float32)
    S, names = R.build_library(xg)
    with torch.no_grad():
        phi = m.reaction_kans[0](torch.tensor(xg).to(DEVICE)).cpu().numpy()
    # mean-centered unit-norm shapes
    def shp(v):
        v = v - v.mean(); n = np.linalg.norm(v) + 1e-10; return v / n
    phi_s = shp(phi)
    f_true = KINETICS[name]['f'](xg)
    f_s = shp(f_true)
    lasso = Lasso(alpha=1e-4, max_iter=100000, tol=1e-10, fit_intercept=False)
    lasso.fit(S, phi_s)
    recon = S @ lasso.coef_
    eps = float(np.linalg.norm(phi_s - recon) / (np.linalg.norm(phi_s) + 1e-10) * 100)
    # subsample to ~40 points for compact figure coordinates
    idx = np.linspace(0, len(xg) - 1, 40).astype(int)
    out[name] = {
        'x': [round(float(v), 4) for v in xg[idx]],
        'f_true': [round(float(v), 4) for v in f_s[idx]],
        'phi_learned': [round(float(v), 4) for v in phi_s[idx]],
        'recon': [round(float(v), 4) for v in recon[idx]],
        'eps_recon': round(eps, 1),
    }
    print(f'{name:11s} eps_recon={eps:5.1f}%')

with open(os.path.join(RESULTS_DIR, 'e1_curves.json'), 'w') as f:
    json.dump(out, f, indent=1)
print('saved results/e1_curves.json')
