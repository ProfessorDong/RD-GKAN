#!/usr/bin/env python3
"""Complete the T-scaling verification: michaelis_menten (all T) and the
linear cells the interrupted sequential run did not reach (T=200,500,1000).

Author: Liang Dong
"""
import os, sys, json, time, warnings, numpy as np, torch
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from run_synthetic_rd import (generate_rd_trajectory, RD_GKAN, train_rd_gkan,
                              symbolic_recovery)
OUT = os.path.join(ROOT, 'results', 't_scaling_verify_rest.json')
N, d, h, D_true, n_seeds = 200, 2, 0.005, 0.5, 5
TODO = [('michaelis_menten', T) for T in [20, 50, 100, 200, 500, 1000]] + \
       [('linear', T) for T in [200, 500, 1000]]
res = {'_meta': {'source': 'completes experiment_t_scaling()', 'N': N,
                 'n_seeds': n_seeds, 'G': 10}}
t0 = time.time()
for kin, T in TODO:
    eps, ok = [], 0
    for seed in range(n_seeds):
        traj, pos, W, L_norm, _ = generate_rd_trajectory(
            N, d, h, D_true, T, kin, sigma_noise=0.005, seed=seed)
        Lt = torch.tensor(L_norm, dtype=torch.float32)
        m = RD_GKAN(1, G=10, k=3, x_range=(-1, 5))
        train_rd_gkan(m, traj, Lt, n_epochs=2000, lr=5e-3, l1=5e-5, patience=100)
        s0 = symbolic_recovery(m, kin, M=1)['species_0']
        eps.append(s0['eps_sym']); ok += bool(s0['support_correct'])
    res.setdefault(kin, {})[str(T)] = {
        'mean_eps_sym': float(np.mean(eps)), 'std_eps_sym': float(np.std(eps)),
        'support_recovery_rate': ok / n_seeds, 'support_correct_count': ok}
    print(f"{kin:18s} T={T:5d}  eps_sym={np.mean(eps):6.1f}%  support={ok}/{n_seeds}"
          f"  [{time.time()-t0:6.0f}s]", flush=True)
    json.dump(res, open(OUT, 'w'), indent=2)
print(f"DONE in {time.time()-t0:.0f}s -> {OUT}", flush=True)
