#!/usr/bin/env python3
"""Re-derive the T-scaling support-recovery rates cited in Sec. VI-A.
The original synthetic_results.json is not on disk; this reproduces
experiment_t_scaling() verbatim (T in {20,50,100,200,500,1000}, N=200,
3 kinetics, 5 seeds, G=10) and writes results incrementally.

Author: Liang Dong
"""
import os, sys, json, time, warnings, numpy as np, torch
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from run_synthetic_rd import (generate_rd_trajectory, RD_GKAN, train_rd_gkan,
                              symbolic_recovery)

OUT = os.path.join(ROOT, 'results', 't_scaling_verify_hill.json')
T_VALUES = [20, 50, 100, 200, 500, 1000]
N, d, h, D_true, n_seeds = 200, 2, 0.005, 0.5, 5
res = {'_meta': {'source': 'verbatim re-run of experiment_t_scaling()',
                 'T_values': T_VALUES, 'N': N, 'n_seeds': n_seeds, 'G': 10}}
t_start = time.time()
for kin in ['hill']:
    res[kin] = {}
    for T in T_VALUES:
        eps, ok = [], 0
        for seed in range(n_seeds):
            traj, pos, W, L_norm, _ = generate_rd_trajectory(
                N, d, h, D_true, T, kin, sigma_noise=0.005, seed=seed)
            Lt = torch.tensor(L_norm, dtype=torch.float32)
            m = RD_GKAN(1, G=10, k=3, x_range=(-1, 5))
            train_rd_gkan(m, traj, Lt, n_epochs=2000, lr=5e-3, l1=5e-5, patience=100)
            s0 = symbolic_recovery(m, kin, M=1)['species_0']
            eps.append(s0['eps_sym']); ok += bool(s0['support_correct'])
        res[kin][str(T)] = {'mean_eps_sym': float(np.mean(eps)),
                            'std_eps_sym': float(np.std(eps)),
                            'support_recovery_rate': ok / n_seeds,
                            'support_correct_count': ok}
        print(f"{kin:18s} T={T:5d}  eps_sym={np.mean(eps):6.1f}%  "
              f"support={ok}/{n_seeds}  [{time.time()-t_start:7.0f}s]", flush=True)
        json.dump(res, open(OUT, 'w'), indent=2)
for kin in ['hill']:
    rates = [res[kin][str(T)]['support_recovery_rate'] for T in T_VALUES]
    print(f"SUMMARY {kin:18s} rates={rates} min={min(rates):.0%} max={max(rates):.0%}", flush=True)
print(f"DONE in {time.time()-t_start:.0f}s -> {OUT}", flush=True)
