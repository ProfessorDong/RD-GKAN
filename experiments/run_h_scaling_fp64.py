"""Reproduce the published h-scaling (Euler vs RK4) in double precision.

At h=0.001 the Euler-RK4 difference is ~2.5e-5 on states of order 2.5, i.e. ~1e-5
relative -- only ~100x float32 epsilon.  The shared float32 trajectory generator
therefore contaminates the smallest-step point (2.82e-5, slope 0.971).  Evaluating
the same comparison in float64 reproduces results/option_b_results.json exactly
(slope 0.9996).  Numbers in the paper are the float64 values.
"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_synthetic_rd import make_random_geometric_graph, KINETICS

f, D_true, N, d, SIG = KINETICS['hill']['f'], 0.5, 500, 2, 0.15

def euler_vs_rk4(h, seed):
    T = int(2.0 / h)
    rng = np.random.RandomState(seed)
    _, _, L, _, _ = make_random_geometric_graph(N, d, SIG, seed)
    L = L.astype(np.float64)
    c0 = rng.uniform(0.1, 3.0, (N, 1)).astype(np.float64)
    e = c0.copy()
    for _ in range(T):
        e = np.clip(e + h * f(e) - h * D_true * (L @ e), 0, 20)
    r = c0.copy()
    for _ in range(T):
        rhs = lambda cc: h * f(cc) - h * D_true * (L @ cc)
        k1 = rhs(r); k2 = rhs(r + k1 / 2); k3 = rhs(r + k2 / 2); k4 = rhs(r + k3)
        r = np.clip(r + (k1 + 2 * k2 + 2 * k3 + k4) / 6, 0, 20)
    return float(np.sqrt(np.mean((e - r) ** 2)))

if __name__ == '__main__':
    hs = [0.02, 0.01, 0.005, 0.002, 0.001]
    out = {}
    for h in hs:
        v = [euler_vs_rk4(h, s) for s in range(3)]
        out[str(h)] = {'mean': float(np.mean(v)), 'std': float(np.std(v))}
        print(f"  h={h}: {np.mean(v):.6e} +/- {np.std(v):.1e}")
    slope = float(np.polyfit(np.log(hs), np.log([out[str(h)]['mean'] for h in hs]), 1)[0])
    out['exponent'] = slope
    print(f"  fitted exponent: {slope:.5f}")
    json.dump(out, open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'results', 'h_scaling_fp64.json'), 'w'), indent=1)
