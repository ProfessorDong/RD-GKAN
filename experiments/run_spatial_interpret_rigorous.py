#!/usr/bin/env python3
"""
Interpretability artifacts for the static-spatial section, recomputed for the
CONSTRAINED eq.(11) RD-GKAN (the model now used in tab:spatial_results), so the
per-gene symbolic basis and the stability diagnostic are consistent with the
re-run table:
  - per-gene learned reaction splines projected onto the symbolic library
    (epsilon_recon, dominant basis), as in eq:library;
  - local stability spectral radius rho(A) at the data-mean operating point,
    A = I + diag(psi'_p(cbar)) - |D_theta| L_norm.

Model trained as a steady-state reconstruction operator with a 3-way region
split (validation early stopping), identical protocol to run_spatial_rigorous.
Author: Liang Dong
"""
import os, sys, json, warnings, numpy as np, torch, torch.nn.functional as F
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, 'results'); sys.path.insert(0, HERE)
from run_spatial_rigorous import (region_split_3way, W_from_edges, L_norm_phase, DEVICE, SIGMA)
from run_synthetic_rd import RD_GKAN
from run_revised_experiments import construct_graph, load_breast_cancer_data
from run_option_b import load_intestinal_spatial
from torch.optim import Adam


def symbolic_library(x, K):
    """eq:library atoms evaluated on grid x (1D)."""
    cols = {'1': np.ones_like(x), 'x': x, 'x^2': x**2, 'x^3': x**3,
            'MM': x/(K+x+1e-9), 'Hill2': x**2/(K**2+x**2+1e-9),
            'exp': np.exp(-np.abs(x)), 'log': np.log1p(np.abs(x))}
    return cols

def project_spline(phi, x, K):
    """LASSO-free L2 projection with centering/scaling (dominant-basis identification)."""
    from numpy.linalg import lstsq
    cols = symbolic_library(x, K)
    names = list(cols.keys())
    S = np.stack([cols[n] for n in names], 1).astype(np.float64)
    phi_c = phi - phi.mean()
    Sc = S - S.mean(0)
    norms = np.linalg.norm(Sc, axis=0) + 1e-9
    Sn = Sc / norms
    alpha, *_ = lstsq(Sn, phi_c, rcond=None)
    recon = Sn @ alpha
    eps = np.linalg.norm(phi_c - recon) / (np.linalg.norm(phi_c) + 1e-9)
    dom = names[int(np.argmax(np.abs(alpha)))]
    return float(eps), dom

def train_recon(model, x_in, x_tgt, W, tr, va, N, epochs=1000, lr=1e-3, l1=1e-4, patience=60):
    model = model.to(DEVICE)
    xi = torch.tensor(x_in, dtype=torch.float32).to(DEVICE); xt = torch.tensor(x_tgt, dtype=torch.float32).to(DEVICE)
    trm = np.zeros(N, bool); trm[tr] = True; tvm = np.zeros(N, bool); tvm[np.concatenate([tr, va])] = True
    L_tr = L_norm_phase(W, trm, N).to(DEVICE); L_tv = L_norm_phase(W, tvm, N).to(DEVICE)
    tr_t = torch.tensor(tr).to(DEVICE); va_t = torch.tensor(va).to(DEVICE)
    opt = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    best, bs, wait = 1e9, None, 0
    for ep in range(epochs):
        model.train(); pred = model.forward(xi, L_tr)
        loss = F.mse_loss(pred[tr_t], xt[tr_t]) + l1*sum(p.abs().sum() for n,p in model.named_parameters() if 'coeff' in n or 'spline' in n)
        opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = F.mse_loss(model.forward(xi, L_tv)[va_t], xt[va_t]).item()
        if vl < best: best, bs, wait = vl, {k:v.clone() for k,v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience: break
    if bs: model.load_state_dict(bs)
    return model

def analyze(name, pos, expr, genes):
    N, M = expr.shape
    ei, ew, L_mat, eig, stats = construct_graph(pos)
    W = W_from_edges(ei, ew, N)
    torch.manual_seed(0); np.random.seed(0)
    tr, va, te = region_split_3way(pos, seed=200)
    noise = np.random.RandomState(100).randn(*expr.shape).astype(np.float32) * SIGMA
    model = RD_GKAN(M, G=5, k=3, x_range=(-4., 4.))
    model = train_recon(model, expr + noise, expr, W, tr, va, N)
    model.eval()
    # per-gene symbolic projection over data-supported range
    eps_list, dom_list = [], []
    for p in range(M):
        lo, hi = np.percentile(expr[:, p], [2, 98])
        xg = np.linspace(lo, hi, 200).astype(np.float32)
        with torch.no_grad():
            phi = model.reaction_kans[p](torch.tensor(xg).to(DEVICE)).cpu().numpy()
        K = float(np.median(np.abs(expr[:, p]))) + 0.5
        eps, dom = project_spline(phi.astype(np.float64), xg.astype(np.float64), K)
        eps_list.append(eps); dom_list.append(dom)
    # stability at data mean
    cbar = expr.mean(0)
    h = 1e-3; slopes = []
    for p in range(M):
        with torch.no_grad():
            xp = torch.tensor([cbar[p]+h], dtype=torch.float32).to(DEVICE)
            xm = torch.tensor([cbar[p]-h], dtype=torch.float32).to(DEVICE)
            sp = (model.reaction_kans[p](xp).item() - model.reaction_kans[p](xm).item())/(2*h)
        slopes.append(sp)
    D = float(abs(model.rd.D.item())) if hasattr(model,'rd') else float(abs(model.D.item()))
    Ln = (np.diag(W.sum(1)) - W)/N
    M_eye = np.eye(N)
    # block per feature: a_{p,i} = 1 + mu_p - D*lambda_i ; rho = max over p,i
    lam = np.linalg.eigvalsh(Ln)
    amax = 0.0
    for mu in slopes:
        vals = np.abs(1 + mu - D*lam)
        amax = max(amax, vals.max())
    return {'N': N, 'M': M, 'genes': list(genes), 'D_theta': D,
            'reaction_slopes_at_mean': [float(s) for s in slopes],
            'eps_recon_per_gene': eps_list, 'dominant_basis_per_gene': dom_list,
            'eps_recon_min': float(min(eps_list)), 'eps_recon_max': float(max(eps_list)),
            'eps_recon_mean': float(np.mean(eps_list)),
            'n_linear_dominant': int(sum(d == 'x' for d in dom_list)),
            'rho_A': float(amax)}


def main():
    print(f"Device: {DEVICE}")
    out = {}
    pos_b, expr_b, genes_b = load_breast_cancer_data(n_spots=500, seed=42)
    out['breast'] = analyze('breast', pos_b, expr_b, genes_b)
    pos_i, expr_i, genes_i = load_intestinal_spatial()
    out['intestine'] = analyze('intestine', pos_i, expr_i, genes_i)
    with open(os.path.join(RES, 'spatial_interpret_rigorous.json'), 'w') as f:
        json.dump(out, f, indent=2)
    for t in out:
        d = out[t]
        print(f"\n{t}: D_theta={d['D_theta']:.4f}, rho(A)={d['rho_A']:.4f}, "
              f"eps_recon={d['eps_recon_min']*100:.1f}-{d['eps_recon_max']*100:.1f}% "
              f"(mean {d['eps_recon_mean']*100:.1f}%), linear-dominant {d['n_linear_dominant']}/{d['M']}")
        print(f"   dominant basis: {d['dominant_basis_per_gene']}")
    print("\nsaved results/spatial_interpret_rigorous.json")


if __name__ == '__main__':
    main()
