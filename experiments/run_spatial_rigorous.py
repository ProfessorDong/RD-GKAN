#!/usr/bin/env python3
"""
RIGOROUS re-run of the static-spatial reconstruction experiments
(tab:spatial_results + masking + static baselines), fixing two issues found in
the prior pipeline:

  (1) TEST LEAKAGE: the old train_model selected/early-stopped the checkpoint on
      *test* loss and then reported it. Here we use a 3-way region split
      (train/val/test); early stopping and checkpoint selection use the
      *validation* set only, and the test RMSE is computed ONCE at the end with
      the validation-selected model.

  (2) MODEL IDENTITY: the old spatial table used CrossFeatureGraphKAN (KAN
      messages on neighbours + cross-feature mixing), which is NOT the constrained
      eq.(11) RD-GKAN the paper describes. Here RD-GKAN / Self-only use the
      constrained RD_GKAN (feature-wise B-spline reaction + explicit Laplacian
      diffusion D_theta * L_norm), identical to the wound-table model.

LEAKAGE-PROOF INDUCTIVE GRAPH: during each phase, message passing is restricted
to edges whose BOTH endpoints are in the active node set (train; then train+val;
then full at test). The 1/N normalisation of L is kept at the FULL N so the
learned D_theta is on a consistent scale across phases. Training never sees
val/test nodes in its computation graph.

Edge weights are the paper's CIR transfer efficiencies (construct_graph).
Self-only = constrained RD-GKAN with D_theta frozen to 0 (no graph coupling).

Author: Liang Dong
"""
import os, sys, json, warnings, numpy as np, torch, torch.nn.functional as F
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, 'results'); os.makedirs(RES, exist_ok=True)
sys.path.insert(0, HERE)

from run_synthetic_rd import RD_GKAN
from run_revised_experiments import (
    GNNBaseline, GATBaseline, GRANDBaseline, GREADBaseline,
    load_breast_cancer_data, construct_graph,
)
from run_option_b import load_intestinal_spatial
from sklearn.cluster import KMeans
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_SEEDS = 5
SIGMA = 0.3
GRAPH_MODELS = ['RD-GKAN', 'GNN', 'GAT', 'GRAND', 'GREAD']   # use the graph
ALL_MODELS = ['RD-GKAN', 'Self-only', 'GNN', 'GAT', 'GRAND', 'GREAD']


# ---------- 3-way region split (whole k-means regions) ----------
def region_split_3way(positions, seed, n_clusters=10, val_frac=0.2, test_frac=0.2):
    N = len(positions)
    labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(positions)
    sizes = np.array([(labels == c).sum() for c in range(n_clusters)])
    order = np.random.RandomState(seed).permutation(n_clusters)
    test_c, val_c, n_test, n_val = [], [], 0, 0
    for c in order:                       # fill test regions first, then val
        if n_test < N * test_frac:
            test_c.append(c); n_test += sizes[c]
        elif n_val < N * val_frac:
            val_c.append(c); n_val += sizes[c]
    test_mask = np.isin(labels, test_c)
    val_mask = np.isin(labels, val_c)
    train_mask = ~(test_mask | val_mask)
    return (np.where(train_mask)[0], np.where(val_mask)[0], np.where(test_mask)[0])


# ---------- inductive graph restriction ----------
def W_from_edges(edge_index, edge_weight, N):
    W = np.zeros((N, N), dtype=np.float32)
    ei = edge_index.numpy(); ew = edge_weight.numpy()
    for e in range(ei.shape[1]):
        W[ei[0, e], ei[1, e]] = ew[e]
    return W

def L_norm_phase(W, active_mask, N_full):
    """Normalised Laplacian (D-W)/N_full using only edges among active nodes."""
    m = active_mask.astype(np.float32)
    Wp = W * np.outer(m, m)                      # zero edges touching inactive nodes
    Lp = (np.diag(Wp.sum(1)) - Wp) / N_full
    return torch.tensor(Lp, dtype=torch.float32)

def edges_phase(edge_index, edge_weight, active_mask):
    """Keep edges with both endpoints active."""
    ei = edge_index.numpy()
    keep = active_mask[ei[0]] & active_mask[ei[1]]
    return (torch.tensor(ei[:, keep], dtype=torch.long),
            edge_weight[torch.tensor(keep)])


# ---------- unified predict ----------
def predict(model, is_rd, x, L_phase, ei_phase, ew_phase):
    if is_rd:
        return model.forward(x, L_phase)          # RD_GKAN: x + psi(x) + D*L@x
    return model.predict(x, ei_phase, ew_phase)


# ---------- rigorous training (val early-stop, single test) ----------
def train_rigorous(model, is_rd, x_in, x_tgt, W, edge_index, edge_weight,
                   tr, va, te, N, n_epochs=1000, lr=1e-3, l1=1e-4, patience=60):
    model = model.to(DEVICE)
    x_in = torch.tensor(x_in, dtype=torch.float32).to(DEVICE)
    x_tgt = torch.tensor(x_tgt, dtype=torch.float32).to(DEVICE)
    tr_t = torch.tensor(tr).to(DEVICE); va_t = torch.tensor(va).to(DEVICE); te_t = torch.tensor(te).to(DEVICE)

    train_mask = np.zeros(N, bool); train_mask[tr] = True
    trval_mask = np.zeros(N, bool); trval_mask[np.concatenate([tr, va])] = True
    full_mask = np.ones(N, bool)

    # phase graphs
    if is_rd:
        L_tr = L_norm_phase(W, train_mask, N).to(DEVICE)
        L_trval = L_norm_phase(W, trval_mask, N).to(DEVICE)
        L_full = L_norm_phase(W, full_mask, N).to(DEVICE)
        g_tr = (L_tr, None, None); g_va = (L_trval, None, None); g_te = (L_full, None, None)
    else:
        ei_tr, ew_tr = edges_phase(edge_index, edge_weight, train_mask)
        ei_tv, ew_tv = edges_phase(edge_index, edge_weight, trval_mask)
        g_tr = (None, ei_tr.to(DEVICE), ew_tr.to(DEVICE))
        g_va = (None, ei_tv.to(DEVICE), ew_tv.to(DEVICE))
        g_te = (None, edge_index.to(DEVICE), edge_weight.to(DEVICE))

    opt = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-5)
    best_val, best_state, wait = float('inf'), None, 0
    for ep in range(n_epochs):
        model.train()
        pred = predict(model, is_rd, x_in, *g_tr)
        loss = F.mse_loss(pred[tr_t], x_tgt[tr_t])
        if l1 > 0:
            loss = loss + l1 * sum(p.abs().sum() for n, p in model.named_parameters()
                                   if ('coeff' in n or 'spline' in n))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()
        model.eval()
        with torch.no_grad():
            vp = predict(model, is_rd, x_in, *g_va)
            vloss = F.mse_loss(vp[va_t], x_tgt[va_t]).item()
        if vloss < best_val:
            best_val = vloss; best_state = {k: v.clone() for k, v in model.state_dict().items()}; wait = 0
        else:
            wait += 1
            if wait >= patience: break
    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        tp = predict(model, is_rd, x_in, *g_te)
        test_rmse = torch.sqrt(F.mse_loss(tp[te_t], x_tgt[te_t])).item()
        val_rmse = torch.sqrt(torch.tensor(best_val)).item()
    return test_rmse, val_rmse


def make_model(name, M):
    if name in ('RD-GKAN', 'Self-only'):
        m = RD_GKAN(M, G=5, k=3, x_range=(-4.0, 4.0))
        if name == 'Self-only':
            m.D.data.zero_(); m.D.requires_grad_(False)
        return m, True
    return {'GNN': GNNBaseline(M, 64, 2), 'GAT': GATBaseline(M, 64, 4),
            'GRAND': GRANDBaseline(M, 64, 2), 'GREAD': GREADBaseline(M, 64, 2)}[name], False


def static_baselines(x_noisy, x_clean, W, te):
    """noisy-identity and neighbour-average (full graph) test RMSE."""
    ni = np.sqrt(np.mean((x_noisy[te] - x_clean[te]) ** 2))
    deg = W.sum(1, keepdims=True); deg[deg == 0] = 1.0
    nbr = (W @ x_noisy) / deg                      # neighbour average of noisy inputs
    na = np.sqrt(np.mean((nbr[te] - x_clean[te]) ** 2))
    return float(ni), float(na)


def run_tissue(name, pos, expr, genes, masked=False):
    N, M = expr.shape
    ei, ew, L_mat, eigvals, stats = construct_graph(pos)
    W = W_from_edges(ei, ew, N)
    print(f"\n=== {name} | N={N} M={M} |E|={stats['n_edges_undirected']} "
          f"deg={stats['avg_degree']:.2f} masked={masked} ===")
    models = GRAPH_MODELS if masked else ALL_MODELS
    res = {m: {'test': [], 'val': []} for m in models}
    sb = {'noisy_identity': [], 'neighbor_avg': []}
    for s in range(N_SEEDS):
        torch.manual_seed(s); np.random.seed(s)
        tr, va, te = region_split_3way(pos, seed=s + 200)
        noise = np.random.RandomState(s + 100).randn(*expr.shape).astype(np.float32) * SIGMA
        x_noisy = expr + noise
        x_in = np.zeros_like(expr) if masked else x_noisy
        if not masked:
            ni, na = static_baselines(x_noisy, expr, W, te)
            sb['noisy_identity'].append(ni); sb['neighbor_avg'].append(na)
        for m in models:
            model, is_rd = make_model(m, M)
            l1 = 1e-4 if m in ('RD-GKAN', 'Self-only') else 0.0
            tr_rmse, va_rmse = train_rigorous(model, is_rd, x_in, expr, W, ei, ew,
                                              tr, va, te, N, n_epochs=1000, lr=1e-3,
                                              l1=l1, patience=60)
            res[m]['test'].append(tr_rmse); res[m]['val'].append(va_rmse)
        print(f"  seed{s} split tr/va/te={len(tr)}/{len(va)}/{len(te)} | " +
              ", ".join(f"{m}={res[m]['test'][-1]:.4f}" for m in models))
    out = {'N': N, 'M': M, 'graph': stats, 'genes': list(genes)}
    for m in models:
        v = res[m]['test']
        out[m] = {'mean': float(np.mean(v)), 'std': float(np.std(v)), 'all': [float(x) for x in v]}
    if not masked:
        for b in sb:
            out[b] = {'mean': float(np.mean(sb[b])), 'std': float(np.std(sb[b])), 'all': sb[b]}
    return out


def main():
    print(f"Device: {DEVICE}")
    pos_b, expr_b, genes_b = load_breast_cancer_data(n_spots=500, seed=42)
    pos_i, expr_i, genes_i = load_intestinal_spatial()
    results = {'_meta': {'n_seeds': N_SEEDS, 'sigma': SIGMA, 'device': str(DEVICE),
                         'model': 'constrained RD_GKAN (eq.11) for RD-GKAN/Self-only',
                         'protocol': '3-way region split (~60/20/20 whole k-means regions); '
                                     'INDUCTIVE (edges among active set only, /N_full); '
                                     'early-stop+checkpoint on VALIDATION; test computed ONCE; '
                                     'CIR edge weights; noise sigma=0.3; '
                                     'Self-only=RD_GKAN with D_theta frozen 0'}}
    results['unmasked'] = {
        'breast': run_tissue('breast', pos_b, expr_b, genes_b, masked=False),
        'intestine': run_tissue('intestine', pos_i, expr_i, genes_i, masked=False)}
    results['masked'] = {
        'breast': run_tissue('breast', pos_b, expr_b, genes_b, masked=True),
        'intestine': run_tissue('intestine', pos_i, expr_i, genes_i, masked=True)}
    with open(os.path.join(RES, 'spatial_rigorous.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("\nsaved results/spatial_rigorous.json")
    # summary
    print("\n==== UNMASKED test RMSE (mean+/-std, 5 seeds, val-selected, inductive) ====")
    for t in ('breast', 'intestine'):
        d = results['unmasked'][t]
        print(f"\n{t}:")
        for m in ALL_MODELS + ['noisy_identity', 'neighbor_avg']:
            if m in d: print(f"   {m:<14} {d[m]['mean']:.3f} +/- {d[m]['std']:.3f}")
    print("\n==== MASKED test RMSE ====")
    for t in ('breast', 'intestine'):
        d = results['masked'][t]
        print(f"{t}: " + ", ".join(f"{m}={d[m]['mean']:.3f}" for m in GRAPH_MODELS))


if __name__ == '__main__':
    main()
