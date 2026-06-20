#!/usr/bin/env python3
"""Extract REAL learned reaction splines from the constrained RD-GKAN on the
GSE241124 wound-healing spatial transcriptomics data (4 conditions), to replace
the schematic spline figure with data-driven curves.

Rigorous protocol:
  * Real Visium coordinates (spatial/tissue_positions_list.csv) matched to the
    filtered expression matrix by barcode (in_tissue spots only).
  * Common gene set: the 4 conditions share the same Visium reference (identical
    gene order); the target gene is the most variable gene COMMON to all four
    conditions, ranked by mean Fano factor (genes with raw mean < 0.5 excluded
    in every condition). The SAME M=8 genes (same target) are used in all
    conditions so the per-gene reaction splines are directly comparable.
  * Per condition: 500 spots (seed 42), log1p + per-condition standardization,
    k-NN graph (k=6), constrained RD-GKAN (eq. 11; feature-wise B-spline psi_p
    per gene + Laplacian diffusion) trained as a STEADY-STATE RECONSTRUCTION
    operator: psi_theta minimizes ||RD-GKAN(c+eps) - c||^2 (input noise sigma=0.3).
  * The learned reaction spline psi_g of the TARGET gene is evaluated over its
    data-supported standardized range; mean +/- std across 5 seeds gives the band.
"""
import os, sys, json, warnings, zipfile, io, numpy as np
import h5py, scipy.sparse
import torch, torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_synthetic_rd import RD_GKAN, DEVICE, DATA_DIR, RESULTS_DIR
from run_new_datasets import build_knn_graph

WH = os.path.join(DATA_DIR, 'WoundHealing_GSE241124')
COND = [('Skin', 'GSM7717016_Skin'), ('Wound d1', 'GSM7717017_Wound1'),
        ('Wound d7', 'GSM7717018_Wound7'), ('Wound d30', 'GSM7717019_Wound30')]
M, N_MAX, N_SEEDS, SIGMA = 8, 500, 5, 0.3


def load_condition(prefix):
    """Return (expr [spots x genes], gene_names, positions [spots x 2]) with real coords."""
    h5 = os.path.join(WH, prefix + '_filtered_feature_bc_matrix.h5')
    with h5py.File(h5, 'r') as f:
        m = f['matrix']
        expr = scipy.sparse.csc_matrix(
            (np.array(m['data']), np.array(m['indices']), np.array(m['indptr'])),
            shape=np.array(m['shape'])).toarray().T.astype(np.float32)  # (spots, genes)
        barcodes = [b.decode() for b in m['barcodes']]
        genes = [g.decode() for g in m['features']['name']]
    # real coordinates from the spatial zip
    zf = os.path.join(WH, prefix + '_spatial_images.zip')
    pos_map = {}
    with zipfile.ZipFile(zf) as z:
        name = [n for n in z.namelist() if n.endswith('tissue_positions_list.csv')
                and not n.startswith('__MACOSX')][0]
        for line in io.TextIOWrapper(z.open(name), 'utf-8'):
            p = line.strip().split(',')
            if len(p) >= 6 and p[1] == '1':  # in_tissue
                pos_map[p[0]] = (float(p[4]), float(p[5]))  # pxl_row, pxl_col
    keep = [i for i, b in enumerate(barcodes) if b in pos_map]
    expr = expr[keep]
    positions = np.array([pos_map[barcodes[i]] for i in keep], dtype=np.float32)
    return expr, genes, positions


def train_reconstruction(model, x_clean, L_norm_t, seed, n_epochs=900, lr=5e-3, l1=1e-4, patience=80):
    """Train constrained RD-GKAN as a steady-state reconstruction operator."""
    torch.manual_seed(seed)
    g = torch.Generator(device='cpu').manual_seed(seed)
    model = model.to(DEVICE)
    Xc = torch.tensor(x_clean, dtype=torch.float32).to(DEVICE)
    Lt = L_norm_t.to(DEVICE)
    opt = Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    sched = CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-6)
    best, best_state, wait = float('inf'), None, 0
    for ep in range(n_epochs):
        model.train()
        noise = torch.randn(Xc.shape, generator=g).to(DEVICE) * SIGMA
        pred = model.forward(Xc + noise, Lt)
        loss = F.mse_loss(pred, Xc)
        if l1 > 0:
            loss = loss + l1 * sum(p.abs().sum() for n, p in model.named_parameters() if 'coeffs' in n)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        lv = loss.item()
        if lv < best - 1e-7:
            best, wait = lv, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return best


def main():
    # ---- load all conditions, select common target gene ----
    data, genes_ref = {}, None
    fano_stack = []
    for label, prefix in COND:
        expr, genes, pos = load_condition(prefix)
        if genes_ref is None:
            genes_ref = genes
        assert genes == genes_ref, "gene order differs across conditions"
        mean = expr.mean(0) + 1e-10
        fano = expr.var(0) / mean
        fano[mean < 0.5] = 0.0          # exclude lowly-expressed genes
        fano_stack.append(fano)
        data[label] = (expr, pos)
        print(f"{label}: {expr.shape[0]} spots, {expr.shape[1]} genes")
    fano_stack = np.array(fano_stack)
    common_ok = (fano_stack > 0).all(0)         # expressed & variable in ALL conditions
    mean_fano = fano_stack.mean(0)
    mean_fano[~common_ok] = -1
    top = np.argsort(mean_fano)[-M:][::-1]
    gene_sel = [genes_ref[i] for i in top]
    target_gene = gene_sel[0]
    print(f"\nCommon HVGs (mean-Fano ranked): {gene_sel}")
    print(f"TARGET gene = {target_gene}\n")

    out = {'target_gene': target_gene, 'gene_set': gene_sel, 'M': M,
           'n_seeds': N_SEEDS, 'noise_sigma': SIGMA, 'conditions': {}}

    for label, prefix in COND:
        expr, pos = data[label]
        N = expr.shape[0]
        rng = np.random.RandomState(42)
        if N > N_MAX:
            idx = rng.choice(N, N_MAX, replace=False)
            expr, pos = expr[idx], pos[idx]
            N = N_MAX
        x = np.log1p(expr[:, top].astype(np.float32))
        x = (x - x.mean(0)) / (x.std(0) + 1e-10)      # per-condition standardization
        L_norm, W, avg_deg = build_knn_graph(pos, k=min(6, N - 1))
        L_t = torch.tensor(L_norm, dtype=torch.float32)
        # data-supported range of the TARGET gene (2nd-98th percentile)
        lo, hi = np.percentile(x[:, 0], [2, 98])
        xg = np.linspace(float(lo), float(hi), 60).astype(np.float32)
        curves, fits = [], []
        for s in range(N_SEEDS):
            model = RD_GKAN(M, G=5, k=3, x_range=(-4.0, 4.0))
            fit = train_reconstruction(model, x, L_t, seed=s)
            with torch.no_grad():
                psi = model.reaction_kans[0](torch.tensor(xg).to(DEVICE)).cpu().numpy()
            curves.append(psi); fits.append(fit)
        curves = np.array(curves)
        out['conditions'][label] = {
            'x': [round(float(v), 4) for v in xg],
            'psi_mean': [round(float(v), 5) for v in curves.mean(0)],
            'psi_std': [round(float(v), 5) for v in curves.std(0)],
            'N': int(N), 'avg_degree': round(float(avg_deg), 2),
            'fit_rmse': round(float(np.mean(fits)), 4),
            'support': [round(float(lo), 3), round(float(hi), 3)],
        }
        print(f"{label}: N={N}, deg={avg_deg:.2f}, fit_rmse={np.mean(fits):.4f}, "
              f"psi range [{curves.mean(0).min():.3f}, {curves.mean(0).max():.3f}]")

    with open(os.path.join(RESULTS_DIR, 'wound_splines.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(f"\nsaved results/wound_splines.json (target gene: {target_gene})")


if __name__ == '__main__':
    main()
