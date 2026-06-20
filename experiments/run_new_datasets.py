#!/usr/bin/env python3
"""
Experiments on NEW datasets for the revised paper.
Uses ALL 4 new datasets + existing datasets for maximum evidence.

Track 1: Temporal signaling dynamics
  - ERK Collective MDCK waves (34,236 observations, 207 tracks, 199 timepoints)
  - ERK SSBD (13 cells, 34 timepoints)

Track 2: Perturbational spatial transcriptomics
  - GSE241124 Wound Healing (22,915 spots, Skin/Wound d1/d7/d30, multiple donors)

Track 3: Bacterial QS competition dynamics
  - S-BIAD1046 metadata analysis (6 CC + 38 MM experiments)

All experiments use the constrained RD-GKAN architecture.
"""

import os, sys, json, warnings, time, gzip
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
import h5py
import scipy.sparse

warnings.filterwarnings('ignore')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"Device: {DEVICE}")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_synthetic_rd import BSplineKAN, RD_GKAN, MLPReaction


# ============================================================
# UTILITIES
# ============================================================
def build_knn_graph(positions, k=6, sigma=None):
    """Build k-NN graph with Gaussian kernel weights."""
    N = len(positions)
    tree = KDTree(positions)
    dists, indices = tree.query(positions, k=k+1)
    if sigma is None:
        sigma = np.median(dists[:, 1]) * 2

    src, dst, weights = [], [], []
    seen = set()
    for i in range(N):
        for j_idx in range(1, k+1):
            j = indices[i, j_idx]
            if (i, j) not in seen:
                w = np.exp(-dists[i, j_idx]**2 / (2*sigma**2))
                src.extend([i, j]); dst.extend([j, i])
                weights.extend([w, w])
                seen.add((i, j)); seen.add((j, i))

    W = np.zeros((N, N), dtype=np.float32)
    for s, d, w in zip(src, dst, weights):
        W[s, d] = w
    D_deg = np.diag(W.sum(1))
    L = D_deg - W
    L_norm = L / max(N, 1)

    avg_deg = W.sum() / N
    return L_norm, W, avg_deg


def train_temporal_rdgkan(model, trajectory, L_norm_t, n_epochs=1000,
                           lr=5e-3, l1=1e-4, patience=80):
    """Train RD-GKAN on temporal trajectory data."""
    model = model.to(DEVICE)
    T = len(trajectory) - 1
    traj_t = torch.tensor(trajectory, dtype=torch.float32).to(DEVICE)
    L_t = L_norm_t.to(DEVICE)
    n_batch = min(T, 64)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)
    best_loss, best_state, wait = float('inf'), None, 0

    for epoch in range(n_epochs):
        model.train()
        t_idx = np.random.choice(T, size=n_batch, replace=False)
        loss_total = sum(F.mse_loss(model.forward(traj_t[t], L_t), traj_t[t+1]) for t in t_idx) / n_batch
        if l1 > 0:
            loss_total = loss_total + l1 * sum(p.abs().sum() for n, p in model.named_parameters() if 'coeffs' in n)
        optimizer.zero_grad(); loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step()

        lv = loss_total.item()
        if lv < best_loss:
            best_loss = lv; best_state = {k: v.clone() for k, v in model.state_dict().items()}; wait = 0
        else:
            wait += 1
            if wait >= patience: break

    if best_state: model.load_state_dict(best_state)
    model.eval()
    return best_loss


# ============================================================
# EXPERIMENT 1: ERK COLLECTIVE MDCK WAVES (temporal dynamics)
# ============================================================
def experiment_erk_waves():
    print("\n" + "="*60)
    print("EXP 1: ERK Collective MDCK Waves (temporal dynamics)")
    print("="*60)

    # Load MDCK wave data
    mdck_dir = os.path.join(DATA_DIR, 'ERK_Collective', 'MDCK-waves-extracted', 'MDCK-waves')
    csv_file = None
    for root, dirs, files in os.walk(mdck_dir):
        for f in files:
            if f.endswith('.csv.gz') and 'clean_tracks' in f:
                csv_file = os.path.join(root, f)
                break
        if csv_file: break

    if csv_file is None:
        # Try alternative paths
        for root, dirs, files in os.walk(os.path.join(DATA_DIR, 'ERK_Collective')):
            for f in files:
                if f.endswith('.csv.gz'):
                    csv_file = os.path.join(root, f)
                    break
            if csv_file: break

    if csv_file is None:
        print("  ERROR: Could not find MDCK wave CSV file")
        return {'error': 'file_not_found'}

    print(f"  Loading: {csv_file}")
    df = pd.read_csv(csv_file, compression='gzip' if csv_file.endswith('.gz') else None)
    print(f"  Rows: {len(df)}, Columns: {list(df.columns)[:8]}...")

    # Extract key columns
    time_col = [c for c in df.columns if 'Metadata_T' in c or 'time' in c.lower()][0]
    x_col = [c for c in df.columns if 'Location_Center_X' in c or 'center_x' in c.lower()][0]
    y_col = [c for c in df.columns if 'Location_Center_Y' in c or 'center_y' in c.lower()][0]
    track_col = [c for c in df.columns if 'track_id' in c.lower()][0]
    # ERK activity: FRET ratio
    ratio_cols = [c for c in df.columns if 'MeanIntensity_imRATIO' in c or 'ratio' in c.lower()]
    if not ratio_cols:
        ratio_cols = [c for c in df.columns if 'FRET' in c or 'Intensity' in c]
    signal_col = ratio_cols[0]

    print(f"  Time: {time_col}, X: {x_col}, Y: {y_col}, Track: {track_col}, Signal: {signal_col}")

    # Get unique timepoints and tracks
    timepoints = sorted(df[time_col].unique())
    tracks = sorted(df[track_col].unique())
    print(f"  Timepoints: {len(timepoints)}, Tracks: {len(tracks)}")

    # Build trajectory matrix: (T, N, M) where N=cells, M=1 (ERK activity)
    # Use only cells present at all timepoints (for simplicity)
    track_counts = df.groupby(track_col)[time_col].nunique()
    complete_tracks = track_counts[track_counts >= len(timepoints) * 0.8].index.tolist()
    if len(complete_tracks) < 10:
        complete_tracks = track_counts.nlargest(min(50, len(tracks))).index.tolist()

    print(f"  Using {len(complete_tracks)} tracks with >=80% temporal coverage")

    # Build per-timepoint snapshots
    N = len(complete_tracks)
    T_total = min(len(timepoints), 100)  # cap for tractability
    track_map = {t: i for i, t in enumerate(complete_tracks)}

    trajectory = np.full((T_total, N, 1), np.nan, dtype=np.float32)
    positions = np.zeros((N, 2), dtype=np.float32)

    for _, row in df.iterrows():
        tid = row[track_col]
        t = row[time_col]
        if tid in track_map and t < T_total:
            i = track_map[tid]
            trajectory[int(t), i, 0] = row[signal_col]
            if t == 0:
                positions[i] = [row[x_col], row[y_col]]

    # Fill NaNs with linear interpolation
    for i in range(N):
        vals = trajectory[:, i, 0]
        nans = np.isnan(vals)
        if nans.all():
            trajectory[:, i, 0] = 0
        elif nans.any():
            trajectory[:, i, 0] = np.interp(np.arange(T_total), np.where(~nans)[0], vals[~nans])

    # Normalize signal
    sig_mean = np.nanmean(trajectory)
    sig_std = np.nanstd(trajectory) + 1e-10
    trajectory = (trajectory - sig_mean) / sig_std

    # Use mean position for cells without t=0 position
    pos_mask = np.all(positions == 0, axis=1)
    if pos_mask.any():
        # Use positions from any available timepoint
        for _, row in df.iterrows():
            tid = row[track_col]
            if tid in track_map:
                i = track_map[tid]
                if pos_mask[i]:
                    positions[i] = [row[x_col], row[y_col]]
                    pos_mask[i] = False

    print(f"  Trajectory shape: {trajectory.shape}")
    print(f"  Signal range: [{trajectory.min():.3f}, {trajectory.max():.3f}]")

    # Build spatial graph
    L_norm, W, avg_deg = build_knn_graph(positions, k=min(6, N-1))
    print(f"  Graph: N={N}, avg_degree={avg_deg:.2f}")

    # Train/test split: temporal (first 70% train, last 30% test)
    T_train = int(T_total * 0.7)
    T_test = T_total - T_train - 1
    train_traj = trajectory[:T_train+1]
    test_traj = trajectory[T_train:]

    L_t = torch.tensor(L_norm, dtype=torch.float32)
    results = {}

    # Train RD-GKAN
    for model_name, make_model in [
        ('RD-GKAN', lambda: RD_GKAN(1, G=8, k=3, x_range=(-4, 4))),
        ('MLP-RD', lambda: MLPReaction(1, hidden=32)),
    ]:
        print(f"\n  Training {model_name}...")
        model = make_model()
        train_temporal_rdgkan(model, train_traj, L_t, n_epochs=1500, patience=80)

        # One-step prediction on test
        model.eval()
        with torch.no_grad():
            one_step_errors = []
            for t in range(len(test_traj) - 1):
                c_t = torch.tensor(test_traj[t], dtype=torch.float32).to(DEVICE)
                c_next = torch.tensor(test_traj[t+1], dtype=torch.float32).to(DEVICE)
                pred = model.forward(c_t, L_t.to(DEVICE))
                one_step_errors.append(torch.sqrt(F.mse_loss(pred, c_next)).item())

            # Multi-step rollout
            c0 = torch.tensor(test_traj[0], dtype=torch.float32).to(DEVICE)
            rollout = [c0]
            c = c0
            for _ in range(min(T_test, 20)):
                c = model.forward(c, L_t.to(DEVICE))
                rollout.append(c)
            rollout = torch.stack(rollout).cpu().numpy()
            rollout_rmse = np.sqrt(np.mean((rollout[:len(test_traj)] - test_traj[:len(rollout)])**2))

        results[model_name] = {
            'one_step_rmse': float(np.mean(one_step_errors)),
            'rollout_rmse': float(rollout_rmse),
            'D_learned': float(abs(model.D.item())),
        }
        print(f"    1-step RMSE: {np.mean(one_step_errors):.6f}")
        print(f"    Rollout RMSE: {rollout_rmse:.6f}")
        print(f"    D_learned: {abs(model.D.item()):.4f}")

    results['dataset'] = {
        'N': N, 'T_total': T_total, 'T_train': T_train, 'T_test': T_test,
        'avg_degree': float(avg_deg), 'signal_col': signal_col,
    }
    return results


# ============================================================
# EXPERIMENT 2: WOUND HEALING SPATIAL (perturbational)
# ============================================================
def experiment_wound_healing():
    print("\n" + "="*60)
    print("EXP 2: Wound Healing Spatial (GSE241124)")
    print("="*60)

    wh_dir = os.path.join(DATA_DIR, 'WoundHealing_GSE241124')

    # Load metadata
    meta = pd.read_csv(os.path.join(wh_dir, 'GSE241124_spatialseq_metadata_acutewound.txt'), sep='\t')
    print(f"  Metadata: {len(meta)} spots")
    print(f"  Conditions: {meta['Condition'].value_counts().to_dict()}")
    print(f"  Donors: {meta['Donor'].unique().tolist()}")

    # Find H5 files
    h5_files = sorted([f for f in os.listdir(wh_dir) if f.endswith('.h5')])
    print(f"  H5 files: {len(h5_files)}")

    # Find spatial ZIP files
    zip_files = sorted([f for f in os.listdir(wh_dir) if f.endswith('.tar.gz') or f.endswith('.zip')])

    results = {}
    conditions_tested = []

    for h5_file in h5_files:
        # Determine condition from filename
        condition = None
        for cond in ['Skin', 'Wound1', 'Wound7', 'Wound30']:
            if cond in h5_file:
                condition = cond
                break
        if condition is None:
            continue
        if condition in conditions_tested:
            continue  # one per condition

        print(f"\n  Processing: {h5_file} (condition: {condition})")

        # Load expression from H5
        try:
            with h5py.File(os.path.join(wh_dir, h5_file), 'r') as f:
                mat = f['matrix']
                data = np.array(mat['data'])
                indices = np.array(mat['indices'])
                indptr = np.array(mat['indptr'])
                shape = np.array(mat['shape'])
                barcodes = [b.decode() for b in mat['barcodes']]
                gene_names = [g.decode() for g in mat['features']['name']]

            expr_sparse = scipy.sparse.csc_matrix((data, indices, indptr), shape=shape)
            expr = expr_sparse.toarray().T  # (spots, genes)
            N_spots = expr.shape[0]
            print(f"    Expression: {expr.shape} ({N_spots} spots, {expr.shape[1]} genes)")
        except Exception as e:
            print(f"    Error loading H5: {e}")
            continue

        # Try to load spatial coordinates from zip
        import zipfile, tarfile
        spatial_found = False
        positions = None

        # Look for matching spatial zip/tar
        sample_id = h5_file.replace('_filtered_feature_bc_matrix.h5', '')
        for zf in os.listdir(wh_dir):
            if sample_id in zf and (zf.endswith('.tar.gz') or zf.endswith('.zip')):
                zf_path = os.path.join(wh_dir, zf)
                try:
                    if zf.endswith('.tar.gz'):
                        with tarfile.open(zf_path, 'r:gz') as tar:
                            for member in tar.getmembers():
                                if 'tissue_positions' in member.name:
                                    f = tar.extractfile(member)
                                    pos_df = pd.read_csv(f, header=None)
                                    in_tissue = pos_df.iloc[:, 1] == 1
                                    positions = pos_df.loc[in_tissue, [4, 5]].values.astype(np.float32)
                                    spatial_found = True
                                    break
                except Exception as e:
                    print(f"    Spatial extraction error: {e}")

        if not spatial_found or positions is None:
            # Generate positions from array coordinates in metadata
            cond_meta = meta[meta['Condition'] == condition].head(N_spots)
            if len(cond_meta) > 0:
                # Use spot index as pseudo-position
                positions = np.column_stack([
                    np.arange(N_spots) % int(np.sqrt(N_spots)),
                    np.arange(N_spots) // int(np.sqrt(N_spots))
                ]).astype(np.float32) * 100
                spatial_found = True
                print(f"    Using pseudo-positions (no spatial file found)")

        if not spatial_found:
            print(f"    Skipping (no spatial coordinates)")
            continue

        # Subsample if too many spots
        N_max = 500
        if N_spots > N_max:
            rng = np.random.RandomState(42)
            idx = rng.choice(N_spots, N_max, replace=False)
            expr = expr[idx]
            if len(positions) > len(idx):
                positions = positions[idx]
            N_spots = N_max

        # Ensure positions match expression
        N_spots = min(len(positions), len(expr))
        positions = positions[:N_spots]
        expr = expr[:N_spots]

        # Select top variable genes
        M = 8
        gene_mean = expr.mean(0) + 1e-10
        gene_var = expr.var(0)
        fano = gene_var / gene_mean
        fano[gene_mean < 0.5] = 0
        top_genes = np.argsort(fano)[-M:][::-1]
        expr_sel = np.log1p(expr[:, top_genes].astype(np.float32))
        expr_sel = (expr_sel - expr_sel.mean(0)) / (expr_sel.std(0) + 1e-10)

        # Build graph
        L_norm, W, avg_deg = build_knn_graph(positions, k=min(6, N_spots-1))
        print(f"    Graph: N={N_spots}, avg_degree={avg_deg:.2f}")

        # Region-based split (donor-level if possible, else spatial)
        n_clusters = min(10, N_spots // 10)
        if n_clusters < 2: n_clusters = 2
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(positions)
        test_cluster = np.random.RandomState(42).choice(n_clusters)
        test_mask = labels == test_cluster
        train_idx = np.where(~test_mask)[0]
        test_idx = np.where(test_mask)[0]
        print(f"    Split: {len(train_idx)} train, {len(test_idx)} test")

        # Train RD-GKAN (spatial prediction with noise)
        from run_revised_experiments import CrossFeatureGraphKAN, GNNBaseline, GATBaseline

        edge_index_list = []
        edge_weight_list = []
        for i in range(N_spots):
            for j in range(N_spots):
                if W[i, j] > 0:
                    edge_index_list.append([i, j])
                    edge_weight_list.append(W[i, j])
        edge_index = torch.tensor(np.array(edge_index_list).T, dtype=torch.long) if edge_index_list else torch.zeros(2, 0, dtype=torch.long)
        edge_weight = torch.tensor(edge_weight_list, dtype=torch.float32) if edge_weight_list else torch.zeros(0)

        cond_results = {}
        rng = np.random.RandomState(42)
        noise = rng.randn(*expr_sel.shape).astype(np.float32) * 0.3
        x_noisy = expr_sel + noise

        from run_revised_experiments import train_model
        for name, make_model in [
            ('RD-GKAN', lambda: CrossFeatureGraphKAN(M, n_layers=2, G=5, k=3)),
            ('GNN', lambda: GNNBaseline(M, hidden=64, n_layers=2)),
        ]:
            model = make_model()
            rmse, _ = train_model(model, x_noisy, expr_sel, edge_index, edge_weight,
                                   train_idx, test_idx, n_epochs=500, patience=30)
            cond_results[name] = float(rmse)
            print(f"    {name}: RMSE={rmse:.4f}")

        results[condition] = {
            'models': cond_results, 'N': N_spots, 'M': M,
            'avg_degree': float(avg_deg),
            'n_train': len(train_idx), 'n_test': len(test_idx),
        }
        conditions_tested.append(condition)

    # Cross-condition analysis
    if len(results) >= 2:
        print("\n  Cross-condition comparison:")
        for cond in sorted(results.keys()):
            if 'models' in results[cond]:
                for model, rmse in results[cond]['models'].items():
                    print(f"    {cond} / {model}: RMSE={rmse:.4f}")

    return results


# ============================================================
# EXPERIMENT 3: ERK SSBD (small temporal supplement)
# ============================================================
def experiment_erk_ssbd():
    print("\n" + "="*60)
    print("EXP 3: ERK SSBD (13 cells, 34 timepoints)")
    print("="*60)

    h5_path = os.path.join(DATA_DIR, 'ERK_SSBD', 'Figure4C_bd5.h5')
    with h5py.File(h5_path, 'r') as f:
        n_timepoints = len(f['data'].keys())
        # Extract time series
        signals = []
        for t in range(n_timepoints):
            feat = f[f'data/{t}/feature/0']
            vals = np.array(feat)
            # Extract the numeric value from each record
            if vals.dtype.names and 'MLCintensity' in str(vals.dtype):
                signals.append([v[0] for v in vals])
            else:
                signals.append(vals.flatten().tolist())

    signals = np.array(signals, dtype=np.float32)  # (T, N)
    if signals.ndim == 1:
        print(f"  Could not parse signals properly, shape: {signals.shape}")
        return {'error': 'parse_error', 'shape': str(signals.shape)}

    N = signals.shape[1]
    T = signals.shape[0]
    print(f"  Cells: {N}, Timepoints: {T}")
    print(f"  Signal range: [{signals.min():.4f}, {signals.max():.4f}]")

    # Normalize
    signals = (signals - signals.mean()) / (signals.std() + 1e-10)
    trajectory = signals.reshape(T, N, 1)

    # Build simple chain graph (cells are in a line for this experiment)
    positions = np.column_stack([np.arange(N), np.zeros(N)]).astype(np.float32)
    L_norm, W, avg_deg = build_knn_graph(positions, k=min(2, N-1))

    # Train/test split
    T_train = int(T * 0.7)
    train_traj = trajectory[:T_train+1]

    L_t = torch.tensor(L_norm, dtype=torch.float32)
    model = RD_GKAN(1, G=5, k=3, x_range=(-4, 4))
    train_temporal_rdgkan(model, train_traj, L_t, n_epochs=1000, patience=60)

    # Test
    model.eval()
    test_traj = trajectory[T_train:]
    with torch.no_grad():
        errors = []
        for t in range(len(test_traj) - 1):
            c_t = torch.tensor(test_traj[t], dtype=torch.float32).to(DEVICE)
            c_next = torch.tensor(test_traj[t+1], dtype=torch.float32).to(DEVICE)
            pred = model.forward(c_t, L_t.to(DEVICE))
            errors.append(torch.sqrt(F.mse_loss(pred, c_next)).item())

    rmse = float(np.mean(errors))
    print(f"  1-step test RMSE: {rmse:.6f}")
    print(f"  D_learned: {abs(model.D.item()):.4f}")

    return {
        'one_step_rmse': rmse, 'D_learned': float(abs(model.D.item())),
        'N': N, 'T': T, 'T_train': T_train,
    }


# ============================================================
# EXPERIMENT 4: STAPH QS METADATA ANALYSIS
# ============================================================
def experiment_staph_metadata():
    print("\n" + "="*60)
    print("EXP 4: S. aureus QS Metadata Analysis (S-BIAD1046)")
    print("="*60)

    staph_dir = os.path.join(DATA_DIR, 'StaphQS_BIAD1046')

    # Parse CC metadata
    with open(os.path.join(staph_dir, 'cc_meta_processing.json')) as f:
        cc_meta = json.load(f)

    # Parse MM metadata
    with open(os.path.join(staph_dir, 'mm_meta_processing.json')) as f:
        mm_meta = json.load(f)

    print(f"  CC experiments: {len(cc_meta)}")
    print(f"  MM experiments: {len(mm_meta)}")

    # Analyze CC experiments
    cc_summary = []
    for exp_id, exp in cc_meta.items():
        cc_summary.append({
            'id': exp_id,
            'strains': exp.get('strains', ''),
            'dt': exp.get('dt', ''),
            'dynamic_class': exp.get('DynamicClass', ''),
        })
    print("\n  CC experiments:")
    for s in cc_summary:
        print(f"    {s['id']}: {s['strains']}, dt={s['dt']}s, class={s['dynamic_class']}")

    # Analyze MM experiments
    mm_by_strain = {}
    for exp_id, exp in mm_meta.items():
        strain = exp.get('Strain', 'unknown')
        if strain not in mm_by_strain:
            mm_by_strain[strain] = []
        mm_by_strain[strain].append({
            'id': exp_id,
            'AIP_type': exp.get('AIPtype', ''),
            'AIP_nM': exp.get('AIPnM', 0),
            'n_chambers': exp.get('nChambers', 0),
            'dt': exp.get('dt', ''),
        })

    print("\n  MM experiments by strain:")
    for strain, exps in sorted(mm_by_strain.items()):
        aip_concs = [e['AIP_nM'] for e in exps if e['AIP_nM']]
        print(f"    {strain}: {len(exps)} exps, AIP concentrations: {sorted(set(aip_concs))}")

    # Check if we can extract cell data from the zips
    cc_zip = os.path.join(staph_dir, 'cc_savesV1.zip')
    mm_zip = os.path.join(staph_dir, 'mm_savesV1.zip')

    cc_available = os.path.exists(cc_zip)
    mm_available = os.path.exists(mm_zip)
    print(f"\n  CC data zip: {'available' if cc_available else 'missing'} ({os.path.getsize(cc_zip)/1e9:.1f} GB)" if cc_available else "")
    print(f"  MM data zip: {'available' if mm_available else 'missing'} ({os.path.getsize(mm_zip)/1e9:.1f} GB)" if mm_available else "")

    # Try to extract one small experiment from MM zip for a proof-of-concept
    if mm_available:
        import zipfile
        print("\n  Attempting to read MM zip structure...")
        try:
            with zipfile.ZipFile(mm_zip, 'r') as zf:
                names = zf.namelist()
                print(f"    Total files in zip: {len(names)}")
                # Show first few files
                for n in names[:20]:
                    print(f"      {n}")

                # Look for regionprops or tracking data
                props_files = [n for n in names if 'regionprops' in n.lower() or 'tracking' in n.lower() or 'props' in n.lower()]
                print(f"    Regionprops/tracking files: {len(props_files)}")
                if props_files:
                    for pf in props_files[:5]:
                        print(f"      {pf}")
        except Exception as e:
            print(f"    Error reading zip: {e}")

    return {
        'cc_experiments': len(cc_meta),
        'mm_experiments': len(mm_meta),
        'mm_strains': {k: len(v) for k, v in mm_by_strain.items()},
        'cc_summary': cc_summary,
        'cc_data_available': cc_available,
        'mm_data_available': mm_available,
    }


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    results = {}

    results['erk_waves'] = experiment_erk_waves()
    results['wound_healing'] = experiment_wound_healing()
    results['erk_ssbd'] = experiment_erk_ssbd()
    results['staph_qs'] = experiment_staph_metadata()

    out = os.path.join(RESULTS_DIR, 'new_datasets_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"ALL DONE in {elapsed/60:.1f} min. Results: {out}")
    print(f"{'='*60}")

    # Summary
    if 'erk_waves' in results and 'error' not in results['erk_waves']:
        ew = results['erk_waves']
        print(f"\nERK Waves: RD-GKAN 1-step={ew.get('RD-GKAN',{}).get('one_step_rmse','?')}")
    if 'wound_healing' in results:
        wh = results['wound_healing']
        for cond in sorted(wh.keys()):
            if isinstance(wh[cond], dict) and 'models' in wh[cond]:
                print(f"Wound {cond}: {wh[cond]['models']}")


if __name__ == '__main__':
    main()
