# RD-GKAN: Interpretable Reaction–Diffusion Learning for Molecular Communication

Code and derived results for the paper

> **Interpretable Reaction–Diffusion Learning for Molecular Communication via Graph Kolmogorov–Arnold Networks**
> Liang Dong, Baylor University / UT Southwestern.
> IEEE Transactions on Molecular, Biological, and Multi-Scale Communications, 2026.
> DOI: [10.1109/TMBMC.2026.3731398](https://doi.org/10.1109/TMBMC.2026.3731398)

RD-GKAN is a graph learning architecture whose forward pass is constrained to implement
reaction–diffusion dynamics: a feature-wise B-spline (Kolmogorov–Arnold) function captures
the per-species reaction kinetics, while diffusion is carried by an explicit, physics-derived
graph Laplacian. Each learned reaction spline is projected onto a symbolic library to identify
the kinetics class, with a reconstruction-error criterion that rejects out-of-library kinetics.

## Repository contents

| Path | Description |
|------|-------------|
| `experiments/` | All experiment code (PyTorch). |
| `results/` | Derived results as JSON (the numbers behind the paper's tables/figures). |
| `figures/` | Standalone TikZ/pgfplots sources for Figures 1-15. |
| `DATA.md` | Public dataset accessions and download instructions. |
| `requirements.txt` | Python dependencies. |

The raw datasets are **not** redistributed here (they are large public datasets owned by their
original providers); see [`DATA.md`](DATA.md) to obtain them and place them under `data/`.

## Key scripts (`experiments/`)

- `run_synthetic_rd.py` — constrained RD-GKAN, synthetic reaction–diffusion experiments,
  time-step (Euler) scaling, stability diagnostic, symbolic-recovery scaling, ablations.
- `run_revision_experiments.py` — out-of-library kinetics rejection; time-step transferability.
- `run_e1_curves.py` — learned-vs-true-vs-library-fit curves for the rejection mechanism figure.
- `run_new_datasets.py` — ERK signaling waves, wound-healing and quorum-sensing temporal data.
- `run_wound_splines.py` — real learned reaction splines (S100A8) across wound-healing conditions.
- `run_wound_rmse.py` — wound-healing reconstruction RMSE (constrained RD-GKAN vs. controls).
- `run_option_b.py`, `run_revised_experiments.py` — spatial transcriptomics, masking, baselines
  (GNN, GAT, GRAND, GREAD).
- `run_spatial_rigorous.py`, `run_wound_ablation_rigorous.py`, `run_erk_rigorous.py`,
  `run_staph_rigorous.py` — leakage-controlled re-runs (three-way region splits, inductive
  message passing, validation-based selection, test read once).
- `run_t_scaling_verify.py`, `run_t_scaling_hill.py`, `run_t_scaling_rest.py`,
  `merge_t_scaling.py` — symbolic support-recovery rates versus the number of snapshots `T`,
  merged into `results/t_scaling_verify.json`.
- `run_staph_masked_rescore.py` — *S. aureus* controls rescored on transitions whose endpoints
  are both directly observed.
- `run_h_scaling_fp64.py` — forward-Euler vs. RK4 discretization scaling in double precision.

## Setup

```bash
conda create -n rdgkan python=3.10
conda activate rdgkan
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended but not required.

## Reproducing results

1. Obtain the datasets as described in [`DATA.md`](DATA.md) and place each under `data/<name>/`.
2. Run the relevant script, e.g.:
   ```bash
   python experiments/run_synthetic_rd.py
   python experiments/run_revision_experiments.py
   ```
   Outputs are written to `results/*.json`. The JSON files committed here are the exact
   derived values used in the paper.

## Funding

Supported in part by the National Cancer Institute (NCI) of the National Institutes of
Health (NIH) under Grant R01CA309499.

## Citation

The paper is published by IEEE TMBMC and citable by its DOI. The volume, issue, and page range
will be added here once the article is assigned to an issue.

```bibtex
@article{Dong2026RDGKAN,
  author  = {Liang Dong},
  title   = {Interpretable Reaction--Diffusion Learning for Molecular Communication
             via Graph {K}olmogorov--{A}rnold Networks},
  journal = {IEEE Trans. Mol. Biol. Multi-Scale Commun.},
  year    = {2026},
  doi     = {10.1109/TMBMC.2026.3731398},
  note    = {Early access}
}
```

## License

Code in this repository is released under the MIT License (see `LICENSE`). The datasets remain
the property of their original providers under their respective licenses (see `DATA.md`).
