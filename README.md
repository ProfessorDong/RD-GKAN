# RD-GKAN: Interpretable Reaction–Diffusion Learning for Molecular Communication

Code and derived results for the paper

> **Interpretable Reaction–Diffusion Learning for Molecular Communication via Graph Kolmogorov–Arnold Networks**
> Liang Dong, Baylor University / UT Southwestern.
> IEEE Transactions on Molecular, Biological, and Multi-Scale Communications (under review).

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

Please cite the paper once published. A BibTeX entry will be added here upon acceptance.

## License

Code in this repository is released under the MIT License (see `LICENSE`). The datasets remain
the property of their original providers under their respective licenses (see `DATA.md`).
