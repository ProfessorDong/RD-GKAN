# Figure sources

Standalone TikZ/pgfplots sources for Figures 1-15 of the paper. Each file compiles on its own
to a tightly cropped figure:

```bash
pdflatex fig7.tex          # -> fig7.pdf
pdftops -eps fig7.pdf fig7.eps   # optional, for a latex+dvips build
```

Every plotted coordinate is written literally in the source, so the figures are reproducible
without rerunning the experiments. The values correspond to the committed `results/*.json`
files; the scripts in `experiments/` regenerate those JSONs from the raw datasets listed in
`DATA.md`.

Figures 1 and 2 are schematics (system model; architecture and symbolic-recovery pipeline).
Figures 3-15 are data plots.

## Copyright

The figures these files produce appear in the published article

> L. Dong, "Interpretable Reaction-Diffusion Learning for Molecular Communication via Graph
> Kolmogorov-Arnold Networks," IEEE Transactions on Molecular, Biological, and Multi-Scale
> Communications, 2026, doi: 10.1109/TMBMC.2026.3731398.

Copyright in the published article, including its figures, is held by the IEEE. These source
files are provided for reproducibility and are covered by the repository's MIT license as
source code. Reuse of the published figures themselves is governed by IEEE policy; see
https://www.ieee.org/publications/rights/index.html.
