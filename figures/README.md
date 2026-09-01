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
