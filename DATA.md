# Datasets

The raw data are **not** redistributed in this repository (they total ~46 GB and are owned by
their original providers). Download each from the source below and place it under
`data/<directory>/`. The committed `results/*.json` files were produced from these datasets.

| `data/` directory | Dataset / paper | Source & accession |
|---|---|---|
| `MacroscaleTestbed/` | Macroscale molecular-communication testbed (Hofmann et al., 2023) | IEEE DataPort, DOI `10.21227/ytkm-xp81` |
| `ProtonPumpingBacteria/` | *E. coli* proton-pump MC testbed (Grebenstein et al., *IEEE T-MBMC*, 2019) | Dataset accompanying the paper, DOI `10.1109/TMBMC.2019.2957783` |
| `AnalogNetworkCoding/` | Analog network coding MC relay testbed (Hofmann et al., *IEEE GLOBECOM*, 2023) | DOI `10.1109/GLOBECOM54140.2023.10437513` |
| `GeorgiaTechQS/` | *P. aeruginosa* quorum-sensing dose-response (Rattray et al., *mBio*, 2022) | DOI `10.1128/mbio.00745-22`; data in the authors' `hierarchy` repository |
| `10XBreastCancer/` | 10x Visium human breast cancer (Visium, 2020) | 10x Genomics public datasets (`10xgenomics.com/datasets`) |
| `10xVisiumCRC/` | 10x Visium HD human colon cancer (not used in the final paper) | 10x Genomics public datasets |
| `CCIBenchmark/` | Human intestine spatial transcriptomics (Fawkner-Corbett et al., *Cell*, 2021) | GEO `GSM4797918` (series `GSE158328`), DOI `10.1016/j.cell.2020.12.016` |
| `ERK_Collective/` | MDCK ERK/Akt collective signaling waves (Gagliardi et al., 2021) | MDCK-waves single-cell FRET dataset accompanying the paper |
| `WoundHealing_GSE241124/` | Human skin wound-healing Visium spatial transcriptomics (Liu et al.) | GEO `GSE241124` |
| `StaphQS_BIAD1046/` | *S. aureus* *agr* quorum sensing, single-cell microfluidics (Bär et al.) | BioStudies `S-BIAD1046`; published in *Nat. Commun.* (2026), DOI `10.1038/s41467-026-73552-9` |
| `ERK_SSBD/` | ERK wave imaging (HDF5), minor supplement | SSBD database |

## Notes

- Database identifiers (GEO `GSE…`, BioStudies `S-BIAD…`, IEEE DataPort DOIs, 10x Genomics
  dataset pages) are sufficient to locate each dataset; follow the provider's download and
  license terms.
- After downloading, the expected layout is `data/<directory>/` with the provider's original
  file names (the loader scripts in `experiments/` locate files by these names).
- Each dataset is governed by its provider's license; this repository's MIT license applies
  only to the code.
