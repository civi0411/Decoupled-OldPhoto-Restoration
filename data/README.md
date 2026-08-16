# Data Policy

Full datasets are external and intentionally excluded from Git.

The repository keeps manifests, split metadata, documentation assets, and small examples only.

## Data lineage

1. Clean image source: DIV2K clean images
2. Crack source: CrackForest crack annotations and source imagery
3. Processed crack bank: RGBA crack assets used by the synthetic pipeline
4. Synthetic pairs: `ds-crack3d-512-n1000-v001` for the historical synthetic pretraining lineage
5. Real fine-tune datasets: `old_photo_pairs_10_hq` and `r013_finetune_set`

## Expected layout

- `data/raw/` documents the local layout for external raw datasets
- `data/processed/` documents the local layout for processed assets and derived datasets
- `data/splits/` stores lightweight split metadata when needed
- `data/manifests/` stores provenance and lineage manifests

Skeleton folders under `data/raw/` and `data/processed/` document the expected local layout only. They do not mean that full dataset payloads are committed to the repository.

## Path mapping

Use `configs/external_paths.example.yaml` together with:

- `data/manifests/`
- `artifacts/manifests/`

to map machine-specific local paths.

## Git policy

- Do not commit full datasets.
- Do not commit large processed image collections.
- Keep only manifests, split metadata, and small examples needed for documentation or lightweight verification.
