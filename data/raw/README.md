# Raw Data Layout

Full datasets are external and intentionally excluded from Git.

This directory documents the expected local layout for raw inputs used by training and evaluation workflows. The repository stores metadata, manifests, split snapshots, and small examples only.

Use:

- `data/manifests/` for dataset-level lineage and provenance
- `artifacts/manifests/` for training/evidence linkage
- `configs/external_paths.example.yaml` to map local machine paths

Do not commit full datasets into this directory.
