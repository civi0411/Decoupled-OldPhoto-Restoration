# Checkpoint Layout

Checkpoint binaries are local artifacts and intentionally excluded from Git.

This directory documents the expected public-facing local placement for segmentation checkpoints. Use `artifacts/manifests/checkpoints_manifest.csv` together with `scripts/verify_artifacts.py` to verify the expected path, role, and SHA256 of each checkpoint artifact.

The repository stores metadata and layout guidance only. Full checkpoint binaries are not committed.

### Segmenter Checkpoints (R006-R013)
The segmenter checkpoints follow canonical experiment-run names (e.g., seg-unet-attn-r013-gen120-fixed118-local) to precisely map to historical logs and metrics.
- **R006-R008**: These are skeleton-only folders because the binary weights are missing from the audit, preserving lineage evidence.
- **R009-R013**: These are available locally if copied from <LOCAL_ARTIFACT_ROOT>.
- **Current Alias**: checkpoints/segmenter/current/ is an alias conceptually pointing to the selected R013 canonical folder without duplicating binaries.
- **Binary Policy**: All checkpoint binaries (.ckpt, .pth) are strictly **local ignored** and **not committed** to the repository to maintain claim-safety and small repository size.
