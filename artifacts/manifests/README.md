# Artifact Manifests

Thư mục này mô tả artifact local và artifact ngoài repo bằng manifest, thay vì commit binary lớn hoặc full datasets vào Git.

## Nội dung chính

- `checkpoints_manifest.csv`: inventory checkpoint và checksum mong đợi
- `datasets_manifest.csv`: lineage dữ liệu gồm cả synthetic lineage và real-domain fine-tuning datasets
- `reproduction_runs_manifest.csv`: liên kết giữa các run, dataset lineage, checkpoint lineage và trạng thái bằng chứng

## Nguyên tắc

- Checkpoint binary không commit vào Git theo mặc định.
- Full datasets không commit vào Git theo mặc định.
- Manifest có thể trỏ tới external local paths để mô tả provenance, nhưng không biến các binary đó thành tracked files.
- Skeleton folders dưới `data/raw/` và `data/processed/` chỉ mô tả local layout mong đợi.
- Skeleton folders không phải bằng chứng rằng full dataset payload đã được commit vào repository.
- Full datasets được resolve qua manifests và external paths ngoài Git.

## Cách hiểu trạng thái

- `available`: artifact nhỏ hoặc docs asset đang có trong repo/workspace theo policy
- `available_external_only`: artifact có external source rõ ràng nhưng không nằm trong Git
- `available_local_only`: artifact local có thể tồn tại cho runtime/demo nhưng không phải source-of-truth training data
- `historical_evidence_only`: chỉ giữ vai trò provenance hoặc training lineage; local full artifact không được đóng gói trong repo này
- `rejected_or_historical`: artifact lịch sử hoặc dataset bị loại, không nên trình bày như final dataset
- `not_found_local_artifact`: không tìm thấy artifact local đầy đủ, nhưng row vẫn hợp lệ nếu chỉ có historical evidence

## Kiểm tra artifact local

```bash
python -B scripts/verify_artifacts.py check-checkpoints --repo-root .
python -B scripts/verify_artifacts.py check-datasets --repo-root .
python -B scripts/verify_artifacts.py check-all --repo-root . --strict
```

Script kiểm tra manifest, path, Git tracking và checksum. Script không tự tải, không tự copy và không sửa repo.
