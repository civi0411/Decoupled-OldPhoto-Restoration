# External Dependencies

## LaMa

Repo submission không vendor official LaMa vào GitHub. Pipeline gọi LaMa qua đường dẫn external lấy từ `configs/external_paths.yaml`.

- `repo_root`: thư mục source LaMa
- `checkpoint`: Big-LaMa `best.ckpt`
- `conda_env_preferred`: environment GPU ưu tiên
- `conda_env_fallback`: environment CPU fallback

`configs/external_paths.example.yaml` chỉ là template. `configs/external_paths.yaml` là file local theo máy và đã bị ignore trong Git.

## CodeFormer

CodeFormer là dependency optional. Repo submission không commit source hoặc weights của CodeFormer.

- `repo_root`: thư mục source CodeFormer
- `checkpoint`: weight CodeFormer
- `conda_env`: environment riêng nếu cần

Trong trạng thái pipeline hiện tại, face restoration chưa phải core requirement của readiness check.

## sitecustomize.py

Repo hiện chưa dùng editable install. [sitecustomize.py](/f:/deeplearning/old_photo_restoration_blueprint21_submission/sitecustomize.py:1) giúp Python resolve `src/` khi chạy script trực tiếp từ repo root.

Đây là giải pháp tạm gọn cho submission. Cách chuẩn hơn về sau là dùng `pip install -e .`.

## Vì Sao Không Commit Dependencies Này

- Checkpoint và weights lớn, không phù hợp để đưa vào Git submission.
- Tách dependency runtime khỏi repo giúp repo nhẹ và rõ ràng hơn cho người chấm.
- Các path tuyệt đối chỉ nên nằm trong config local, không hard-code trong source.
