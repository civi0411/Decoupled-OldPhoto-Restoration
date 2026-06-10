from __future__ import annotations

import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


ASSET_ROOT = PROJECT_ROOT / "docs" / "assets" / "demo3"
INPUT_PATH = PROJECT_ROOT / "examples" / "inputs" / "demo3.png"
GOLDEN_ROOT = PROJECT_ROOT / "examples" / "golden" / "demo3_r013_repair_wide"
MASK_PATH = GOLDEN_ROOT / "final_mask.png"
RESTORED_PATH = GOLDEN_ROOT / "restored_before_face.png"


def load_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", 24)
    except Exception:
        return ImageFont.load_default()


def copy_asset(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def fit_panel(image: Image.Image, panel_width: int = 512, panel_height: int = 512) -> Image.Image:
    canvas = Image.new("RGB", (panel_width, panel_height), (245, 245, 245))
    image = image.convert("RGB")
    image.thumbnail((panel_width, panel_height), Image.Resampling.LANCZOS)
    offset_x = (panel_width - image.width) // 2
    offset_y = (panel_height - image.height) // 2
    canvas.paste(image, (offset_x, offset_y))
    return canvas


def add_label(image: Image.Image, label: str, font: ImageFont.ImageFont) -> Image.Image:
    label_height = 44
    canvas = Image.new("RGB", (image.width, image.height + label_height), (255, 255, 255))
    canvas.paste(image, (0, label_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), label, fill=(20, 20, 20), font=font)
    return canvas


def build_contact_sheet(input_path: Path, mask_path: Path, restored_path: Path, output_path: Path) -> None:
    font = load_font()
    panels = [
        ("Input", fit_panel(Image.open(input_path))),
        ("Final Mask", fit_panel(Image.open(mask_path).convert("L").convert("RGB"))),
        ("Restored", fit_panel(Image.open(restored_path))),
    ]
    labeled = [add_label(image, label, font) for label, image in panels]
    sheet = Image.new("RGB", (sum(image.width for image in labeled), labeled[0].height), (255, 255, 255))
    offset_x = 0
    for image in labeled:
        sheet.paste(image, (offset_x, 0))
        offset_x += image.width
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def build_overlay_mask(input_path: Path, mask_path: Path, output_path: Path) -> None:
    image = Image.open(input_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    if image.size != mask.size:
        mask = mask.resize(image.size, Image.Resampling.NEAREST)

    base = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (255, 0, 0, 0))
    alpha = mask.point(lambda value: 120 if value > 0 else 0)
    overlay.putalpha(alpha)
    composed = Image.alpha_composite(base, overlay).convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    composed.save(output_path)


def main() -> int:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy input demo3: {INPUT_PATH}")
    if not MASK_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy final_mask golden: {MASK_PATH}")
    if not RESTORED_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy restored golden: {RESTORED_PATH}")

    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    copy_asset(INPUT_PATH, ASSET_ROOT / "input.png")
    copy_asset(MASK_PATH, ASSET_ROOT / "final_mask.png")
    copy_asset(RESTORED_PATH, ASSET_ROOT / "restored_before_face.png")
    build_contact_sheet(INPUT_PATH, MASK_PATH, RESTORED_PATH, ASSET_ROOT / "contact_sheet.png")
    build_overlay_mask(INPUT_PATH, MASK_PATH, ASSET_ROOT / "overlay_mask.png")

    print(f"asset_dir: {ASSET_ROOT}")
    print(f"input: {ASSET_ROOT / 'input.png'}")
    print(f"final_mask: {ASSET_ROOT / 'final_mask.png'}")
    print(f"restored_before_face: {ASSET_ROOT / 'restored_before_face.png'}")
    print(f"contact_sheet: {ASSET_ROOT / 'contact_sheet.png'}")
    print(f"overlay_mask: {ASSET_ROOT / 'overlay_mask.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
