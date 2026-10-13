import os
from pathlib import Path
from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

ROOT = Path("/MIR/nas1/dreamyou070/NegatingDiffusion/2_concept_negate")
OUT_ROOT = ROOT / "_concat_1x7"
SUBSETS = ["picasso", "vangogh"]

# 1x7 순서 고정
#ORDER = ["1_base", "1_SPEED", "2_base", "2_erasing", "2_RECE", "3_base", "3_ours"]
ORDER = ["1_base", "1_MACE","1_SPEED", "2_base","2_RECE", "3_base",  "3_UCE", "3_ours"]


def list_images(folder: Path):
    files = {}
    if not folder.exists():
        return files
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files[p.name] = p
    return files


def load_rgb(p: Path) -> Image.Image:
    return Image.open(p).convert("RGB")


def resize_to_same_height(imgs, target_h=None):
    if target_h is None:
        target_h = min(im.height for im in imgs)
    outs = []
    for im in imgs:
        if im.height == target_h:
            outs.append(im)
        else:
            w = int(round(im.width * (target_h / im.height)))
            outs.append(im.resize((w, target_h), Image.BICUBIC))
    return outs


def concat_h(imgs):
    total_w = sum(im.width for im in imgs)
    max_h = max(im.height for im in imgs)
    canvas = Image.new("RGB", (total_w, max_h))
    x = 0
    for im in imgs:
        canvas.paste(im, (x, 0))
        x += im.width
    return canvas


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # subset별로 7개 폴더가 모두 존재해야 함
    subset_dirs = {subset: [ROOT / name / subset for name in ORDER] for subset in SUBSETS}

    for subset in SUBSETS:
        dirs = subset_dirs[subset]
        missing_dirs = [str(d) for d in dirs if not d.exists()]
        if missing_dirs:
            print(f"[{subset}] missing subset dirs -> skipped")
            for d in missing_dirs:
                print(f"  - {d}")
            continue

        # 7개 폴더 각각에서 파일 맵 로드
        maps = [list_images(d) for d in dirs]

        # 파일명 교집합
        common = set(maps[0].keys())
        for m in maps[1:]:
            common &= set(m.keys())

        common = sorted(common)
        if not common:
            print(f"[{subset}] no common filenames across all 7 folders.")
            continue

        save_dir = OUT_ROOT / subset
        save_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        for fname in common:
            try:
                imgs = [load_rgb(m[fname]) for m in maps]  # ORDER 순서대로 로드
                imgs = resize_to_same_height(imgs)  # 높이 통일
                out = concat_h(imgs)  # 1x7 concat
                out_path = save_dir / fname
                out.save(out_path)
                saved += 1
            except Exception as e:
                print(f"  - failed {subset}/{fname}: {e}")

        print(f"[{subset}] saved {saved} (1x7) concats -> {save_dir}")

    print(f"\nDone. All outputs saved under: {OUT_ROOT}")


if __name__ == "__main__":
    main()
