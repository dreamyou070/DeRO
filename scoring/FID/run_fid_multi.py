# run_fid_multi.py
# ------------------------------------------------------------
# FIX:
# torch-fidelity 내부 DataLoader multiprocessing 버그 방지
# (RuntimeError: Trying to resize storage that is not resizable)
# ------------------------------------------------------------
import os
os.environ["TORCH_FIDELITY_NUM_WORKERS"] = "0"  # 🔥 핵심 수정

import argparse
import pandas as pd
from pathlib import Path
import torch_fidelity


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}


def has_images(d: Path) -> bool:
    if not d.is_dir():
        return False
    for p in d.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            return True
    return False


def find_image_dir(model_dir: Path) -> Path | None:
    # 1) images directly under model_dir
    if has_images(model_dir):
        return model_dir

    # 2) common nested folder names
    for sub in ["images", "imgs", "samples", "results", "output", "outputs"]:
        cand = model_dir / sub
        if has_images(cand):
            return cand

    # 3) one-level deep search
    for child in model_dir.iterdir():
        if child.is_dir() and has_images(child):
            return child

    return None


def list_models(root_dir: Path, include: list[str] | None, exclude: list[str] | None) -> list[Path]:
    models = []
    for p in sorted(root_dir.iterdir()):
        if not p.is_dir():
            continue
        name = p.name
        if include and name not in include:
            continue
        if exclude and name in exclude:
            continue
        img_dir = find_image_dir(p)
        if img_dir is None:
            continue
        models.append(p)
    return models


def calc_metrics(real_dir: str, fake_dir: str, kid: bool, prc: bool):
    """
    torch-fidelity expects directories that directly contain image files
    """
    metrics = torch_fidelity.calculate_metrics(
        input1=real_dir,
        input2=fake_dir,
        fid=True,
        kid=kid,
        prc=prc,
        cuda=True,
        verbose=False,
    )

    out = {
        "FID": float(metrics.get("frechet_inception_distance", float("nan"))),
    }

    if kid:
        out["KID_mean"] = float(metrics.get("kernel_inception_distance_mean", float("nan")))
        out["KID_std"] = float(metrics.get("kernel_inception_distance_std", float("nan")))

    if prc:
        out["precision"] = float(metrics.get("precision", float("nan")))
        out["recall"] = float(metrics.get("recall", float("nan")))
        out["density"] = float(metrics.get("density", float("nan")))
        out["coverage"] = float(metrics.get("coverage", float("nan")))

    return out


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", type=str, required=True,
                    help="root directory that contains per-model folders")
    ap.add_argument("--real_dir", type=str, required=True,
                    help="directory with real images (e.g., COCO val2017)")
    ap.add_argument("--out_csv", type=str, required=True,
                    help="output csv path")
    ap.add_argument("--kid", action="store_true", default=False)
    ap.add_argument("--prc", action="store_true", default=False)
    ap.add_argument("--include", type=str, default="",
                    help="comma-separated model folder names to include")
    ap.add_argument("--exclude", type=str, default="",
                    help="comma-separated model folder names to exclude")
    return ap.parse_args()


def main():
    args = parse_args()

    root_dir = Path(args.root_dir)
    real_dir = Path(args.real_dir)

    if not root_dir.is_dir():
        raise FileNotFoundError(f"root_dir not found: {root_dir}")
    if not real_dir.is_dir():
        raise FileNotFoundError(f"real_dir not found: {real_dir}")

    include = [x.strip() for x in args.include.split(",") if x.strip()] if args.include else None
    exclude = [x.strip() for x in args.exclude.split(",") if x.strip()] if args.exclude else None

    model_dirs = list_models(root_dir, include, exclude)
    if len(model_dirs) == 0:
        raise RuntimeError(
            f"No valid model directories with images found under: {root_dir}\n"
            f"(Expected each model folder to contain images directly, or in a common subfolder like images/)"
        )

    rows = []

    for mdir in model_dirs:
        img_dir = find_image_dir(mdir)
        assert img_dir is not None

        print(f"[RUN] model={mdir.name} | fake_dir={img_dir}")

        try:
            met = calc_metrics(
                real_dir=str(real_dir),
                fake_dir=str(img_dir),
                kid=args.kid,
                prc=args.prc,
            )
            row = {
                "model": mdir.name,
                "fake_dir": str(img_dir),
                "real_dir": str(real_dir),
            }
            row.update(met)

        except Exception as e:
            row = {
                "model": mdir.name,
                "fake_dir": str(img_dir),
                "real_dir": str(real_dir),
                "error": f"{type(e).__name__}: {e}",
            }

        rows.append(row)

    df = pd.DataFrame(rows)

    if "FID" in df.columns:
        df = df.sort_values(by="FID", ascending=True, na_position="last")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"[DONE] saved: {out_path}")


if __name__ == "__main__":
    main()