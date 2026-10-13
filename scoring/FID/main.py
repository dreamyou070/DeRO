import os
os.environ["TORCH_FIDELITY_NUM_WORKERS"] = "0"

from pathlib import Path
from datetime import datetime

import pandas as pd
import torch_fidelity


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}


def has_images(folder: Path) -> bool:
    if not folder.is_dir():
        return False

    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            return True

    return False


def compute_fid(real_dir: str, fake_dir: str) -> float:
    metrics = torch_fidelity.calculate_metrics(
        input1=real_dir,
        input2=fake_dir,
        fid=True,
        kid=False,
        prc=False,
        cuda=True,
        verbose=False,
    )

    return float(metrics["frechet_inception_distance"])


# Real image folder
real_dir = "/lustre/dreamyou070/DeRO/src/english/DERO_Result/ablation_study_sdxl_base_uni_bi/bidirectional"

folders = {
    "sdxl_object_base": "/lustre/dreamyou070/DeRO/src/english/DERO_Result/ablation_study_sdxl_base_uni_bi/base",
    "sdxl_object_ours": "/lustre/dreamyou070/DeRO/src/english/DERO_Result/ablation_study_sdxl_base_uni_bi/bidirectional",
    "sdxl_concept_base": "/lustre/dreamyou070/DeRO/src/english/DERO_Result/concept_ablation_study_sdxl_base_uni_bi/base",
    "sdxl_concept_ours": "/lustre/dreamyou070/DeRO/src/english/DERO_Result/concept_ablation_study_sdxl_base_uni_bi/bidirectional",
}

real_dir = Path(real_dir)

if not real_dir.is_dir():
    raise FileNotFoundError(f"real_dir not found: {real_dir}")

if not has_images(real_dir):
    raise RuntimeError(f"real_dir must directly contain images: {real_dir}")

all_results = []

for model_name, fake_dir in folders.items():
    fake_dir = Path(fake_dir)

    row = {
        "model": model_name,
        "real_dir": str(real_dir),
        "fake_dir": str(fake_dir),
        "ts": datetime.now().isoformat(timespec="seconds"),
    }

    if not fake_dir.is_dir():
        row["FID"] = None
        row["error"] = f"fake_dir not found: {fake_dir}"
        all_results.append(row)
        print(f"[ERROR] {model_name}: fake_dir not found")
        continue

    if not has_images(fake_dir):
        row["FID"] = None
        row["error"] = f"fake_dir has no images: {fake_dir}"
        all_results.append(row)
        print(f"[ERROR] {model_name}: fake_dir has no images")
        continue

    try:
        fid = compute_fid(
            real_dir=str(real_dir),
            fake_dir=str(fake_dir),
        )

        row["FID"] = fid
        row["error"] = ""

        print("=" * 80)
        print(f"Model: {model_name}")
        print(f"Real dir: {real_dir}")
        print(f"Fake dir: {fake_dir}")
        print(f"FID: {fid:.4f}")

    except Exception as e:
        row["FID"] = None
        row["error"] = f"{type(e).__name__}: {e}"

        print("=" * 80)
        print(f"[ERROR] {model_name}")
        print(row["error"])

    all_results.append(row)


df = pd.DataFrame(all_results)
df = df.sort_values(by="FID", ascending=True, na_position="last")
df.to_csv("fid_results.csv", index=False)

print("=" * 80)
print("Saved detailed results to fid_results.csv")