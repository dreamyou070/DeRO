import os
from pathlib import Path
from PIL import Image
import numpy as np

base_folder = Path("/MIR/nas1/dreamyou070/NegatingDiffusion/2_concept_negate")

# -----------------------------
# Utils
# -----------------------------
def load_rgb(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0

def mse(a: np.ndarray, b: np.ndarray) -> float:
    # if size mismatch, resize b to a (or skip). here: resize b to a.
    if a.shape != b.shape:
        b_img = Image.fromarray((b * 255.0).astype(np.uint8))
        b_img = b_img.resize((a.shape[1], a.shape[0]), Image.BILINEAR)
        b = np.asarray(b_img, dtype=np.float32) / 255.0
    return float(np.mean((a - b) ** 2))

def psnr_from_mse(m: float, eps: float = 1e-12) -> float:
    # max_val = 1.0
    return float(10.0 * np.log10(1.0 / max(m, eps)))

def get_category_prefix(folder_name: str) -> str:
    # "1_base" -> "1", "12_modelX" -> "12"
    return folder_name.split("_", 1)[0]

def is_base_folder(folder_name: str) -> bool:
    # "1_base" or "1_Base" etc.
    parts = folder_name.split("_", 1)
    return len(parts) == 2 and parts[1].lower() == "base"

# -----------------------------
# Build category -> folders map
# -----------------------------
cats = {}  # cat_id -> {"base": Path, "models": [Path,...]}
for p in base_folder.iterdir():
    if not p.is_dir():
        continue
    cat = get_category_prefix(p.name)
    cats.setdefault(cat, {"base": None, "models": []})
    if is_base_folder(p.name):
        cats[cat]["base"] = p
    else:
        cats[cat]["models"].append(p)

# -----------------------------
# Compare each model with its category base
# Only compare within picasso/vangogh subfolders
# and only image files with the same filename
# -----------------------------
subsets = ["picasso", "vangogh"]
img_ext = {".png", ".jpg", ".jpeg", ".webp"}

overall_stats = []  # list of mse values across all categories/models/subsets

for cat, info in sorted(cats.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0]):
    base_dir = info["base"]
    model_dirs = info["models"]

    if base_dir is None or not base_dir.exists():
        print(f"[Category {cat}] base folder missing -> skipped")
        continue
    if not model_dirs:
        print(f"[Category {cat}] no model folders -> skipped")
        continue

    print(f"\n========== Category {cat} ==========")
    print(f"Base: {base_dir.name}")

    for model_dir in sorted(model_dirs, key=lambda p: p.name.lower()):
        model_mses = []
        subset_detail = {}

        for subset in subsets:
            b_sub = base_dir / subset
            m_sub = model_dir / subset
            if not b_sub.exists() or not m_sub.exists():
                continue

            b_files = {f.name: f for f in b_sub.iterdir() if f.is_file() and f.suffix.lower() in img_ext}
            m_files = {f.name: f for f in m_sub.iterdir() if f.is_file() and f.suffix.lower() in img_ext}
            common = sorted(set(b_files.keys()) & set(m_files.keys()))

            if not common:
                continue

            mses_here = []
            for name in common:
                a = load_rgb(b_files[name])
                b = load_rgb(m_files[name])
                mses_here.append(mse(a, b))

            subset_detail[subset] = {
                "count": len(mses_here),
                "mse": float(np.mean(mses_here)),
                "psnr": psnr_from_mse(float(np.mean(mses_here))),
            }
            model_mses.extend(mses_here)

        if not model_mses:
            print(f"- {model_dir.name}: no comparable images (missing subsets or no filename matches)")
            continue

        model_mse_mean = float(np.mean(model_mses))
        model_psnr_mean = psnr_from_mse(model_mse_mean)
        overall_stats.extend(model_mses)

        # Clean print
        print(f"- {model_dir.name}: MSE={model_mse_mean:.6f}, PSNR={model_psnr_mean:.2f} dB, N={len(model_mses)}")
        for subset in subsets:
            if subset in subset_detail:
                d = subset_detail[subset]
                print(f"    {subset}: MSE={d['mse']:.6f}, PSNR={d['psnr']:.2f} dB, N={d['count']}")

# -----------------------------
# Grand average across everything
# -----------------------------
if overall_stats:
    grand_mse = float(np.mean(overall_stats))
    grand_psnr = psnr_from_mse(grand_mse)
    print("\n========== Overall ==========")
    print(f"Grand MSE={grand_mse:.6f}, Grand PSNR={grand_psnr:.2f} dB, Total N={len(overall_stats)}")
else:
    print("\nNo comparable images found anywhere.")
