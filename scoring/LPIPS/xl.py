import os
import pandas as pd
import torch
import lpips
from PIL import Image
from torchvision import transforms


device = "cuda" if torch.cuda.is_available() else "cpu"

folders = {
    "sdxl_concept_base": "/lustre/dreamyou070/DeRO/src/english/DERO_Result/concept_ablation_study_sdxl_base_uni_bi/base",
    "sdxl_concept_ours": "/lustre/dreamyou070/DeRO/src/english/DERO_Result/concept_ablation_study_sdxl_base_uni_bi/bidirectional",
}

reference_folder = folders["sdxl_concept_base"]
target_folder = folders["sdxl_concept_ours"]

loss_fn = lpips.LPIPS(net="alex").to(device).eval()

tfm = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


@torch.no_grad()
def compute_lpips(img_a_path, img_b_path):
    img_a = tfm(Image.open(img_a_path).convert("RGB")).unsqueeze(0).to(device)
    img_b = tfm(Image.open(img_b_path).convert("RGB")).unsqueeze(0).to(device)

    score = loss_fn(img_a, img_b)
    return float(score.item())


all_results = []
scores = []

base_files = sorted(os.listdir(reference_folder))

for file in base_files:
    if not file.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        continue

    base_path = os.path.join(reference_folder, file)
    ours_path = os.path.join(target_folder, file)

    if not os.path.exists(ours_path):
        print(f"[MISSING] {file}")
        continue

    try:
        name, ext = os.path.splitext(file)
        prompt = name.replace("_", " ")

        lpips_score = compute_lpips(base_path, ours_path)
        scores.append(lpips_score)

        all_results.append({
            "file": file,
            "prompt": prompt,
            "base_path": base_path,
            "ours_path": ours_path,
            "lpips": lpips_score,
        })

    except Exception as e:
        print(f"[ERROR] {file}: {e}")

avg_lpips = sum(scores) / len(scores) if len(scores) > 0 else 0.0

print("=" * 80)
print("LPIPS: SDXL concept base vs ours")
print(f"Num pairs: {len(scores)}")
print(f"Average LPIPS: {avg_lpips:.4f}")

df = pd.DataFrame(all_results)
df.to_csv("lpips_sdxl_concept_base_vs_ours.csv", index=False)

print("=" * 80)
print("Saved detailed results to lpips_sdxl_concept_base_vs_ours.csv")