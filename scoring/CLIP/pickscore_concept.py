import os
from PIL import Image
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel


# ==========================================
# Setting
# ==========================================
base_folder = "/lustre/dreamyou070/DeRO/src/english/DERO_Result/FLUX_bidirectional_extraction_method_example_chocolate_cup"
prompt = "a cookie without chocolate chip and a glass with nothing on it"

save_per_sample = "pickscore_per_sample.csv"
save_summary = "pickscore_summary_by_model.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"

processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

processor = AutoProcessor.from_pretrained(processor_name_or_path)
model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp")


# ==========================================
# Function
# ==========================================
@torch.no_grad()
def compute_pickscore(image_path, prompt):
    image = Image.open(image_path).convert("RGB")

    image_inputs = processor(
        images=image,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)

    text_inputs = processor(
        text=prompt,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)

    image_embs = model.get_image_features(**image_inputs)
    image_embs = image_embs / image_embs.norm(dim=-1, keepdim=True)

    text_embs = model.get_text_features(**text_inputs)
    text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)

    pick_score = model.logit_scale.exp() * (text_embs @ image_embs.T)[0, 0]
    return pick_score.item()


def collect_images_recursive(root_dir):
    image_paths = []

    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.lower().endswith(IMG_EXTS):
                image_paths.append(os.path.join(dirpath, file))

    return sorted(image_paths)


# ==========================================
# Run
# ==========================================
image_paths = collect_images_recursive(base_folder)

print("=" * 80)
print(f"Base folder : {base_folder}")
print(f"Found images: {len(image_paths)}")
print(f"Prompt      : {prompt}")
print("=" * 80)

results = []

for img_path in tqdm(image_paths, desc="Computing PickScore"):
    file = os.path.basename(img_path)

    # base_folder 바로 아래 folder를 model/group 이름으로 사용
    rel_dir = os.path.relpath(os.path.dirname(img_path), base_folder)
    model_name = rel_dir.split(os.sep)[0] if rel_dir != "." else "root"

    try:
        pick_score = compute_pickscore(img_path, prompt)

        results.append({
            "model": model_name,
            "subfolder": rel_dir,
            "file": file,
            "prompt": prompt,
            "pick_score": pick_score,
            "image_path": img_path,
        })

    except Exception as e:
        print(f"[ERROR] {img_path}: {e}")


# ==========================================
# Save per-sample result
# ==========================================
df = pd.DataFrame(results)

if len(df) == 0:
    raise RuntimeError("No valid PickScore results were computed.")

df = df.sort_values(["model", "pick_score"], ascending=[True, False])
df.to_csv(save_per_sample, index=False)


# ==========================================
# Model-wise comparison summary
# ==========================================
summary = (
    df.groupby("model")
    .agg(
        num_images=("pick_score", "count"),
        avg_pick_score=("pick_score", "mean"),
        std_pick_score=("pick_score", "std"),
        min_pick_score=("pick_score", "min"),
        max_pick_score=("pick_score", "max"),
    )
    .reset_index()
    .sort_values("avg_pick_score", ascending=False)
)

summary.to_csv(save_summary, index=False)


# ==========================================
# Print
# ==========================================
print("=" * 80)
print("Model-wise PickScore comparison")
print(summary.to_string(index=False))

print("=" * 80)
print("Saved:")
print(f"- {save_per_sample}")
print(f"- {save_summary}")