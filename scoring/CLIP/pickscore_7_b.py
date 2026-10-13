import os
from PIL import Image
import torch
import pandas as pd
from transformers import AutoProcessor, AutoModel

# ==========================================
# Setting
# ==========================================

base_folder = "/lustre/dreamyou070/DeRO/src/english/DERO_Result/FLUX_bidirectional_extraction_method_example_chocolate_cup"
prompt = "a cookie without chocolate chip and a glass with nothing on it"

device = "cuda" if torch.cuda.is_available() else "cpu"

processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

processor = AutoProcessor.from_pretrained(processor_name_or_path)
model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp")


# ==========================================
# Function
# ==========================================

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

    with torch.no_grad():
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
print(f"Base folder: {base_folder}")
print(f"Found images: {len(image_paths)}")
print(f"Prompt: {prompt}")

results = []

for img_path in image_paths:
    file = os.path.basename(img_path)
    subfolder = os.path.relpath(os.path.dirname(img_path), base_folder)

    try:
        pick_score = compute_pickscore(img_path, prompt)

        results.append({
            "subfolder": subfolder,
            "file": file,
            "prompt": prompt,
            "pick_score": pick_score,
            "image_path": img_path,
        })

        print(f"[OK] {file} | PickScore: {pick_score:.4f}")

    except Exception as e:
        print(f"[ERROR] {img_path}: {e}")


# ==========================================
# Save
# ==========================================

df = pd.DataFrame(results)
df = df.sort_values("pick_score", ascending=False)

df.to_csv("pickscore_per_sample.csv", index=False)

summary = pd.DataFrame([{
    "num_images": len(df),
    "avg_pick_score": df["pick_score"].mean(),
    "std_pick_score": df["pick_score"].std(),
    "best_file": df.iloc[0]["file"] if len(df) > 0 else None,
    "best_pick_score": df.iloc[0]["pick_score"] if len(df) > 0 else None,
}])

summary.to_csv("pickscore_summary.csv", index=False)

print("=" * 80)
print("Summary")
print(summary)

print("=" * 80)
print("Saved:")
print("- pickscore_per_sample.csv")
print("- pickscore_summary.csv")