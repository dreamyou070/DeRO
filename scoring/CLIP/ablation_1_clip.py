import os
import re
from PIL import Image

import torch
import pandas as pd
from transformers import CLIPProcessor, CLIPModel


base_folder = "/lustre/dreamyou070/DeRO/src/english/Ablation_Study_1"
save_dir = "/lustre/dreamyou070/DeRO/scoring/CLIP/ablation_study_1"
os.makedirs(save_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model_name = "openai/clip-vit-large-patch14"
clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
clip_model.eval()

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def extract_n_from_filename(filename):
    lower = filename.lower()

    if "base" in lower:
        return 0

    if "normalize_false" not in lower:
        return None

    match = re.search(r"_num_(\d+)", filename)
    if match is None:
        return None

    return int(match.group(1))


def compute_clip_score(image_path, prompt, w=2.5):
    image = Image.open(image_path).convert("RGB")

    inputs = clip_processor(
        text=[prompt],
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        outputs = clip_model(**inputs)

        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        cosine = (image_embeds @ text_embeds.T)[0, 0]
        clip_score = w * torch.clamp(cosine, min=0.0)

    return clip_score.item(), cosine.item()


results = []

models = sorted(os.listdir(base_folder))

for model in models:
    model_path = os.path.join(base_folder, model)

    if not os.path.isdir(model_path):
        continue

    sub_folders = sorted(os.listdir(model_path))

    for sub_folder in sub_folders:
        sub_folder_path = os.path.join(model_path, sub_folder)

        if not os.path.isdir(sub_folder_path):
            continue

        prompt = sub_folder.replace("_", " ")

        files = sorted(os.listdir(sub_folder_path))

        for file in files:
            if not file.lower().endswith(IMG_EXTS):
                continue

            n = extract_n_from_filename(file)

            if n is None:
                print(f"[SKIP] {model} | {sub_folder} | {file}")
                continue

            image_path = os.path.join(sub_folder_path, file)

            try:
                clip_score, clip_cosine = compute_clip_score(
                    image_path=image_path,
                    prompt=prompt,
                )

                results.append({
                    "model": model,
                    "prompt_folder": sub_folder,
                    "prompt": prompt,
                    "n": n,
                    "file": file,
                    "clip_score": clip_score,
                    "clip_cosine": clip_cosine,
                    "image_path": image_path,
                })

                print(
                    f"[OK] model={model} | n={n} | "
                    f"CLIP={clip_score:.4f} | {file}"
                )

            except Exception as e:
                print(f"[ERROR] {image_path}: {e}")


df = pd.DataFrame(results)

if len(df) == 0:
    print("[ERROR] No valid images found.")
    raise SystemExit


per_image_path = os.path.join(save_dir, "clipscore_per_image.csv")
df.to_csv(per_image_path, index=False)


by_model_n = (
    df.groupby(["model", "n"])
    .agg(
        num_images=("clip_score", "count"),
        avg_clip_score=("clip_score", "mean"),
        std_clip_score=("clip_score", "std"),
        avg_clip_cosine=("clip_cosine", "mean"),
        std_clip_cosine=("clip_cosine", "std"),
    )
    .reset_index()
    .sort_values(["model", "n"])
)

by_model_n_path = os.path.join(save_dir, "clipscore_by_model_n.csv")
by_model_n.to_csv(by_model_n_path, index=False)


by_prompt_n = (
    df.groupby(["model", "prompt_folder", "prompt", "n"])
    .agg(
        num_images=("clip_score", "count"),
        avg_clip_score=("clip_score", "mean"),
        std_clip_score=("clip_score", "std"),
        avg_clip_cosine=("clip_cosine", "mean"),
        std_clip_cosine=("clip_cosine", "std"),
    )
    .reset_index()
    .sort_values(["model", "prompt_folder", "n"])
)

by_prompt_n_path = os.path.join(save_dir, "clipscore_by_prompt_n.csv")
by_prompt_n.to_csv(by_prompt_n_path, index=False)


pivot_path = os.path.join(save_dir, "clipscore_pivot_model_by_n.csv")

pivot = by_model_n.pivot(
    index="model",
    columns="n",
    values="avg_clip_score",
).reset_index()

pivot.to_csv(pivot_path, index=False)


print("=" * 80)
print("Saved:")
print(per_image_path)
print(by_model_n_path)
print(by_prompt_n_path)
print(pivot_path)