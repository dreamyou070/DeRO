import os
import re
from PIL import Image

import torch
import pandas as pd
from transformers import AutoProcessor, AutoModel


base_folder = "/lustre/dreamyou070/DeRO/src/english/Ablation_Study_1"
save_dir = "/lustre/dreamyou070/DeRO/scoring/PickScore/ablation_study_1"
os.makedirs(save_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

processor = AutoProcessor.from_pretrained(processor_name_or_path)
model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)

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


results = []

for model_name in sorted(os.listdir(base_folder)):
    model_path = os.path.join(base_folder, model_name)

    if not os.path.isdir(model_path):
        continue

    for sub_folder in sorted(os.listdir(model_path)):
        sub_folder_path = os.path.join(model_path, sub_folder)

        if not os.path.isdir(sub_folder_path):
            continue

        prompt = sub_folder.replace("_", " ")

        for file in sorted(os.listdir(sub_folder_path)):
            if not file.lower().endswith(IMG_EXTS):
                continue

            n = extract_n_from_filename(file)

            if n is None:
                print(f"[SKIP] {model_name} | {sub_folder} | {file}")
                continue

            image_path = os.path.join(sub_folder_path, file)

            try:
                pick_score = compute_pickscore(image_path, prompt)

                results.append({
                    "model": model_name,
                    "prompt_folder": sub_folder,
                    "prompt": prompt,
                    "n": n,
                    "file": file,
                    "pick_score": pick_score,
                    "image_path": image_path,
                })

                print(
                    f"[OK] model={model_name} | n={n} | "
                    f"PickScore={pick_score:.4f} | {file}"
                )

            except Exception as e:
                print(f"[ERROR] {image_path}: {e}")


df = pd.DataFrame(results)

if len(df) == 0:
    print("[ERROR] No valid images found.")
    raise SystemExit


per_image_path = os.path.join(save_dir, "pickscore_per_image.csv")
df.to_csv(per_image_path, index=False)


by_model_n = (
    df.groupby(["model", "n"])
    .agg(
        num_images=("pick_score", "count"),
        avg_pick_score=("pick_score", "mean"),
        std_pick_score=("pick_score", "std"),
    )
    .reset_index()
    .sort_values(["model", "n"])
)

by_model_n_path = os.path.join(save_dir, "pickscore_by_model_n.csv")
by_model_n.to_csv(by_model_n_path, index=False)


by_prompt_n = (
    df.groupby(["model", "prompt_folder", "prompt", "n"])
    .agg(
        num_images=("pick_score", "count"),
        avg_pick_score=("pick_score", "mean"),
        std_pick_score=("pick_score", "std"),
    )
    .reset_index()
    .sort_values(["model", "prompt_folder", "n"])
)

by_prompt_n_path = os.path.join(save_dir, "pickscore_by_prompt_n.csv")
by_prompt_n.to_csv(by_prompt_n_path, index=False)


pivot = by_model_n.pivot(
    index="model",
    columns="n",
    values="avg_pick_score",
).reset_index()

pivot_path = os.path.join(save_dir, "pickscore_pivot_model_by_n.csv")
pivot.to_csv(pivot_path, index=False)


print("=" * 80)
print("Saved:")
print(per_image_path)
print(by_model_n_path)
print(by_prompt_n_path)
print(pivot_path)