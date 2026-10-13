import os
import re
from PIL import Image

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel


# ==========================================
# Setting
# ==========================================
base_folder = "/lustre/dreamyou070/DeRO/src/english/Ablation_study2/SD15"

save_per_image = "sd15_factor_pickscore_per_image.csv"
save_summary = "sd15_factor_pickscore_summary.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"

processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp")


# ==========================================
# Load PickScore
# ==========================================
processor = AutoProcessor.from_pretrained(processor_name_or_path)
model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)


# ==========================================
# Functions
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


def extract_num(filename):
    match = re.search(r"_num_(\d+)", filename)
    return int(match.group(1)) if match else None


def extract_factor_from_filename(filename):
    match = re.search(r"factor_(\d+)", filename)
    return f"factor_{match.group(1)}" if match else None


def make_prompt_from_filename(filename):
    name, _ = os.path.splitext(filename)

    # Example:
    # a_cake_without_a_cream_num_1_clean_code_True_normalize_False_extraction_factor_2.png
    # -> a cake without a cream
    name = re.sub(r"_num_\d+.*$", "", name)

    prompt = name.replace("_", " ")
    prompt = " ".join(prompt.split())

    return prompt


# ==========================================
# Run
# ==========================================
results = []

sample_folders = sorted(os.listdir(base_folder))

for sample_folder in sample_folders:
    sample_folder_path = os.path.join(base_folder, sample_folder)

    if not os.path.isdir(sample_folder_path):
        continue

    factor_folders = sorted(os.listdir(sample_folder_path))

    for factor_folder in factor_folders:
        factor_folder_path = os.path.join(sample_folder_path, factor_folder)

        if not os.path.isdir(factor_folder_path):
            continue

        if factor_folder == "base":
            continue

        files = sorted([
            f for f in os.listdir(factor_folder_path)
            if f.lower().endswith(IMG_EXTS)
        ])

        for file in tqdm(files, desc=f"{sample_folder}/{factor_folder}"):
            image_path = os.path.join(factor_folder_path, file)

            num = extract_num(file)
            factor_from_file = extract_factor_from_filename(file)
            prompt = make_prompt_from_filename(file)

            if factor_from_file is not None and factor_from_file != factor_folder:
                print(
                    f"[Warning] factor mismatch: "
                    f"folder={factor_folder}, file={factor_from_file}, path={image_path}"
                )

            try:
                pick_score = compute_pickscore(image_path, prompt)

                results.append({
                    "sample_folder": sample_folder,
                    "factor_folder": factor_folder,
                    "num": num,
                    "file": file,
                    "prompt": prompt,
                    "pick_score": pick_score,
                    "image_path": image_path,
                })

            except Exception as e:
                print(f"[ERROR] {image_path}: {e}")


# ==========================================
# Save
# ==========================================
df = pd.DataFrame(results)

if len(df) == 0:
    print("[Warning] No valid images were scored.")
    exit(0)

df = df.sort_values(["sample_folder", "num", "factor_folder"])
df.to_csv(save_per_image, index=False)

summary = (
    df.groupby(["factor_folder", "num"])
    .agg(
        num_images=("pick_score", "count"),
        avg_pick_score=("pick_score", "mean"),
        std_pick_score=("pick_score", "std"),
        min_pick_score=("pick_score", "min"),
        max_pick_score=("pick_score", "max"),
    )
    .reset_index()
    .sort_values(["num", "factor_folder"])
)

summary.to_csv(save_summary, index=False)


# ==========================================
# Print
# ==========================================
print("=" * 80)
print("Summary by factor_folder and num")
print("=" * 80)
print(summary.to_string(index=False))

print("=" * 80)
print("Saved:")
print(f"- {save_per_image}")
print(f"- {save_summary}")