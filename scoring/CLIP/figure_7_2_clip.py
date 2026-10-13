import os
import re
from PIL import Image

import torch
import pandas as pd
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel


# ==========================================
# Setting
# ==========================================
base_folder = "/lustre/dreamyou070/DeRO/src/english/Ablation_study2/SD15"
save_per_image = "sd15_factor_clip_scores_per_image.csv"
save_summary = "sd15_factor_clip_scores_summary.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model_name = "openai/clip-vit-large-patch14"

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp")


# ==========================================
# Load CLIP
# ==========================================
clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
clip_model.eval()


# ==========================================
# Functions
# ==========================================
@torch.no_grad()
def compute_clip_score(image_path, prompt, w=2.5):
    image = Image.open(image_path).convert("RGB")

    inputs = clip_processor(
        text=[prompt],
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    outputs = clip_model(**inputs)

    image_embeds = outputs.image_embeds
    text_embeds = outputs.text_embeds

    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    cosine = (image_embeds @ text_embeds.T)[0, 0]
    clip_score = w * torch.clamp(cosine, min=0.0)

    return clip_score.item(), cosine.item()


def extract_num(filename):
    """
    Example:
    a_cake_without_a_cream_num_1_clean_code_True_normalize_False_extraction_factor_2.png
    -> 1
    """
    match = re.search(r"_num_(\d+)", filename)
    return int(match.group(1)) if match else None


def extract_factor_from_filename(filename):
    match = re.search(r"factor_(\d+)", filename)
    return f"factor_{match.group(1)}" if match else None


def make_prompt_from_filename(filename):
    """
    Remove experimental suffix and convert filename to prompt.
    Example:
    a_cake_without_a_cream_num_1_clean_code_True_normalize_False_extraction_factor_2.png
    -> a cake without a cream
    """
    name, _ = os.path.splitext(filename)

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

        # base는 제외하고 싶으면 아래 유지
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

            # 파일명 factor와 폴더 factor가 다르면 warning만 찍고 폴더 기준으로 저장
            if factor_from_file is not None and factor_from_file != factor_folder:
                print(f"[Warning] factor mismatch: folder={factor_folder}, file={factor_from_file}, path={image_path}")

            try:
                clip_score, cosine = compute_clip_score(image_path, prompt)

                results.append({
                    "sample_folder": sample_folder,
                    "factor_folder": factor_folder,
                    "num": num,
                    "file": file,
                    "prompt": prompt,
                    "clip_score": clip_score,
                    "cosine": cosine,
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
        num_images=("clip_score", "count"),
        avg_clip_score=("clip_score", "mean"),
        std_clip_score=("clip_score", "std"),
        avg_cosine=("cosine", "mean"),
        std_cosine=("cosine", "std"),
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