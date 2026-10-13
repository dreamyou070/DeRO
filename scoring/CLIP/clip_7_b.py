import os
import re
from PIL import Image

import torch
import pandas as pd
from transformers import CLIPProcessor, CLIPModel


base_folder = "/lustre/dreamyou070/DeRO/src/english/DERO_Result/SD15"
save_dir = "/lustre/dreamyou070/DeRO/scoring/CLIP/clipscore_SD15"
os.makedirs(save_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model_name = "openai/clip-vit-large-patch14"
clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
clip_model.eval()

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def extract_num_from_filename(filename):
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

sub_folders = sorted(os.listdir(base_folder))

for sub_folder in sub_folders:
    sub_folder_path = os.path.join(base_folder, sub_folder)

    if not os.path.isdir(sub_folder_path):
        continue

    prompt = sub_folder.replace("_", " ")

    files = sorted(os.listdir(sub_folder_path))

    for file in files:
        if not file.lower().endswith(IMG_EXTS):
            continue

        prompt_num = extract_num_from_filename(file)

        if prompt_num is None:
            print(f"[SKIP] num not found: {file}")
            continue

        image_path = os.path.join(sub_folder_path, file)

        try:
            clip_score, cosine_score = compute_clip_score(image_path, prompt)

            results.append({
                "sub_folder": sub_folder,
                "prompt": prompt,
                "prompt_num": prompt_num,
                "file": file,
                "clip_score": clip_score,
                "clip_cosine": cosine_score,
                "image_path": image_path,
            })

            print(
                f"[OK] {sub_folder} | num={prompt_num} | "
                f"CLIP={clip_score:.4f} | cosine={cosine_score:.4f}"
            )

        except Exception as e:
            print(f"[ERROR] {image_path}: {e}")


df = pd.DataFrame(results)

if len(df) == 0:
    print("[ERROR] No valid images found.")
    raise SystemExit


per_sample_path = os.path.join(save_dir, "clipscore_per_sample.csv")
df.to_csv(per_sample_path, index=False)


by_prompt_num = (
    df.groupby("prompt_num")
    .agg(
        num_images=("clip_score", "count"),
        avg_clip_score=("clip_score", "mean"),
        std_clip_score=("clip_score", "std"),
        avg_clip_cosine=("clip_cosine", "mean"),
        std_clip_cosine=("clip_cosine", "std"),
    )
    .reset_index()
    .sort_values("prompt_num")
)

by_prompt_num_path = os.path.join(save_dir, "clipscore_by_prompt_num.csv")
by_prompt_num.to_csv(by_prompt_num_path, index=False)


by_prompt = (
    df.groupby(["sub_folder", "prompt"])
    .agg(
        num_images=("clip_score", "count"),
        avg_clip_score=("clip_score", "mean"),
        std_clip_score=("clip_score", "std"),
        avg_clip_cosine=("clip_cosine", "mean"),
        std_clip_cosine=("clip_cosine", "std"),
    )
    .reset_index()
    .sort_values("avg_clip_score", ascending=False)
)

by_prompt_path = os.path.join(save_dir, "clipscore_by_prompt.csv")
by_prompt.to_csv(by_prompt_path, index=False)


overall = pd.DataFrame([{
    "model_folder": "SD15",
    "num_images": len(df),
    "num_prompts": df["prompt"].nunique(),
    "num_prompt_nums": df["prompt_num"].nunique(),
    "avg_clip_score": df["clip_score"].mean(),
    "std_clip_score": df["clip_score"].std(),
    "avg_clip_cosine": df["clip_cosine"].mean(),
    "std_clip_cosine": df["clip_cosine"].std(),
}])

overall_path = os.path.join(save_dir, "clipscore_overall_summary.csv")
overall.to_csv(overall_path, index=False)

print("=" * 80)
print("Overall Summary")
print(overall)

print("=" * 80)
print("Saved:")
print(per_sample_path)
print(by_prompt_num_path)
print(by_prompt_path)
print(overall_path)