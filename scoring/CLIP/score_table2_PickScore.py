import os
import re
from PIL import Image

import torch
import pandas as pd
from transformers import AutoProcessor, AutoModel


csv_file = "/lustre/dreamyou070/DeRO/src/english/data/DeRO_Data/concept/art_prompts.csv"
#base_folder = "/lustre/dreamyou070/DeRO/src/english/DERO_Result/SD14_Concept"
#save_dir = "/lustre/dreamyou070/DeRO/scoring/PickScore/pickscore_SD14_concept"
base_folder = "/lustre/dreamyou070/DeRO/src/english/DERO_Result/SD15_Concept"
save_dir = "/lustre/dreamyou070/DeRO/scoring/PickScore/pickscore_SD15_concept"
os.makedirs(save_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

processor = AutoProcessor.from_pretrained(processor_name_or_path)
model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp")


# =========================
# Load prompt CSV
# =========================

prompt_df = pd.read_csv(csv_file)

artindex_to_prompt = {
    row["art_index"]: row["prompt"]
    for _, row in prompt_df.iterrows()
}


# =========================
# Helpers
# =========================

def extract_num(filename):
    lower = filename.lower()

    if "base" in lower:
        return "base"

    match = re.search(r"_num_(\d+)", filename)
    if match:
        return int(match.group(1))

    return None


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


# =========================
# Run
# =========================

results = []

for sub_folder in sorted(os.listdir(base_folder)):
    sub_folder_path = os.path.join(base_folder, sub_folder)

    if not os.path.isdir(sub_folder_path):
        continue

    if sub_folder not in artindex_to_prompt:
        print(f"[SKIP] prompt not found for folder: {sub_folder}")
        continue

    prompt = artindex_to_prompt[sub_folder]

    print("=" * 80)
    print(f"Folder : {sub_folder}")
    print(f"Prompt : {prompt}")

    for file in sorted(os.listdir(sub_folder_path)):
        if not file.lower().endswith(IMG_EXTS):
            continue

        prompt_num = extract_num(file)

        if prompt_num is None:
            print(f"[SKIP] num/base not found: {file}")
            continue

        image_path = os.path.join(sub_folder_path, file)

        try:
            pick_score = compute_pickscore(image_path, prompt)

            results.append({
                "folder": sub_folder,
                "prompt": prompt,
                "prompt_num": prompt_num,
                "file": file,
                "pick_score": pick_score,
                "image_path": image_path,
            })

            print(
                f"[OK] {sub_folder} | {prompt_num} | "
                f"PickScore={pick_score:.4f}"
            )

        except Exception as e:
            print(f"[ERROR] {image_path}: {e}")


# =========================
# Save
# =========================

df = pd.DataFrame(results)

if len(df) == 0:
    print("[ERROR] No valid images found.")
    raise SystemExit


per_image_path = os.path.join(save_dir, "pickscore_per_image.csv")
df.to_csv(per_image_path, index=False)


# aux num별 평균
by_aux_num = (
    df.groupby("prompt_num")
    .agg(
        num_images=("pick_score", "count"),
        avg_pick_score=("pick_score", "mean"),
        std_pick_score=("pick_score", "std"),
    )
    .reset_index()
)

by_aux_num = by_aux_num.sort_values(
    "prompt_num",
    key=lambda x: x.astype(str)
)

by_aux_num_path = os.path.join(save_dir, "pickscore_by_aux_num.csv")
by_aux_num.to_csv(by_aux_num_path, index=False)


# folder별 평균
by_folder = (
    df.groupby(["folder", "prompt"])
    .agg(
        num_images=("pick_score", "count"),
        avg_pick_score=("pick_score", "mean"),
        std_pick_score=("pick_score", "std"),
    )
    .reset_index()
    .sort_values("avg_pick_score", ascending=False)
)

by_folder_path = os.path.join(save_dir, "pickscore_by_folder.csv")
by_folder.to_csv(by_folder_path, index=False)


# base만 따로 평균
base_df = df[df["prompt_num"] == "base"]

base_summary = pd.DataFrame([{
    "num_images": len(base_df),
    "avg_pick_score": base_df["pick_score"].mean(),
    "std_pick_score": base_df["pick_score"].std(),
}])

base_path = os.path.join(save_dir, "pickscore_base_summary.csv")
base_summary.to_csv(base_path, index=False)


# 전체 평균
overall = pd.DataFrame([{
    "model_folder": os.path.basename(base_folder),
    "num_images": len(df),
    "num_folders": df["folder"].nunique(),
    "num_prompt_nums": df["prompt_num"].nunique(),
    "avg_pick_score": df["pick_score"].mean(),
    "std_pick_score": df["pick_score"].std(),
}])

overall_path = os.path.join(save_dir, "pickscore_overall_summary.csv")
overall.to_csv(overall_path, index=False)


print("=" * 80)
print("Saved:")
print(per_image_path)
print(by_aux_num_path)
print(by_folder_path)
print(base_path)
print(overall_path)