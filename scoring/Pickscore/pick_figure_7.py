import os
import re
from PIL import Image
import torch
import pandas as pd
from transformers import AutoProcessor, AutoModel


device = "cuda" if torch.cuda.is_available() else "cpu"

processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

processor = AutoProcessor.from_pretrained(processor_name_or_path)
model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp")


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
        cosine_score = (text_embs @ image_embs.T)[0, 0]

    return pick_score.item(), cosine_score.item()


def extract_aux_num(filename):
    match = re.search(r"aux_(\d+)", filename)
    if match is None:
        return None
    return int(match.group(1))


def make_prompt_from_filename(filename):
    name, _ = os.path.splitext(filename)

    # aux/style_ratio 등 실험 suffix 제거
    name = re.sub(r"_aux_\d+.*$", "", name)

    prompt = name.replace("_", " ")
    prompt = " ".join(prompt.split())
    return prompt


def collect_images_recursive(root_dir):
    image_paths = []

    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.lower().endswith(IMG_EXTS):
                image_paths.append(os.path.join(dirpath, file))

    return sorted(image_paths)


folders = {
    "sdxl": "/lustre/dreamyou070/DeRO/src/english/DERO_Result/ablation_study_sdxl_prompt_amount",
    "flux": "/lustre/dreamyou070/DeRO/src/english/DERO_Result/ablation_study_flux_prompt_amount",
}


all_results = []

for model_name, model_path in folders.items():
    image_paths = collect_images_recursive(model_path)

    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Found images: {len(image_paths)}")

    for img_path in image_paths:
        file = os.path.basename(img_path)
        aux_num = extract_aux_num(file)

        if aux_num is None:
            print(f"[SKIP: no aux] {img_path}")
            continue

        prompt = make_prompt_from_filename(file)
        subfolder = os.path.relpath(os.path.dirname(img_path), model_path)

        try:
            pick_score, cosine_score = compute_pickscore(img_path, prompt)

            all_results.append({
                "model": model_name,
                "subfolder": subfolder,
                "aux_num": aux_num,
                "file": file,
                "prompt": prompt,
                "image_path": img_path,
                "pick_score": pick_score,
                "pick_cosine": cosine_score,
            })

        except Exception as e:
            print(f"[ERROR] {model_name} | {img_path}: {e}")


df = pd.DataFrame(all_results)

df.to_csv("pickscore_aux_per_image.csv", index=False)

summary = (
    df.groupby(["model", "aux_num"], as_index=False)
      .agg(
          num_images=("pick_score", "count"),
          avg_pick_score=("pick_score", "mean"),
          std_pick_score=("pick_score", "std"),
          avg_pick_cosine=("pick_cosine", "mean"),
          std_pick_cosine=("pick_cosine", "std"),
      )
      .sort_values(["model", "aux_num"])
)

summary.to_csv("pickscore_aux_summary.csv", index=False)

print("=" * 80)
print("Aux-wise PickScore summary")
print(summary)

print("=" * 80)
print("Saved:")
print("- pickscore_aux_per_image.csv")
print("- pickscore_aux_summary.csv")