import os
import re
from PIL import Image
import torch
import pandas as pd
from transformers import CLIPProcessor, CLIPModel


device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model_name = "openai/clip-vit-large-patch14"
clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
clip_model.eval()

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp")


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
            clip_score, cosine_score = compute_clip_score(img_path, prompt)

            all_results.append({
                "model": model_name,
                "subfolder": subfolder,
                "aux_num": aux_num,
                "file": file,
                "prompt": prompt,
                "image_path": img_path,
                "clip_score": clip_score,
                "clip_cosine": cosine_score,
            })

        except Exception as e:
            print(f"[ERROR] {model_name} | {img_path}: {e}")


df = pd.DataFrame(all_results)

df.to_csv("clipscore_aux_per_image.csv", index=False)

summary = (
    df.groupby(["model", "aux_num"], as_index=False)
      .agg(
          num_images=("clip_score", "count"),
          avg_clip_score=("clip_score", "mean"),
          std_clip_score=("clip_score", "std"),
          avg_clip_cosine=("clip_cosine", "mean"),
          std_clip_cosine=("clip_cosine", "std"),
      )
      .sort_values(["model", "aux_num"])
)

summary.to_csv("clipscore_aux_summary.csv", index=False)

print("=" * 80)
print("Aux-wise summary")
print(summary)

print("=" * 80)
print("Saved:")
print("- clipscore_aux_per_image.csv")
print("- clipscore_aux_summary.csv")