import os
from PIL import Image
import torch
import pandas as pd
from transformers import CLIPProcessor, CLIPModel


device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model_name = "openai/clip-vit-large-patch14"
clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
clip_model.eval()


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


folders = {
    "sdxl_object_base": "/lustre/dreamyou070/DeRO/src/english/DERO_Result/ablation_study_sdxl_base_uni_bi/base",
    "sdxl_object_ours": "/lustre/dreamyou070/DeRO/src/english/DERO_Result/ablation_study_sdxl_base_uni_bi/bidirectional",
    "sdxl_object_uni": "/lustre/dreamyou070/DeRO/src/english/DERO_Result/ablation_study_sdxl_base_uni_bi/unidirectional",

    "flux_object_base": "/lustre/dreamyou070/DeRO/src/english/DERO_Result/ablation_study_flux_base_uni_bi/base",
    "flux_object_ours": "/lustre/dreamyou070/DeRO/src/english/DERO_Result/ablation_study_flux_base_uni_bi/bidirectional",

    "sdxl_concept_base": "/lustre/dreamyou070/DeRO/src/english/DERO_Result/concept_ablation_study_sdxl_base_uni_bi/base",
    "sdxl_concept_ours": "/lustre/dreamyou070/DeRO/src/english/DERO_Result/concept_ablation_study_sdxl_base_uni_bi/bidirectional",

    "flux_concept_base": "/lustre/dreamyou070/DeRO/src/english/DERO_Result/concept_ablation_study_flux_base_uni_bi/base",
    "flux_concept_ours": "/lustre/dreamyou070/DeRO/src/english/DERO_Result/concept_ablation_study_flux_base_uni_bi/bidirectional",
    "flux_concept_uni": "/lustre/dreamyou070/DeRO/src/english/DERO_Result/concept_ablation_study_flux_base_uni_bi/unidirectional",
}

all_results = []

for model_name, model_path in folders.items():
    clip_scores = []
    cosine_scores = []

    files = sorted(os.listdir(model_path))

    for file in files:
        if not file.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            continue

        name, ext = os.path.splitext(file)
        prompt = name.replace("_", " ")
        img_path = os.path.join(model_path, file)

        try:
            clip_score, cosine_score = compute_clip_score(img_path, prompt)

            clip_scores.append(clip_score)
            cosine_scores.append(cosine_score)

            all_results.append({
                "model": model_name,
                "file": file,
                "prompt": prompt,
                "image_path": img_path,
                "clip_score": clip_score,
                "clip_cosine": cosine_score,
            })

        except Exception as e:
            print(f"[ERROR] {model_name} | {file}: {e}")

    avg_clip_score = sum(clip_scores) / len(clip_scores) if len(clip_scores) > 0 else 0.0
    avg_cosine_score = sum(cosine_scores) / len(cosine_scores) if len(cosine_scores) > 0 else 0.0

    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Num images: {len(clip_scores)}")
    print(f"Average CLIPScore: {avg_clip_score:.4f}")
    print(f"Average CLIP Cosine: {avg_cosine_score:.4f}")

df = pd.DataFrame(all_results)
df.to_csv("clipscore_results.csv", index=False)

print("=" * 80)
print("Saved detailed results to clipscore_results.csv")