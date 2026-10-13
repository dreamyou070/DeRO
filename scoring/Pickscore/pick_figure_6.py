import os
from PIL import Image
import torch
import pandas as pd
from transformers import AutoProcessor, AutoModel


device = "cuda" if torch.cuda.is_available() else "cpu"

processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

processor = AutoProcessor.from_pretrained(processor_name_or_path)
model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)


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

        score = model.logit_scale.exp() * (text_embs @ image_embs.T)[0, 0]

    return score.item()


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
    scores = []

    files = sorted(os.listdir(model_path))

    for file in files:
        if not file.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            continue

        name, ext = os.path.splitext(file)
        prompt = name.replace("_", " ")
        img_path = os.path.join(model_path, file)

        try:
            pick_score = compute_pickscore(img_path, prompt)
            scores.append(pick_score)

            all_results.append({
                "model": model_name,
                "file": file,
                "prompt": prompt,
                "image_path": img_path,
                "pick_score": pick_score,
            })

        except Exception as e:
            print(f"[ERROR] {model_name} | {file}: {e}")

    avg_score = sum(scores) / len(scores) if len(scores) > 0 else 0.0

    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Num images: {len(scores)}")
    print(f"Average PickScore: {avg_score:.4f}")

df = pd.DataFrame(all_results)
df.to_csv("pickscore_results.csv", index=False)

print("=" * 80)
print("Saved detailed results to pickscore_results.csv")