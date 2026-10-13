import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel


# ==========================================
# Setting
# ==========================================
csv_path = "/lustre/dreamyou070/DeRO/src/english/data/DeRO_Data/concept/art_prompts.csv"
base_folder = "/lustre/dreamyou070/DeRO/src/english/DERO_Result/SD15_Concept_Base_Model"
save_path = "/lustre/dreamyou070/DeRO/src/english/DERO_Result/sd15_concept_clip_scores.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"
image_exts = {".png", ".jpg", ".jpeg", ".webp"}


# ==========================================
# Load CLIP
# ==========================================
model_name = "openai/clip-vit-large-patch14"
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name).to(device)
model.eval()


# ==========================================
# Load CSV
# ==========================================
df_prompt = pd.read_csv(csv_path)

required_cols = {"id", "prompt", "artist_concept", "art_index", "auxiliary_semantic"}
missing = required_cols - set(df_prompt.columns)
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

prompt_dict = {
    str(row["art_index"]): {
        "id": row["id"],
        "prompt": row["prompt"],
        "artist_concept": row["artist_concept"],
        "auxiliary_semantic": row["auxiliary_semantic"],
    }
    for _, row in df_prompt.iterrows()
}


# ==========================================
# CLIP Score Function
# ==========================================
@torch.no_grad()
def compute_clip_score(image_path, text):
    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        text=[text],
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
    text_features = model.get_text_features(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
    )

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    score = (image_features @ text_features.T).item()
    return score


# ==========================================
# Iterate Models
# ==========================================
results = []

models = sorted([
    m for m in os.listdir(base_folder)
    if os.path.isdir(os.path.join(base_folder, m))
])

for model_name_dir in models:
    model_path = os.path.join(base_folder, model_name_dir)

    files = sorted([
        f for f in os.listdir(model_path)
        if os.path.splitext(f)[1].lower() in image_exts
    ])

    print(f"[Model] {model_name_dir}: {len(files)} images")

    for file in tqdm(files, desc=model_name_dir):
        name, ext = os.path.splitext(file)

        if name not in prompt_dict:
            print(f"[Skip] No matching art_index in CSV: {name}")
            continue

        info = prompt_dict[name]
        image_path = os.path.join(model_path, file)

        try:
            clip_score = compute_clip_score(image_path, info["prompt"])
        except Exception as e:
            print(f"[Error] {image_path}: {e}")
            continue

        results.append({
            "model": model_name_dir,
            "file": file,
            "art_index": name,
            "id": info["id"],
            "artist_concept": info["artist_concept"],
            "prompt": info["prompt"],
            "auxiliary_semantic": info["auxiliary_semantic"],
            "clip_score": clip_score,
        })


# ==========================================
# Save
# ==========================================
df_result = pd.DataFrame(results)
df_result.to_csv(save_path, index=False)

print(f"\nSaved: {save_path}")
print(df_result.groupby("model")["clip_score"].mean().sort_values(ascending=False))