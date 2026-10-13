import os
import json
from glob import glob

folders = {
    "sdxl_object_base": "/lustre/dreamyou070/DeRO/src/english/DERO_Result/ablation_study_sdxl_base_uni_bi/base",
    "sdxl_object_ours": "/lustre/dreamyou070/DeRO/src/english/DERO_Result/ablation_study_sdxl_base_uni_bi/bidirectional",
    "sdxl_concept_base": "/lustre/dreamyou070/DeRO/src/english/DERO_Result/concept_ablation_study_sdxl_base_uni_bi/base",
    "sdxl_concept_ours": "/lustre/dreamyou070/DeRO/src/english/DERO_Result/concept_ablation_study_sdxl_base_uni_bi/bidirectional",
}

save_dir = "./sample"
os.makedirs(save_dir, exist_ok=True)

def clean_caption_from_filename(stem):
    caption = stem

    # suffix 제거
    for suffix in ["_base", "_bidirectional", "_ours"]:
        if caption.endswith(suffix):
            caption = caption[: -len(suffix)]

    # sim threshold 제거
    parts = caption.split("_sim_threshold")
    caption = parts[0]

    # 파일명 → 문장
    caption = caption.replace("_", " ")
    caption = caption.replace(",", ", ")
    caption = " ".join(caption.split())

    return caption

def make_simple_questions(caption_id, caption):
    return [
        {
            "id": caption_id,
            "caption": caption,
            "question": "does this image match the caption?",
            "choices": ["yes", "no"],
            "answer": "yes",
            "element_type": "caption",
            "element": caption,
        }
    ]

for name, folder in folders.items():
    img_json = {}
    qa_json = []

    img_paths = []
    for ext in ["png", "jpg", "jpeg", "webp"]:
        img_paths.extend(glob(os.path.join(folder, f"*.{ext}")))
        img_paths.extend(glob(os.path.join(folder, f"*.{ext.upper()}")))

    for img_path in sorted(img_paths):
        file = os.path.basename(img_path)
        stem, ext = os.path.splitext(file)

        caption_id = stem
        caption = clean_caption_from_filename(stem)

        img_json[caption_id] = img_path
        qa_json.extend(make_simple_questions(caption_id, caption))

    img_save_path = os.path.join(save_dir, f"{name}_imgs.json")
    qa_save_path = os.path.join(save_dir, f"{name}_question_answers.json")

    with open(img_save_path, "w") as f:
        json.dump(img_json, f, indent=4)

    with open(qa_save_path, "w") as f:
        json.dump(qa_json, f, indent=4)

    print("=" * 80)
    print(name)
    print("num images:", len(img_json))
    print("num questions:", len(qa_json))
    print("saved imgs:", img_save_path)
    print("saved qa:", qa_save_path)
    if img_json:
        print("example id:", next(iter(img_json.keys())))
        print("example image:", next(iter(img_json.values())))
        print("example qa:", qa_json[0])