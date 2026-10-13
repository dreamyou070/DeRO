import os
import json
import re

os.makedirs("./sample", exist_ok=True)

image_dirs = [
    "/lustre/dreamyou070/DeRO/src/english/DERO_Result/concept_ablation_study_sdxl_base_uni_bi/base",
    "/lustre/dreamyou070/DeRO/src/english/DERO_Result/concept_ablation_study_sdxl_base_uni_bi/bidirectional",
    "/lustre/dreamyou070/DeRO/src/english/DERO_Result/ablation_study_sdxl_base_uni_bi/base",
    "/lustre/dreamyou070/DeRO/src/english/DERO_Result/ablation_study_sdxl_base_uni_bi/bidirectional",
]

qa_save_paths = [
    "./sample/sample_sdxl_concept_question_answering_base.json",
    "./sample/sample_sdxl_concept_question_answering_bidirectional.json",
    "./sample/sample_sdxl_object_question_answering_base.json",
    "./sample/sample_sdxl_object_question_answering_bidirectional.json",
]

img_save_paths = [
    "./sample/sample_sdxl_concept_base.json",
    "./sample/sample_sdxl_concept_bidirectional.json",
    "./sample/sample_sdxl_object_base.json",
    "./sample/sample_sdxl_object_bidirectional.json",
]

task_types = ["concept", "concept", "object", "object"]


def is_image_file(filename):
    return filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))


def strip_extra_suffix(name):
    name = re.sub(r"_sim_threshold_\d+(?:\.\d+)?", "", name)
    name = re.sub(r"_(base|bidirectional)$", "", name)
    return name


def parse_concept(filename):
    file_id = os.path.splitext(filename)[0]
    clean = strip_extra_suffix(file_id)

    match = re.search(r"not_in_the_style_of_([^,]+)", clean)
    if match is None:
        return None

    artist = match.group(1).replace("_", " ")
    caption = clean.replace("_", " ")

    return {
        "caption": caption,
        "question": f"is this image in the style of {artist}?",
        "choices": ["yes", "no"],
        "answer": "no",
        "element_type": "concept_negation",
        "element": artist,
    }


def parse_object(filename):
    file_id = os.path.splitext(filename)[0]
    clean = strip_extra_suffix(file_id)
    caption = clean.replace("_", " ")

    target = None
    patterns = [
        r"\bwithout (?:a |an |the )?(.+)$",
        r"\bwith no (?:a |an |the )?(.+)$",
    ]

    for pattern in patterns:
        match = re.search(pattern, caption)
        if match:
            target = match.group(1).strip()
            break

    if target is None:
        return None

    target = re.sub(r"\s+", " ", target).strip(" .,")

    return {
        "caption": caption,
        "question": f"is there {target} in the image?",
        "choices": ["yes", "no"],
        "answer": "no",
        "element_type": "object_negation",
        "element": target,
    }


def build_json(image_dir, qa_save_path, img_save_path, task_type):
    qa_data = []
    img_data = []

    for filename in sorted(os.listdir(image_dir)):
        if not is_image_file(filename):
            continue

        file_id = os.path.splitext(filename)[0]
        image_path = os.path.join(image_dir, filename)

        parsed = parse_concept(filename) if task_type == "concept" else parse_object(filename)

        if parsed is None:
            print(f"[SKIP] {task_type}: {filename}")
            continue

        qa_data.append({
            "id": file_id,
            **parsed,
        })

        img_data.append({
            "id": file_id,
            "image_path": image_path,
        })

    with open(qa_save_path, "w", encoding="utf-8") as f:
        json.dump(qa_data, f, indent=4, ensure_ascii=False)

    with open(img_save_path, "w", encoding="utf-8") as f:
        json.dump(img_data, f, indent=4, ensure_ascii=False)

    print(f"[DONE] {task_type}")
    print(f"QA  : {qa_save_path} ({len(qa_data)})")
    print(f"IMG : {img_save_path} ({len(img_data)})")


for image_dir, qa_path, img_path, task_type in zip(
    image_dirs, qa_save_paths, img_save_paths, task_types
):
    build_json(image_dir, qa_path, img_path, task_type)