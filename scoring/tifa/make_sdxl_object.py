import os
import json
import re

image_dir = "/lustre/dreamyou070/DeRO/src/english/DERO_Result/Model_StableDiffusionXL/base"
save_dir = "human_annotations/sample"
os.makedirs(save_dir, exist_ok=True)

qa_save_path = os.path.join(save_dir, "sample_sdxl_object_question_answering.json")
img_save_path = os.path.join(save_dir, "sample_sdxl_object.json")

def filename_to_prompt(filename):
    name = os.path.splitext(filename)[0]
    name = re.sub(r"_sim_threshold_\d+(\.\d+)?", "", name)
    return name.replace("_", " ")

def extract_negated_object(prompt):
    patterns = [
        r"without (?:a |an |the )?(.+)$",
        r"with no (.+)$",
        r"no (.+)$",
    ]
    for p in patterns:
        m = re.search(p, prompt)
        if m:
            return m.group(1).strip()
    return None

qa_list = []
img_list = []

for filename in sorted(os.listdir(image_dir)):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        continue

    prompt = filename_to_prompt(filename)
    target = extract_negated_object(prompt)

    if target is None:
        print(f"[SKIP] cannot extract target: {filename}")
        continue

    sample_id = os.path.splitext(filename)[0]
    image_path = os.path.join(image_dir, filename)

    img_list.append({
        "id": sample_id,
        "image_path": image_path
    })

    qa_list.append({
        "id": sample_id,
        "caption": prompt,
        "question": f"is there {target} in the image?",
        "choices": ["yes", "no"],
        "answer": "no",
        "element_type": "object_negation",
        "element": target
    })

with open(qa_save_path, "w", encoding="utf-8") as f:
    json.dump(qa_list, f, indent=4, ensure_ascii=False)

with open(img_save_path, "w", encoding="utf-8") as f:
    json.dump(img_list, f, indent=4, ensure_ascii=False)

print(f"Saved QA file: {qa_save_path}")
print(f"Saved image file: {img_save_path}")
print(f"Total samples: {len(img_list)}")