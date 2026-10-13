import os
import json
import csv
import subprocess

csv_file = "data/object.csv"

json_folder = "/lustre/dreamyou070/DeRO/src/english/data/DeRO_Data/object"
python_file = "step1_generate_aux_and_sort.py"

cuda_visible_devices = "0"
aux_num = 20
batch_generate = 20
task = "object"
target_word = 'without' #"object" #'nobody'

def has_existing_prompts(json_path):
    if not os.path.exists(json_path):
        return False

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) == 0:
        return False

    item = data[0]

    biased = item.get("base_entity_biased_prompts", [])
    absent = item.get("base_entity_bias_absent_prompts", [])

    return len(biased) > 0 and len(absent) > 0


def ensure_json_file(json_path, row):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    if os.path.exists(json_path):
        return

    item = {
        "id": int(row["Index"]),
        "prompt": row["Prompt"],
        "base_entity_biased_prompts": [],
        "base_entity_bias_absent_prompts": [],
        "class": row["class"],
        "class_index": int(row["class_index"]),
        "filename": row["file_name"]
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([item], f, ensure_ascii=False, indent=2)


def run_generation(row, json_path):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    cmd = [
        "python", python_file,
        "--id", str(row["Index"]),
        "--prompt", row["Prompt"],
        "--biased_semantic", row["bias"],
        "--bias_absent_semantic", row["semantic"],
        "--aux_num", str(aux_num),
        "--batch_generate", str(batch_generate),
        "--output_file", json_path,
        "--task", task,
    ]

    print("\n" + "=" * 80)
    print(f"[RUN] {row['file_name']}")
    print(" ".join(cmd))
    print("=" * 80)

    subprocess.run(cmd, env=env, check=True)


with open(csv_file, "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)

    for row in reader:
        file_name = row["file_name"]
        json_path = os.path.join(json_folder, f"{file_name}.json")

        ensure_json_file(json_path, row)
        #if 'nothing' in file_name :
        if target_word in file_name:
            if has_existing_prompts(json_path):
                print(f"[SKIP] already filled: {json_path}")
                continue

            run_generation(row, json_path)


print("\nDone.")