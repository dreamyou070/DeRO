from tifascore import tifa_score_benchmark
import json
import os

jobs = [
    (
        "sdxl_concept_base",
        "./sample/sample_sdxl_concept_question_answering_base.json",
        "./sample/sample_sdxl_concept_base.json",
    ),
    (
        "sdxl_concept_bidirectional",
        "./sample/sample_sdxl_concept_question_answering_bidirectional.json",
        "./sample/sample_sdxl_concept_bidirectional.json",
    ),
    (
        "sdxl_object_base",
        "./sample/sample_sdxl_object_question_answering_base.json",
        "./sample/sample_sdxl_object_base.json",
    ),
    (
        "sdxl_object_bidirectional",
        "./sample/sample_sdxl_object_question_answering_bidirectional.json",
        "./sample/sample_sdxl_object_bidirectional.json",
    ),
]

os.makedirs("./sample/tifa_results", exist_ok=True)

for name, qa_path, img_path in jobs:
    print(f"\n[RUN] {name}")
    print(f"QA : {qa_path}")
    print(f"IMG: {img_path}")

    results = tifa_score_benchmark(
        "mplug-large",
        # "blip-large",
        qa_path,
        img_path,
    )

    save_path = f"./sample/tifa_results/{name}_tifa_result.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"[SAVED] {save_path}")