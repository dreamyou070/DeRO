from tifascore import tifa_score_benchmark
import json
import os

jobs = [
    {
        "name": "sdxl_object_base",
        "qa": "sample/sdxl_object_base_question_answers.json",
        "imgs": "sample/sdxl_object_base_imgs.json",
    },
    {
        "name": "sdxl_object_ours",
        "qa": "sample/sdxl_object_ours_question_answers.json",
        "imgs": "sample/sdxl_object_ours_imgs.json",
    },
    {
        "name": "sdxl_concept_base",
        "qa": "sample/sdxl_concept_base_question_answers.json",
        "imgs": "sample/sdxl_concept_base_imgs.json",
    },
    {
        "name": "sdxl_concept_ours",
        "qa": "sample/sdxl_concept_ours_question_answers.json",
        "imgs": "sample/sdxl_concept_ours_imgs.json",
    },
]

os.makedirs("sample/tifa_results", exist_ok=True)

for job in jobs:
    print("=" * 80)
    print(f"Running TIFA: {job['name']}")

    results = tifa_score_benchmark(
        "mplug-large",
        job["qa"],
        job["imgs"],
    )

    out_path = f"sample/tifa_results/{job['name']}_evaluation_result.json"

    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved: {out_path}")