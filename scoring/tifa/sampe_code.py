from tifascore import tifa_score_benchmark
import json

def main() :

    results = tifa_score_benchmark("mplug-large",
                                   "human_annotations/sample/sample_question_answers.json",
                                   "human_annotations/sample/sample_imgs.json")
    with open("human_annotations/sample/sample_evaluation_result.json", "w") as f:
        json.dump(results, f, indent=4)