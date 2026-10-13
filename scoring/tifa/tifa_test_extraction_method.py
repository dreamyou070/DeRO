from tifascore import get_question_and_answers, filter_question_and_answers, UnifiedQAModel, tifa_score_benchmark, \
    tifa_score_single, VQAModel
import json
import openai

if __name__ == "__main__":

    #####################################
    ## Test TIFA score on benchmark
    #####################################

    json_files = [
        '/lustre/dreamyou070/Project/DeRO/src/english/data/tifa_qa_extraction_method_ica_images.json',
        '/lustre/dreamyou070/Project/DeRO/src/english/data/tifa_qa_extraction_method_pca_images.json',
        '/lustre/dreamyou070/Project/DeRO/src/english/data/tifa_qa_extraction_method_rpca_images.json',
        '/lustre/dreamyou070/Project/DeRO/src/english/data/tifa_qa_extraction_method_svd_images.json'
    ]

    result_files = [
        '/lustre/dreamyou070/Project/DeRO/src/english/data/tifa_qa_extraction_method_ica_images_result.json',
        '/lustre/dreamyou070/Project/DeRO/src/english/data/tifa_qa_extraction_method_pca_images_result.json',
        '/lustre/dreamyou070/Project/DeRO/src/english/data/tifa_qa_extraction_method_rpca_images_result.json',
        '/lustre/dreamyou070/Project/DeRO/src/english/data/tifa_qa_extraction_method_svd_images_result.json'
    ]
    for sample_json, result_json in zip(json_files, result_files):
        results = tifa_score_benchmark("mplug-large",
                                       "/lustre/dreamyou070/Project/DeRO/src/english/data/tifa_qa_extraction_component_num_scoring.json",
                                       sample_json)
        with open(result_json, "w") as f:
            json.dump(results, f, indent=4)