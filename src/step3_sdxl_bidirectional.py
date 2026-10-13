import os,json
from sdxl_pipeline_bidirection import StableDiffusionXLPipeline
from torch import nn
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import pandas as pd
from PIL import Image
import argparse
import csv
import random

def main(args):

    pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
    pipe.to("cuda")

    print("step 2. Inference Preparing")
    if args.task == 'object':
        #json_folders = '/lustre/dreamyou070/DeRO/src/english/data/DeRO_Data/object3'
        #save_folder = '/lustre/dreamyou070/DeRO/src/english/Ablation_Study_1/SDXL_20260525'
        json_folders = '/lustre/dreamyou070/DeRO/src/english/data/DeRO_Data/object'
        save_folder = '/lustre/dreamyou070/DeRO/src/english/Ablation_Study_1/SDXL_Bi_0526'
        files = os.listdir(json_folders)
        for file in files:

            json_file = os.path.join(json_folders, file)
            with open(json_file, "r", encoding="utf-8") as f:
                contents = json.load(f)
            if type(contents) == list:
                contents = contents[0]

            prompt = contents["prompt"]
            if 'plant with' in prompt and 'leaves' in prompt:

                base_entity_biased_prompts = contents['base_entity_biased_prompts']
                base_entity_bias_absent_prompts = contents['base_entity_bias_absent_prompts']

                for num in range(1, 20):
                    for k in [1]:
                        save_prompt = prompt.replace(" ", "_")
                        os.makedirs(os.path.join(save_folder, save_prompt), exist_ok=True)
                        if len(base_entity_biased_prompts) < num or len(base_entity_bias_absent_prompts) < num:
                            continue

                        def select_num_prompt(base, num):
                            num = min(num, len(base))  # 리스트 길이 초과 방지
                            selected_prompts = random.sample(base, num)
                            return selected_prompts

                        bias_prompts = select_num_prompt(base_entity_biased_prompts, num)
                        bias_absent_prompts = select_num_prompt(base_entity_bias_absent_prompts, num)
                        normalize = False
                        images = pipe([prompt, prompt],
                                      base_entity_biased_prompts=bias_prompts,
                                      base_entity_bias_absent_prompts=bias_absent_prompts,
                                      style_ratio=0.2,
                                      num_inference_steps=15,
                                      normalize =- normalize,
                                      extraction_k=k,
                                      generator=torch.Generator(device="cuda").manual_seed(42)).images
                        base_image = images[0]
                        ours_image = images[-1]
                        a = os.path.join(os.path.join(save_folder, save_prompt),
                                         f'{save_prompt}_num_{num}_clean_code_True_normalize_{normalize}_extraction_{k}.png')
                        print(f' save at {a}')
                        if not os.path.exists(a):
                            ours_image.save(a)
                            base_image.save(
                                os.path.join(os.path.join(save_folder, save_prompt), f'{save_prompt}_base.png'))



    else :
        csv_file = "/lustre/dreamyou070/DeRO/src/english/data/DeRO_Data/concept/art_prompts.csv"
        save_folder = '/lustre/dreamyou070/DeRO/src/english/Ablation_Study_1/SDXL_Concept_Bi_0526'
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            prompt = row["prompt"]
            if 'impressionist landscape of a water lilies pond, with a focus on light' in prompt :
                base_entity_biased_prompt = row["artist_concept"]
                base_entity_bias_absent_prompt = row["auxiliary_semantic"]
                save_folder_name = row["art_index"]
                cur_save_folder = os.path.join(save_folder, save_folder_name)
                os.makedirs(cur_save_folder, exist_ok=True)
                base_save_path = os.path.join(cur_save_folder, f"{save_folder_name}_base.png")
                for num in range(1, 7):
                    for k in [1] :
                        style_ratio = 0.2
                        normalize = False
                        ours_save_path = os.path.join(
                            cur_save_folder,
                            f"{save_folder_name}_ax_num_{num}_k_{k}_clean_code_normalize_{normalize}.png")


                        base_entity_biased_prompts = [
                            base_entity_biased_prompt for _ in range(num)
                        ]

                        base_entity_bias_absent_prompts = [
                            base_entity_bias_absent_prompt for _ in range(num)
                        ]

                        images = pipe([prompt, prompt],
                                      base_entity_biased_prompts=base_entity_biased_prompts,
                                      base_entity_bias_absent_prompts=base_entity_bias_absent_prompts,
                                      style_ratio=style_ratio,
                                      num_inference_steps=15,
                                      normalize =- normalize,
                                      extraction_k=k,
                                      generator=torch.Generator(device="cuda").manual_seed(42)).images

                        base_image = images[0]
                        ours_image = images[-1]
                        ours_image.save(ours_save_path)
                        if not os.path.exists(base_save_path):
                            base_image.save(base_save_path)
                        print(f"[SAVE] {ours_save_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--json_dir", type=str, default=0)
    parser.add_argument("--save_path", type=str, default=0)
    parser.add_argument("--base_save", type=str)
    parser.add_argument("--task", type=str, default = 'object')
    parser.add_argument("--analize_only_others", action='store_true')
    args = parser.parse_args()
    main(args)