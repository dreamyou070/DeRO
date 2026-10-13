import os,json
from diffusers import FluxPipeline
from flux1_bidirectional import init_pipeline
from torch import nn
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import pandas as pd
from PIL import Image
import random

def main(args):

    print(f' step 1. set model')
    dev_0 = "cuda:0"
    DTYPE = torch.bfloat16
    MODEL_ID = "black-forest-labs/FLUX.1-dev"
    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE)
    pipe.to(dev_0)
    pipe = init_pipeline(pipe)

    print("step 2. Inference Preparing")
    save_base_folder = '/lustre/dreamyou070/EMNLP_DeRO/result_rebutal'
    os.makedirs(save_base_folder, exist_ok=True)
    json_folder_base = '/lustre/dreamyou070/EMNLP_DeRO/data'
    json_folder_list = ['object_llm1'] # : #','object_llm2','object_manual']
    for json_folder in json_folder_list:
        save_folder = os.path.join(save_base_folder, json_folder)
        os.makedirs(save_folder, exist_ok=True)
        json_folder_path = os.path.join(json_folder_base, json_folder)
        files = os.listdir(json_folder_path)
        for file in files:
            json_file = os.path.join(json_folder_path, file)
        with open(json_file, "r", encoding="utf-8") as f:
            contents = json.load(f)
        if type(contents) == list:
            contents = contents[0]
        prompt = contents["prompt"]
        base_entity_biased_prompts = contents['base_entity_biased_prompts']
        base_entity_bias_absent_prompts = contents['base_entity_bias_absent_prompts']
        for num in range(1,5):
            for k in [1,2,3]:
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
                a = os.path.join(os.path.join(save_folder, save_prompt),f'{save_prompt}_num_{num}_extraction_{k}.png')
                if not os.path.exists(a):
                    images = pipe([prompt, prompt],base_entity_biased_prompts=bias_prompts,
                                  base_entity_bias_absent_prompts=bias_absent_prompts,style_ratio=0.2,
                                  num_inference_steps=15,normalize= False,extraction_k=k,
                                  generator=torch.Generator(device="cuda").manual_seed(42)).images
                    base_image = images[0]
                    ours_image = images[-1]
                    ours_image.save(a)
                    base_image.save(os.path.join(os.path.join(save_folder, save_prompt), f'{save_prompt}_base.png'))



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--json_dir", type=str, default=0)
    parser.add_argument("--save_path", type=str, default=0)
    parser.add_argument("--base_save", type=str)
    parser.add_argument("--task", type=str, default = "object")
    parser.add_argument("--use_new_code", action='store_true')
    args = parser.parse_args()
    main(args)