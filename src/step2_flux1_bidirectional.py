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

    dev_0 = "cuda:0"
    DTYPE = torch.bfloat16
    MODEL_ID = "black-forest-labs/FLUX.1-dev"
    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE)
    pipe.to(dev_0)
    pipe = init_pipeline(pipe)
    print("step 2. Inference Preparing")
    if args.task == 'object':
        json_folders = '/lustre/dreamyou070/DeRO/src/english/data/DeRO_Data/object3'
        save_folder = '/lustre/dreamyou070/DeRO/src/english/Ablation_Study_1/FLUX_NewCode'
        files = os.listdir(json_folders)
        for file in files:

            json_file = os.path.join(json_folders, file)
            with open(json_file, "r", encoding="utf-8") as f:
                contents = json.load(f)
            if type(contents) == list:
                contents = contents[0]
            prompt = contents["prompt"]
            base_entity_biased_prompts = contents['base_entity_biased_prompts']
            base_entity_bias_absent_prompts = contents['base_entity_bias_absent_prompts']
            for num in range(1,21):
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
                    a = os.path.join(os.path.join(save_folder, save_prompt),
                                     f'{save_prompt}_num_{num}_extraction_{k}.png')
                    if not os.path.exists(a):
                        images = pipe([prompt, prompt],
                                      base_entity_biased_prompts=bias_prompts,
                                      base_entity_bias_absent_prompts=bias_absent_prompts,
                                      style_ratio=0.2,
                                      num_inference_steps=15,
                                      normalize= False,
                                      extraction_k=k,
                                      generator=torch.Generator(device="cuda").manual_seed(42)).images
                        base_image = images[0]
                        ours_image = images[-1]
                        ours_image.save(a)
                        base_image.save(os.path.join(os.path.join(save_folder, save_prompt), f'{save_prompt}_base.png'))

                # print(f"Error processing file: {file}")
    else :
        csv_file = "/lustre/dreamyou070/DeRO/src/english/data/DeRO_Data/concept/art_prompts.csv"
        save_folder = "/lustre/dreamyou070/DeRO/src/english/DERO_Result/FLUX_Dev_Concept"
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            prompt = row["prompt"]
            base_entity_biased_prompt = row["artist_concept"]
            base_entity_bias_absent_prompt = row["auxiliary_semantic"]
            save_folder_name = row["art_index"]
            cur_save_folder = os.path.join(save_folder, save_folder_name)
            os.makedirs(cur_save_folder, exist_ok=True)
            base_save_path = os.path.join(cur_save_folder, f"{save_folder_name}_base.png")
            for num in range(1,5):
                for normalize in [True, False]:
                    for k in [1]:
                        ours_save_path = os.path.join(cur_save_folder,f"{save_folder_name}_ax_num_{num}_k_{k}_clean_code_True_normalize_{normalize}.png")
                        if os.path.exists(ours_save_path):
                            print(f"[SKIP] {ours_save_path}")
                            continue
                        print(f' save on {ours_save_path}')
                        base_entity_biased_prompts = [base_entity_biased_prompt for _ in range(num)]
                        base_entity_bias_absent_prompts = [base_entity_bias_absent_prompt for _ in range(num)]
                        images = pipe([prompt, prompt],
                                      base_entity_biased_prompts=base_entity_biased_prompts,
                                      base_entity_bias_absent_prompts=base_entity_bias_absent_prompts,
                                      num_inference_steps=15,
                                      style_ratio=0.2,
                                      normalize=normalize,
                                      extraction_k=k,
                                      clean_code=True,
                                      generator=torch.Generator(device="cuda").manual_seed(42),
                                      guidance_scale=3.5).images
                        base_image = images[0]
                        ours_image = images[-1]
                        ours_image.save(ours_save_path)

                        if not os.path.exists(base_save_path):
                            base_image.save(base_save_path)

                        print(f"[SAVE] {ours_save_path}")





    json_folder = '/lustre/dreamyou070/DeRO/src/english/data/DeRO_Data/object'
    files_num = len(os.listdir(json_folder))
    new_json_file = os.path.join(json_folder, f"object_{str(files_num)}.json")
    """
    prompt = "tree without leaves"
    base_entity_biased_prompts = [
        "a large tree covered with lush green leaves",
        "a tree with dense green foliage under sunlight",
        "a tall tree filled with vibrant green leaves",
        "a leafy summer tree in a bright forest",
        "a huge oak tree with abundant green leaves",
        "a healthy tree with thick leafy branches",
        "a green tree with rich foliage in a park",
        "a flourishing tree with countless leaves",
        "a tree with bright green leaves swaying in the wind",
        "a forest tree covered in dense foliage",
        "a beautiful tree with fresh green leaves",
        "a giant tree with leafy branches spreading wide",
        "a summer tree with lush green foliage",
        "a tree full of vibrant leaves in daylight",
        "a leafy maple tree in a sunny garden",
        "a green tree with thick natural foliage",
        "a tree with rich leafy textures",
        "a large leafy tree casting shade",
        "a vibrant tree with abundant greenery",
        "a tall tree densely covered with leaves",
        "a flourishing woodland tree with green leaves",
        "a tree with healthy green foliage",
        "a bright green leafy tree in nature",
        "a lush tree with dense summer leaves",
        "a green tree with thick branches full of leaves",
        "a beautiful forest tree with vibrant foliage",
        "a richly leafed tree under a blue sky",
        "a mature tree covered in green leaves",
        "a summer oak tree with dense foliage",
        "a natural tree with lush leafy branches",
        "a green leafy tree illuminated by sunlight",
        "a large park tree with rich foliage",
        "a healthy woodland tree with many leaves",
        "a tree with abundant leafy details",
        "a bright leafy tree in a natural landscape",
        "a dense green tree in a summer forest",
        "a vibrant tree full of fresh leaves",
        "a flourishing green tree with thick foliage",
        "a richly textured leafy tree",
        "a giant leafy tree surrounded by nature"
    ]
    base_entity_bias_absent_prompts = [
        "a tall bare tree standing in a snowy field during winter sunset",
        "a leafless tree with thin branches against a cold winter sky",
        "a winter landscape with a bare tree and snow-covered ground",
        "a dry tree without foliage in a quiet snowy countryside",
        "a tall empty tree silhouette during a winter evening",
        "a lifeless tree with exposed branches in a frozen field",
        "a winter tree with no leaves under soft sunset light",
        "a barren tree standing alone in a snowy landscape",
        "a cold winter scene with a completely leafless tree",
        "a skeletal tree with thin branches in a snowy environment",
        "a bare winter tree illuminated by warm sunset light",
        "a quiet countryside with a dry tree during winter",
        "a frozen landscape featuring a tall tree without leaves",
        "a winter sunset behind a leafless tree in an open field",
        "a tall dry tree with empty branches under a blue sky",
        "a leafless woodland tree during a snowy winter morning",
        "a winter field with barren trees and soft sunlight",
        "a snow-covered rural landscape with bare trees",
        "a dry tree with intricate branches in winter nature",
        "a lonely leafless tree standing in snow",
        "a winter countryside scene with tall barren trees",
        "a frozen field with a tree that has no foliage",
        "a tall winter tree with exposed dark branches",
        "a quiet snowy area with dry leafless trees",
        "a winter landscape with thin bare branches against the sky",
        "a tree without leaves during a cold snowy sunset",
        "a dry winter tree beside a snowy rural house",
        "a silent snowy field with tall barren trees",
        "a winter nature scene with completely bare branches",
        "a leafless tree casting shadows on snow at sunset",
        "a cold outdoor winter landscape with dry trees",
        "a bare deciduous tree in a snow-covered countryside",
        "a winter evening with tall empty trees and snow",
        "a frozen rural field with a large leafless tree",
        "a detailed winter tree with exposed branches",
        "a snowy landscape with dry trees under warm sunlight",
        "a barren tree silhouette during golden hour in winter",
        "a tall winter tree without any green foliage",
        "a quiet frozen field with dark bare branches",
        "a winter sunset scene with skeletal trees and snow"
    ]
    """

    prompt = "calendar without text"

    base_entity_biased_prompts = [
        "a wall calendar filled with printed text and numbers",
        "a monthly calendar with many dates and written labels",
        "a desk calendar covered with text, numbers, and notes",
        "a calendar page showing printed months, weekdays, and dates",
        "a paper calendar with clear black text and numbered days",
        "a planner calendar filled with handwritten notes and text",
        "a calendar sheet containing many letters, numbers, and schedules",
        "a monthly schedule calendar with dense written information",
        "a calendar page full of printed date numbers and weekday names",
        "a white calendar covered with text labels and small numbers",
        "a calendar with visible month title, weekdays, and date numbers",
        "a large wall calendar containing many printed words and numbers",
        "a desk planner with calendar dates and handwritten annotations",
        "a calendar grid filled with numbers, labels, and event text",
        "a printed calendar page with readable text and organized dates",
        "a business calendar filled with schedules, words, and numbers",
        "a calendar notebook showing written plans and date markings",
        "a monthly planner page full of text and date numbers",
        "a calendar with bold printed letters and numbered boxes",
        "a paper calendar displaying weekdays, dates, and notes",
        "a schedule calendar covered in handwritten text and symbols",
        "a wall calendar with visible typography and date information",
        "a calendar page containing dense text and calendar numbers",
        "a planner calendar with many written reminders and date labels",
        "a printed monthly calendar filled with readable date text",
        "a calendar grid with clear numbers and written event names",
        "a desk calendar showing text-heavy schedule information",
        "a calendar page full of letters, numbers, and annotations",
        "a wall calendar with printed words, numbers, and marked dates",
        "a monthly calendar covered with readable text and numeric dates"
    ]
    base_entity_bias_absent_prompts = [
        "a clean blank sheet of white paper on a wooden desk",
        "a plain notebook with completely empty pages",
        "a stack of blank paper sheets with no markings",
        "an open notebook showing clean empty paper pages",
        "a minimalist white paper page under soft lighting",
        "a blank notebook lying on a simple desk",
        "a clean sheet of paper with smooth texture",
        "a plain paper notebook with no printed content",
        "an empty notebook page photographed from above",
        "a white paper sheet with no visible symbols or markings",
        "a simple notebook with blank pages and soft shadows",
        "a completely empty paper page on a flat surface",
        "a clean open notebook with untouched white pages",
        "a minimal blank notebook on a wooden table",
        "a stack of smooth white paper under natural light",
        "a plain empty notebook with clean page texture",
        "a blank sheet of paper photographed in daylight",
        "a minimal notebook with no writing or printed content",
        "a white notebook page with completely empty space",
        "a clean paper sheet with subtle paper texture",
        "a simple blank notebook resting on a desk",
        "a white paper notebook with empty pages",
        "an open blank notebook with smooth white paper",
        "a plain paper page with no visible content",
        "a clean empty notebook photographed from top view",
        "a white paper sheet with soft realistic shadows",
        "a blank notebook with simple minimalist composition",
        "a completely empty sheet of notebook paper",
        "a clean white paper stack in soft studio lighting",
        "a minimalist notebook showing blank paper pages"
    ]

    save_dict = {
        "prompt": prompt,
        "base_entity_biased_prompts": base_entity_biased_prompts,
        "base_entity_bias_absent_prompts": base_entity_bias_absent_prompts}
    with open(new_json_file, "w", encoding="utf-8") as f:
        json.dump(save_dict, f, indent=4, ensure_ascii=False)
    pipe = init_pipeline(pipe)
    save_prompt = prompt.replace(" ", "_")
    save_path = f"./result/algorithm_check_{save_prompt}"
    os.makedirs(save_path, exist_ok=True)

    for num in range(1,40):
        for k in [1,2,3]:
            for clean_code in [True, False]:
                style_ratio = 0.2
                a = os.path.join(save_path, f'aux_{num}_extraction_k_{k}_clean_code_{clean_code}.png')
                if not os.path.exists(a):
                    svd_image = pipe([prompt, prompt],
                                     base_entity_biased_prompts=base_entity_biased_prompts[:num],
                                     base_entity_bias_absent_prompts=base_entity_bias_absent_prompts[:num],
                                     num_inference_steps=15,
                                     style_ratio=style_ratio,
                                     extraction_k=k,
                                     clean_code=clean_code,
                                     generator=torch.Generator(device="cuda").manual_seed(42),
                                     guidance_scale=3.5).images
                    svd_image[0].save(os.path.join(save_path, f'base.png'))
                    svd_image[-1].save(a)

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