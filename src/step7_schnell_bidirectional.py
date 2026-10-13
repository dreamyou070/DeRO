import os,json
from diffusers import FluxPipeline
from flux1_bidirectional import init_pipeline
from torch import nn
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import pandas as pd
from PIL import Image


def main(args):

    dev_0 = "cuda:0"
    DTYPE = torch.bfloat16
    MODEL_ID = "black-forest-labs/FLUX.1-schnell"
    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE)
    pipe.to(dev_0)
    pipe = init_pipeline(pipe)
    print("step 2. Inference Preparing")
    if args.task == 'object' :

        json_folders = '/lustre/dreamyou070/DeRO/src/english/data/DeRO_Data/object'
        files = os.listdir(json_folders)
        for file in files:

            try :
                json_file = os.path.join(json_folders, file)
                with open(json_file, "r", encoding="utf-8") as f:
                    contents = json.load(f)
                if type(contents) == list:
                    contents = contents[0]
                prompt = contents["prompt"]
                base_entity_biased_prompts = contents['base_entity_biased_prompts']
                base_entity_bias_absent_prompts = contents['base_entity_bias_absent_prompts']
                save_folder = '/lustre/dreamyou070/DeRO/src/english/DERO_Result/FLUX_Schnell'
                os.makedirs(save_folder, exist_ok=True)
                save_prompt = prompt.replace(" ", "_")
                save_path = os.path.join(save_folder,save_prompt)
                os.makedirs(save_path, exist_ok=True)
                for num in range(1,5):
                    for style_ratio in [0.2]:
                        if len(base_entity_biased_prompts) > num and len(base_entity_bias_absent_prompts) > num:
                            a = os.path.join(save_path, f'{save_prompt}_aux_{num}_style_ratio_{style_ratio}.png')
                            if not os.path.exists(a):
                                svd_image = pipe([prompt, prompt],
                                                 base_entity_biased_prompts=base_entity_biased_prompts[:num],
                                                 base_entity_bias_absent_prompts=base_entity_bias_absent_prompts[:num],
                                                 num_inference_steps=15,
                                                 style_ratio=style_ratio,
                                                 generator=torch.Generator(device="cuda").manual_seed(42),
                                                 guidance_scale=3.5).images
                                svd_image[0].save(os.path.join(save_path, f'{save_prompt}_FLUX_Schnell_base.png'))
                                svd_image[-1].save(a)
            except :
                print(f"error: {file}")
    else:

        csv_file = "/lustre/dreamyou070/DeRO/src/english/data/DeRO_Data/concept/art_prompts.csv"
        save_folder = "/lustre/dreamyou070/DeRO/src/english/DERO_Result/FLUX_Schnell_Concept"
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
                    ours_save_path = os.path.join(
                        cur_save_folder,
                        f"{save_folder_name}_num_{num}_clean_code_True_normalize_{normalize}.png")
                    if os.path.exists(ours_save_path):
                        print(f"[SKIP] {ours_save_path}")
                        continue
                    base_entity_biased_prompts = [
                        base_entity_biased_prompt for _ in range(num)
                    ]

                    base_entity_bias_absent_prompts = [
                        base_entity_bias_absent_prompt for _ in range(num)
                    ]

                    images = pipe([prompt, prompt],
                                     base_entity_biased_prompts=base_entity_biased_prompts[:num],
                                     base_entity_bias_absent_prompts=base_entity_bias_absent_prompts[:num],
                                     num_inference_steps=15,

                                     generator=torch.Generator(device="cuda").manual_seed(42),
                                     guidance_scale=3.5).images

                    base_image = images[0]
                    ours_image = images[-1]

                    ours_image.save(ours_save_path)

                    if not os.path.exists(base_save_path):
                        base_image.save(base_save_path)

                    print(f"[SAVE] {ours_save_path}")

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