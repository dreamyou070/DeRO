import os, torch, json
from sd15_bidirectional import StableDiffusionPipeline
import pandas as pd
import time

def main(args):

    model_id = 'CompVis/stable-diffusion-v1-4'
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)
    pipe = pipe.to("cuda")
    pipe.unet = pipe.unet.float()

    csv_file = "/lustre/dreamyou070/DeRO/src/english/data/DeRO_Data/concept/art_prompts.csv"
    save_folder = "/lustre/dreamyou070/DeRO/src/english/DERO_Result/SD14_Concept"

    df = pd.read_csv(csv_file)

    for _, row in df.iterrows():
        prompt = row["prompt"]
        base_entity_biased_prompt = row["artist_concept"]
        base_entity_bias_absent_prompt = row["auxiliary_semantic"]
        save_folder_name = row["art_index"]

        cur_save_folder = os.path.join(save_folder, save_folder_name)
        os.makedirs(cur_save_folder, exist_ok=True)

        base_save_path = os.path.join(cur_save_folder, f"{save_folder_name}_base.png")

        for num in range(1, 30):
            for normalize in [True, False]:
                ours_save_path = os.path.join(
                    cur_save_folder,
                    f"{save_folder_name}_num_{num}_clean_code_True_normalize_{normalize}.png"
                )

                #if os.path.exists(ours_save_path):
                #    print(f"[SKIP] {ours_save_path}")
                #    continue

                base_entity_biased_prompts = [
                    base_entity_biased_prompt for _ in range(num)
                ]

                base_entity_bias_absent_prompts = [
                    base_entity_bias_absent_prompt for _ in range(num)
                ]
                start_time = time.time()
                images = pipe(
                    [prompt, prompt],
                    base_entity_biased_prompts=base_entity_biased_prompts,
                    clean_code=True,
                    normalize=normalize,
                    base_entity_bias_absent_prompts=base_entity_bias_absent_prompts,
                ).images
                elased_time = time.time() - start_time
                print(f"elased_time: {elased_time}")
                exit()
                base_image = images[0]
                ours_image = images[-1]

                ours_image.save(ours_save_path)

                if not os.path.exists(base_save_path):
                    base_image.save(base_save_path)

                print(f"[SAVE] {ours_save_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='object', help='object or concept')
    args = parser.parse_args()
    main(args)