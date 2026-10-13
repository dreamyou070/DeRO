import os, torch, json
from sd15_bidirectional import StableDiffusionPipeline
import pandas as pd

def main(args):

    model_id = 'CompVis/stable-diffusion-v1-4'
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    pipe.unet = pipe.unet.float()

    save_folder = "/lustre/dreamyou070/DeRO/src/english/DERO_Result/SD14_Concept_nude"
    semantic_prompts = ["ultra realistic DSLR street fashion photography of an elegant woman walking on a sunny European city street, wearing a light blue button-up blouse and white wide-leg pants, fashionable casual outfit, stylish sunglasses, confident walking pose, luxury urban atmosphere, soft natural sunlight casting detailed shadows on the pavement, photorealistic, highly detailed clothing fabric and skin texture, cinematic composition, shallow depth of field, realistic city background, yellow taxi parked on the street, premium lifestyle photography, clean summer aesthetic, ultra detailed, high resolution, 85mm lens, f/1.8, magazine-quality fashion editorial, realistic lighting, premium DSLR camera quality",
                        "ultra realistic DSLR fashion photography of a handsome athletic man standing on a beach wearing a black and dark gray long-sleeve rash guard and black compression leggings, stylish sporty beachwear, fit muscular body, modern swimwear fashion, sunglasses, cinematic beach atmosphere, soft golden sunset lighting, highly detailed fabric texture, realistic shadows and skin texture, photorealistic, professional sportswear advertisement, shallow depth of field, sharp focus, luxury fitness brand style, ultra detailed, high resolution, premium DSLR camera quality, clean beach background, fashionable athletic pose, magazine-quality commercial photography"]
    biased_prompts   = ["not wearing cloth",
                        "not wearing cloth"]
    for idx, prompt in enumerate(["beautiful young woman walking on the street, without the nudity concept",
                                  "a man wearing a swimming suit, without the nudity concept"]):
        base_entity_biased_prompt = biased_prompts[idx]
        base_entity_bias_absent_prompt = semantic_prompts[idx]

        cur_save_folder = os.path.join(save_folder, f'nude_{str(idx)}')
        os.makedirs(cur_save_folder, exist_ok=True)

        base_save_path = os.path.join(cur_save_folder, f"nude_{str(idx)}_base.png")

        for num in range(1, 3):
            for normalize in [True, False]:
                ours_save_path = os.path.join(
                    cur_save_folder,
                    f"nude_{str(idx)}_num_{num}_clean_code_True_normalize_{normalize}.png"
                )

                if os.path.exists(ours_save_path):
                    print(f"[SKIP] {ours_save_path}")
                    continue

                base_entity_biased_prompts = [
                    base_entity_biased_prompt for _ in range(num)
                ]

                base_entity_bias_absent_prompts = [
                    base_entity_bias_absent_prompt for _ in range(num)
                ]

                images = pipe(
                    [prompt, prompt],
                    base_entity_biased_prompts=base_entity_biased_prompts,
                    clean_code=True,
                    normalize=normalize,
                    base_entity_bias_absent_prompts=base_entity_bias_absent_prompts,
                ).images

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