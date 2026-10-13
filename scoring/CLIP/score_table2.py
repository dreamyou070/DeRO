import os
import re
from PIL import Image

import torch
import pandas as pd
from transformers import CLIPProcessor, CLIPModel


csv_file = '/lustre/dreamyou070/DeRO/src/english/data/DeRO_Data/concept/art_prompts.csv'

#base_folder = '/lustre/dreamyou070/DeRO/src/english/DERO_Result/SD14_Concept'
#save_dir = "/lustre/dreamyou070/DeRO/scoring/CLIP/clipscore_SD14_concept"

base_folder = '/lustre/dreamyou070/DeRO/src/english/DERO_Result/SD15_Concept'
save_dir = "/lustre/dreamyou070/DeRO/scoring/CLIP/clipscore_SD15_concept"
os.makedirs(save_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model_name = "openai/clip-vit-large-patch14"

clip_model = CLIPModel.from_pretrained(
    clip_model_name
).to(device)

clip_processor = CLIPProcessor.from_pretrained(
    clip_model_name
)

clip_model.eval()

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp")


##################################################
# load prompt csv
##################################################

prompt_df = pd.read_csv(csv_file)

artindex_to_prompt = {}

for _, row in prompt_df.iterrows():

    art_index = row["art_index"]
    prompt = row["prompt"]

    artindex_to_prompt[art_index] = prompt


##################################################
# helper
##################################################

def extract_num(filename):

    if "base" in filename.lower():
        return "base"

    match = re.search(r"_num_(\d+)", filename)

    if match:
        return int(match.group(1))

    return None


def compute_clip_score(
    image_path,
    prompt,
    w=2.5
):

    image = Image.open(
        image_path
    ).convert("RGB")

    inputs = clip_processor(
        text=[prompt],
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    with torch.no_grad():

        outputs = clip_model(
            **inputs
        )

        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

        image_embeds = image_embeds / image_embeds.norm(
            dim=-1,
            keepdim=True
        )

        text_embeds = text_embeds / text_embeds.norm(
            dim=-1,
            keepdim=True
        )

        cosine = (
            image_embeds @ text_embeds.T
        )[0,0]

        clip_score = (
            w * torch.clamp(
                cosine,
                min=0.0
            )
        )

    return (
        clip_score.item(),
        cosine.item()
    )


##################################################
# main loop
##################################################

results=[]

sub_folders=sorted(
    os.listdir(base_folder)
)

for sub_folder in sub_folders:

    sub_folder_path=os.path.join(
        base_folder,
        sub_folder
    )

    if not os.path.isdir(
        sub_folder_path
    ):
        continue

    ##################################################
    # prompt matching
    ##################################################

    if sub_folder not in artindex_to_prompt:

        print(
            f"[SKIP] prompt not found: {sub_folder}"
        )

        continue

    prompt=artindex_to_prompt[
        sub_folder
    ]

    print("="*80)
    print(
        f"Folder={sub_folder}"
    )
    print(
        f"Prompt={prompt}"
    )

    files=sorted(
        os.listdir(
            sub_folder_path
        )
    )

    for file in files:

        if not file.lower().endswith(
            IMG_EXTS
        ):
            continue

        prompt_num=extract_num(
            file
        )

        if prompt_num is None:

            print(
                f"[SKIP] no num: {file}"
            )
            continue

        image_path=os.path.join(
            sub_folder_path,
            file
        )

        try:

            clip_score,\
            cosine_score=compute_clip_score(
                image_path,
                prompt
            )

            results.append({

                "folder":sub_folder,
                "prompt":prompt,
                "prompt_num":prompt_num,
                "file":file,

                "clip_score":clip_score,
                "clip_cosine":cosine_score,

                "image_path":image_path
            })

            print(
                f"[OK] "
                f"{sub_folder} | "
                f"{prompt_num} | "
                f"CLIP={clip_score:.4f}"
            )

        except Exception as e:

            print(
                f"[ERROR] "
                f"{image_path}"
            )

            print(e)


##################################################
# save raw
##################################################

df=pd.DataFrame(
    results
)

raw_path=os.path.join(
    save_dir,
    "clipscore_per_image.csv"
)

df.to_csv(
    raw_path,
    index=False
)


##################################################
# average by num
##################################################

by_num=(
    df.groupby(
        "prompt_num"
    )
    .agg(

        num_images=(
            "clip_score",
            "count"
        ),

        avg_clip_score=(
            "clip_score",
            "mean"
        ),

        std_clip_score=(
            "clip_score",
            "std"
        ),

        avg_cosine=(
            "clip_cosine",
            "mean"
        ),

        std_cosine=(
            "clip_cosine",
            "std"
        )
    )
    .reset_index()
)

by_num=by_num.sort_values(
    "prompt_num",
    key=lambda x:
    x.astype(str)
)

by_num_path=os.path.join(
    save_dir,
    "clipscore_by_aux_num.csv"
)

by_num.to_csv(
    by_num_path,
    index=False
)


##################################################
# base summary only
##################################################

base_df=df[
    df["prompt_num"]=="base"
]

base_summary=pd.DataFrame([{

    "num_images":
    len(base_df),

    "avg_clip_score":
    base_df[
        "clip_score"
    ].mean(),

    "std_clip_score":
    base_df[
        "clip_score"
    ].std(),

    "avg_cosine":
    base_df[
        "clip_cosine"
    ].mean(),

    "std_cosine":
    base_df[
        "clip_cosine"
    ].std(),

}])

base_path=os.path.join(
    save_dir,
    "base_summary.csv"
)

base_summary.to_csv(
    base_path,
    index=False
)


print("="*80)

print(
    "saved:"
)

print(raw_path)
print(by_num_path)
print(base_path)