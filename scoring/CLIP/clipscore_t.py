# clipscore_t.py
# - original CLIPScore style (get_clip_score, normalization, per = w*clip(dot,0,None))
# - multi-model folders under --root_dir
# - prompts from excel (id -> 6-digit) matched with image filename stem
# - saves aggregated json with mean + per_instance per model
# - creates parent dir for --save_json
# - does NOT exit on empty folder (skip)

import os
import json
import glob
import argparse
import pathlib
import warnings

import numpy as np
import pandas as pd
import torch
import clip
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from packaging import version
import sklearn.preprocessing
import tqdm


# -----------------------------
# CLIPScore core (from clipscore.py, trimmed)
# -----------------------------
class CLIPCapDataset(torch.utils.data.Dataset):
    def __init__(self, data, prefix="A photo depicts"):
        self.data = data
        self.prefix = prefix
        if self.prefix and self.prefix[-1] != " ":
            self.prefix += " "

    def __getitem__(self, idx):
        c = self.data[idx]
        tok = clip.tokenize(self.prefix + c, truncate=True).squeeze(0)
        return {"caption": tok}

    def __len__(self):
        return len(self.data)


class CLIPImageDataset(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.preprocess = Compose([
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ])

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p)
        img = self.preprocess(img)
        return {"image": img}

    def __len__(self):
        return len(self.paths)


@torch.no_grad()
def extract_all_captions(captions, model, device, batch_size=256, num_workers=8, prefix="A photo depicts"):
    loader = torch.utils.data.DataLoader(
        CLIPCapDataset(captions, prefix=prefix),
        batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    feats = []
    for b in tqdm.tqdm(loader, desc="text"):
        tok = b["caption"].to(device)
        feats.append(model.encode_text(tok).cpu().numpy())
    return np.vstack(feats)


@torch.no_grad()
def extract_all_images(image_paths, model, device, batch_size=64, num_workers=8):
    loader = torch.utils.data.DataLoader(
        CLIPImageDataset(image_paths),
        batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    feats = []
    for b in tqdm.tqdm(loader, desc="image"):
        img = b["image"].to(device)
        if device == "cuda" or str(device).startswith("cuda"):
            img = img.to(torch.float16)
        feats.append(model.encode_image(img).cpu().numpy())
    return np.vstack(feats)


def get_clip_score(model, images, candidates, device, w=2.5):
    """
    Modified CLIPScore:
      sim = 1 - cosine_similarity
      per = w * clip(sim, 0, inf)
    """
    if isinstance(images, list):
        images = extract_all_images(images, model, device)

    if isinstance(candidates, list):
        candidates = extract_all_captions(candidates, model, device)

    # normalize (same as original)
    if version.parse(np.__version__) < version.parse("1.21"):
        images = sklearn.preprocessing.normalize(images, axis=1)
        candidates = sklearn.preprocessing.normalize(candidates, axis=1)
    else:
        warnings.warn(
            "due to a numerical instability, new numpy normalization is slightly different than paper results. "
            "to exactly replicate paper results, please use numpy version less than 1.21, e.g., 1.20.3."
        )
        images = images / np.sqrt(np.sum(images**2, axis=1, keepdims=True))
        candidates = candidates / np.sqrt(np.sum(candidates**2, axis=1, keepdims=True))

    # cosine similarity
    sim = np.sum(images * candidates, axis=1)

    # 핵심 변경: similarity 뒤집기
    sim = 1.0 - sim

    per = w * np.clip(sim, 0, None)
    return float(np.mean(per)), per, candidates


# -----------------------------
# helper: excel -> {id6: prompt}
# -----------------------------
def load_id2prompt(excel_path, id_col="id", prompt_col="prompt", sheet_name=0):
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    if id_col not in df.columns or prompt_col not in df.columns:
        raise ValueError(f"excel must contain columns: {id_col}, {prompt_col}")

    id2prompt = {}
    for _, row in df.iterrows():
        rid = row[id_col]
        prompt = row[prompt_col]
        if pd.isna(rid) or pd.isna(prompt):
            continue

        # id -> 6-digit string (handle excel floats like 12.0)
        if isinstance(rid, (int, np.integer, float, np.floating)) and not pd.isna(rid):
            rid_str = str(int(rid))
        else:
            rid_str = str(rid).strip()

        if rid_str.endswith(".0"):
            rid_str = rid_str[:-2]

        rid6 = rid_str.zfill(6)
        id2prompt[rid6] = str(prompt)

    return id2prompt


def list_model_dirs(root_dir):
    out = []
    for name in sorted(os.listdir(root_dir)):
        p = os.path.join(root_dir, name)
        if os.path.isdir(p):
            out.append(p)
    return out


def collect_images(model_dir):
    exts = ("png", "jpg", "jpeg", "tiff")
    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(model_dir, f"*.{e}"))
        paths += glob.glob(os.path.join(model_dir, f"*.{e.upper()}"))
    return sorted(paths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", required=True)
    ap.add_argument("--excel", required=True)
    ap.add_argument("--id_col", default="id")
    ap.add_argument("--prompt_col", default="prompt")
    ap.add_argument("--sheet", default=0)
    ap.add_argument("--prefix", default="A photo depicts")
    ap.add_argument("--batch_image", type=int, default=64)
    ap.add_argument("--batch_text", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--save_json", default=None)
    ap.add_argument("--clip_model", default="ViT-B/32")
    ap.add_argument("--w", type=float, default=2.5)
    args = ap.parse_args()

    id2prompt = load_id2prompt(args.excel, args.id_col, args.prompt_col, args.sheet)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        warnings.warn(
            "CLIP runs in full float32 on CPU. Results in paper were computed on GPU (float16)."
        )

    model, _ = clip.load(args.clip_model, device=device, jit=False)
    model.eval()

    results = {}
    models = list_model_dirs(args.root_dir)
    print(f"models: {models}")

    for model_dir in models:
        model_name = os.path.basename(model_dir)
        img_paths = collect_images(model_dir)

        # do NOT exit; just skip
        if len(img_paths) == 0:
            print(f"skip {model_name} (no images)")
            continue

        image_ids = [pathlib.Path(p).stem for p in img_paths]  # expected 6-digit
        captions = []
        kept_img_paths = []
        kept_ids = []
        missing = 0

        for img_id, p in zip(image_ids, img_paths):
            if img_id in id2prompt:
                captions.append(id2prompt[img_id])
                kept_img_paths.append(p)
                kept_ids.append(img_id)
            else:
                missing += 1

        if len(kept_img_paths) == 0:
            print(f"skip {model_name} (no matching ids in excel)")
            continue

        print(f"\nmodel={model_name}  images={len(img_paths)}  matched={len(kept_img_paths)}  missing={missing}")

        # extract image features
        image_feats = extract_all_images(
            kept_img_paths, model, device,
            batch_size=args.batch_image, num_workers=args.num_workers
        )

        # extract text features (with prefix)
        text_feats = extract_all_captions(
            captions, model, device,
            batch_size=args.batch_text, num_workers=args.num_workers,
            prefix=args.prefix
        )

        mean_score, per, _ = get_clip_score(model, image_feats, text_feats, device, w=args.w)

        results[model_name] = {
            "mean_CLIPScore": float(mean_score),
            "per_instance": {img_id: float(sc) for img_id, sc in zip(kept_ids, per)}
        }

        print(f"{model_name} CLIPScore mean = {mean_score:.4f}")

    # summary
    if len(results) > 0:
        print("\n=== summary (higher is better) ===")
        for k, v in sorted(results.items(), key=lambda x: x[1]["mean_CLIPScore"], reverse=True):
            print(f"{k}\t{v['mean_CLIPScore']:.4f}")

    # save json (mkdir parent)
    if args.save_json:
        parent = os.path.dirname(args.save_json)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nsaved: {args.save_json}")


if __name__ == "__main__":
    main()
