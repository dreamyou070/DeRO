import os
import re
import pandas as pd
import torch
import lpips
from PIL import Image
from torchvision import transforms

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

def to_id6(x):
    try:
        return f"{int(x):06d}"
    except Exception:
        s = str(x).strip()
        s = re.sub(r"\D", "", s)
        return s.zfill(6) if s else None

def build_lpips(net="alex", device="cuda"):
    loss_fn = lpips.LPIPS(net=net).to(device).eval()
    tfm = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1, 1]
    ])
    return loss_fn, tfm

@torch.no_grad()
def lpips_distance(loss_fn, tfm, img_a_path, img_b_path, device):
    a = tfm(Image.open(img_a_path).convert("RGB")).unsqueeze(0).to(device)
    b = tfm(Image.open(img_b_path).convert("RGB")).unsqueeze(0).to(device)
    d = loss_fn(a, b)
    return float(d.item())

def main():

    model_name = 'NAG'
    base_name = 'FLUX'

    base_folder = f"/MIR/nas1/dreamyou070/NegatingDiffusion/Result/Compare/2_negation_art/{base_name}"
    reference_folder = os.path.join(base_folder, "base")


    target_folder = os.path.join(base_folder,model_name)

    excel_file = "/MIR/nas1/dreamyou070/NegatingDiffusion/data_final/2_negation_vangogh.xlsx"
    df = pd.read_excel(excel_file)

    if "id" not in df.columns or "object" not in df.columns:
        raise ValueError(f"Excel must contain columns: id, object. Found: {list(df.columns)}")

    df["id6"] = df["id"].apply(to_id6)
    df = df.dropna(subset=["id6"])
    df["object"] = df["object"].astype(str).str.strip()

    # group ids by object
    obj2ids = {}
    for _, r in df.iterrows():
        obj = r["object"]
        obj2ids.setdefault(obj, []).append(r["id6"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn, tfm = build_lpips(net="alex", device=device)

    rows = []
    missing = 0

    for obj, ids in obj2ids.items():
        for id6 in ids:
            ref_path = os.path.join(reference_folder, f"{id6}.png")
            tgt_path = os.path.join(target_folder, f"{id6}.png")

            if not (os.path.exists(ref_path) and os.path.exists(tgt_path)):
                missing += 1
                continue

            d = lpips_distance(loss_fn, tfm, ref_path, tgt_path, device)
            rows.append({
                "object": obj,
                "id6": id6,
                "lpips": d,
                "ref_path": ref_path,
                "tgt_path": tgt_path,
            })

    res = pd.DataFrame(rows)
    if res.empty:
        print("No matched image pairs found. Check folders and filenames.")
        return

    # per-object mean + overall mean
    per_object = res.groupby("object", as_index=False).agg(
        n=("lpips", "count"),
        lpips_mean=("lpips", "mean"),
        lpips_std=("lpips", "std"),
    ).sort_values("lpips_mean", ascending=False)

    overall_mean = float(res["lpips"].mean())
    overall_std = float(res["lpips"].std())

    out_dir = "/MIR/nas1/dreamyou070/NegatingDiffusion/Result/Compare/2_negation_art"
    os.makedirs(out_dir, exist_ok=True)
    per_image_csv = os.path.join(out_dir, f"{model_name}_lpips_per_image.csv")
    per_object_csv = os.path.join(out_dir,  f"{model_name}_lpips_per_object.csv")

    res.to_csv(per_image_csv, index=False)
    per_object.to_csv(per_object_csv, index=False)

    print("LPIPS Summary")
    print(f"- overall_mean: {overall_mean}")
    print(f"- overall_std : {overall_std}")
    print(f"- per-image  : {per_image_csv}")
    print(f"- per-object : {per_object_csv}")
    if missing:
        print(f"- missing pairs skipped: {missing}")

if __name__ == "__main__":
    main()
