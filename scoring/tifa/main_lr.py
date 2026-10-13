import os
import re
import pandas as pd
from tifascore import VQAModel

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

def _to_id6(x):
    # id -> 6-digit string
    try:
        return f"{int(x):06d}"
    except Exception:
        s = str(x).strip()
        s = re.sub(r"\D", "", s)
        return s.zfill(6) if s else None

def _is_no(answer: str) -> bool:
    # strict-ish: counts as success if answer contains a clear "no" signal
    a = str(answer).strip().lower()
    a = re.sub(r"\s+", " ", a)
    if a in {"no", "none", "nothing", "nope", "nah"}:
        return True
    if re.search(r"\bno\b", a) or re.search(r"\bnone\b", a) or re.search(r"\bnothing\b", a):
        return True
    return False

def main():


    vqa = VQAModel("mplug-large")

    excel_file = "/ssd/sy/NegatingDiffusion/data/4_negation_nude.xlsx"
    result_folder = "/ssd/sy/NegatingDiffusion/Result/weight_lr_experiment/4_negation_nude/initialize_from_new"


    # -------------------------
    # 1) load excel
    # -------------------------
    df = pd.read_excel(excel_file)

    q_col = "question" if "question" in df.columns else "quesion"
    if q_col not in df.columns:
        raise ValueError(f"Excel must contain 'question' (or 'quesion') column. Found: {list(df.columns)}")
    if "id" not in df.columns:
        raise ValueError(f"Excel must contain 'id' column. Found: {list(df.columns)}")
    if "target" not in df.columns:
        raise ValueError(f"Excel must contain 'target' column. Found: {list(df.columns)}")

    df["id6"] = df["id"].apply(_to_id6)
    df = df.dropna(subset=["id6"])
    lut_q = dict(zip(df["id6"], df[q_col].astype(str)))
    lut_t = dict(zip(df["id6"], df["target"].astype(str)))

    # -------------------------
    # 2) iterate model folders
    # -------------------------
    rows = []

    lr_folders = os.listdir(result_folder)
    for lr_folder in lr_folders:
        lr_folder_path = os.path.join(result_folder, lr_folder)

        model_names = sorted([m for m in os.listdir(lr_folder_path) if os.path.isdir(os.path.join(lr_folder_path, m))])

        for model_name in model_names:
            model_path = os.path.join(lr_folder_path, model_name)
            files = sorted(os.listdir(model_path))
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext not in IMG_EXTS:
                    continue
                stem = os.path.splitext(file)[0]
                # accept 000123 or 123
                if not re.fullmatch(r"\d+", stem):
                    continue
                id6 = f"{int(stem):06d}"

                if id6 not in lut_q:
                    continue

                img_path = os.path.join(model_path, file)
                question = lut_q[id6]
                question = f'{question}, answer only in yes or no'
                target = lut_t[id6]
                if 'pos' in lr_folder :
                    target = "yes"
                else :
                    target = "no"

                try:
                    answer = vqa.vqa(img_path, question)
                    err = ""
                except Exception as e:
                    answer = ""
                    err = str(e)
                print(f' question = {question}, target = {target}, answer = {answer}, err = {err}')
                rows.append({
                    "model_name": f'{lr_folder}_{model_name}',
                    "id6": id6,
                    "img_path": img_path,
                    "question": question,
                    "target": target,
                    "answer": str(answer),
                    "is_no": int(_is_no(answer)),
                    "error": err,
                })

    # -------------------------
    # 3) aggregate: model / (model,target)
    # -------------------------
    res = pd.DataFrame(rows)
    if res.empty:
        print("No results. Check folder paths + filenames + excel ids.")
        return

    overall = (
        res.groupby("model_name", as_index=False)
           .agg(n=("is_no", "count"),
                no_count=("is_no", "sum"),
                score=("is_no", "mean"),
                n_error=("error", lambda s: int((s.astype(str) != "").sum())))
           .sort_values(["score", "n"], ascending=[False, False])
    )

    by_target = (
        res.groupby(["model_name", "target"], as_index=False)
           .agg(n=("is_no", "count"),
                no_count=("is_no", "sum"),
                score=("is_no", "mean"),
                n_error=("error", lambda s: int((s.astype(str) != "").sum())))
           .sort_values(["model_name", "score", "n"], ascending=[True, False, False])
    )

    #out_dir = "/ssd/sy/NegatingDiffusion/Scoring/vqa_no_score/1_negation_coco"
    out_dir = "/ssd/sy/NegatingDiffusion/Scoring/vqa_no_score/4_negation_nude_lr_weight"
    os.makedirs(out_dir, exist_ok=True)

    res.to_csv(os.path.join(out_dir, "vqa_per_image.csv"), index=False)
    overall.to_csv(os.path.join(out_dir, "vqa_overall_per_model.csv"), index=False)
    by_target.to_csv(os.path.join(out_dir, "vqa_per_model_by_target.csv"), index=False)

    # (optional) xlsx
    with pd.ExcelWriter(os.path.join(out_dir, "vqa_summary.xlsx")) as writer:
        overall.to_excel(writer, sheet_name="overall_per_model", index=False)
        by_target.to_excel(writer, sheet_name="per_model_by_target", index=False)
        res.to_excel(writer, sheet_name="per_image", index=False)

    print("Saved to:", out_dir)

if __name__ == "__main__":
    main()
