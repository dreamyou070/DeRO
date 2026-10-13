import os
import re
from pathlib import Path
import pandas as pd
from tifascore import VQAModel

# =========================
# Config
# =========================
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

def _to_id6(x):
    """
    Robust: supports 1, 1.0, "000001", "000001.png", "img_000001", etc.
    """
    if pd.isna(x):
        return None
    s = str(x).strip()
    s = Path(s).stem            # drop extension if any
    if s.endswith(".0"):        # handle "1.0"
        s = s[:-2]
    m = re.search(r"(\d+)", s)  # first digit chunk
    if not m:
        return None
    return f"{int(m.group(1)):06d}"

def _norm_answer(a):
    a = "" if a is None else str(a)
    a = a.strip().lower()
    a = re.sub(r"\s+", " ", a)
    a = re.sub(r"[^\w\s]", "", a)  # remove punctuation
    return a

def _is_no(answer: str) -> bool:
    """
    Negation VQA success criterion.
    Works when questions are of the form:
      - "Is there a ... ?" / "Do you see ... ?" / "Is ... present?"
    In that case, more 'no' answers => better (target absent).
    """
    a = _norm_answer(answer)
    if a in {"no", "none", "nothing", "nope", "nah"}:
        return True
    if re.search(r"\bno\b", a) or re.search(r"\bnone\b", a) or re.search(r"\bnothing\b", a):
        return True
    return False

def _iter_images(folder: Path):
    # recursive + stable order
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p

def _pick_col(df: pd.DataFrame, candidates):
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None

def main():
    vqa = VQAModel("mplug-large")

    excel_file = Path("/ssd/sy/NegatingDiffusion/Result/2_negation_vangogh.xlsx")
    result_folder = Path("/ssd/sy/NegatingDiffusion/Result/2_negation_art")
    out_dir = Path("/ssd/sy/NegatingDiffusion/Result/2_negation_art_tifascore")
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # 1) load excel: id -> question, object(target)
    # -------------------------------------------------
    df = pd.read_excel(excel_file)

    id_col = _pick_col(df, ["id", "image_id", "img_id", "index", "idx", "file", "file_name", "filename"])
    q_col  = _pick_col(df, ["question", "quesion"])
    obj_col = _pick_col(df, ["object", "target", "concept"])

    if id_col is None or q_col is None or obj_col is None:
        raise ValueError(
            "Excel must contain columns for id, question, and object/target.\n"
            f"Found: {list(df.columns)}\n"
            f"Detected: id_col={id_col}, q_col={q_col}, obj_col={obj_col}"
        )

    df["id6"] = df[id_col].apply(_to_id6)
    df = df.dropna(subset=["id6"])

    lut_q = dict(zip(df["id6"], df[q_col].astype(str)))
    lut_t = dict(zip(df["id6"], df[obj_col].astype(str)))

    # -------------------------------------------------
    # 2) iterate folders:
    #   result_folder/<model>/<artist>/*.png
    # -------------------------------------------------
    rows = []
    model_dirs = sorted([p for p in result_folder.iterdir() if p.is_dir()])

    for model_dir in model_dirs:
        model_name = model_dir.name

        artist_dirs = sorted([p for p in model_dir.iterdir() if p.is_dir()])
        if not artist_dirs:
            # if someone accidentally has images directly under model_dir, still handle them
            artist_dirs = [model_dir]

        for artist_dir in artist_dirs:
            artist_name = artist_dir.name if artist_dir != model_dir else "NA"

            for img_path in _iter_images(artist_dir):
                stem = img_path.stem
                if not re.fullmatch(r"\d+", stem):
                    # only accept numeric stems to match excel ids (000001.png etc.)
                    continue
                id6 = f"{int(stem):06d}"

                question = lut_q.get(id6)
                if question is None:
                    continue
                target = lut_t.get(id6, "")

                try:
                    answer = vqa.vqa(str(img_path), question)
                    err = ""
                except Exception as e:
                    answer = ""
                    err = str(e)

                rows.append({
                    "model_name": model_name,
                    "artist": artist_name,
                    "id6": id6,
                    "img_path": str(img_path),
                    "question": str(question),
                    "target": str(target),
                    "answer": str(answer),
                    "is_no": int(_is_no(answer)),
                    "error": err,
                })

    # -------------------------------------------------
    # 3) aggregate
    #   - per (model, artist)
    #   - per (model, artist, target)
    # -------------------------------------------------
    res = pd.DataFrame(rows)
    if res.empty:
        print("No results. Check: folder structure, filenames (000001.png), and excel ids.")
        return

    # NOTE:
    # If your question is "Is <negated object/style> present?",
    # then higher score (= more 'no') is better (more absence).
    # If your question is reversed (e.g., "Is the image safe?"),
    # then you'll need to flip the criterion.
    overall_model_artist = (
        res.groupby(["model_name", "artist"], as_index=False)
           .agg(
               n=("is_no", "count"),
               no_count=("is_no", "sum"),
               score=("is_no", "mean"),
               n_error=("error", lambda s: int((s.astype(str) != "").sum())),
           )
           .sort_values(["model_name", "artist", "score", "n"], ascending=[True, True, False, False])
    )

    by_target = (
        res.groupby(["model_name", "artist", "target"], as_index=False)
           .agg(
               n=("is_no", "count"),
               no_count=("is_no", "sum"),
               score=("is_no", "mean"),
               n_error=("error", lambda s: int((s.astype(str) != "").sum())),
           )
           .sort_values(["model_name", "artist", "score", "n"], ascending=[True, True, False, False])
    )

    # also keep a pure per-model summary (collapsed over artist)
    overall_model = (
        res.groupby(["model_name"], as_index=False)
           .agg(
               n=("is_no", "count"),
               no_count=("is_no", "sum"),
               score=("is_no", "mean"),
               n_error=("error", lambda s: int((s.astype(str) != "").sum())),
           )
           .sort_values(["score", "n"], ascending=[False, False])
    )

    # -------------------------------------------------
    # 4) save
    # -------------------------------------------------
    res.to_csv(out_dir / "vqa_per_image.csv", index=False)
    overall_model.to_csv(out_dir / "vqa_overall_per_model.csv", index=False)
    overall_model_artist.to_csv(out_dir / "vqa_overall_per_model_artist.csv", index=False)
    by_target.to_csv(out_dir / "vqa_per_model_artist_by_target.csv", index=False)

    with pd.ExcelWriter(out_dir / "vqa_summary.xlsx") as writer:
        overall_model.to_excel(writer, sheet_name="overall_per_model", index=False)
        overall_model_artist.to_excel(writer, sheet_name="overall_per_model_artist", index=False)
        by_target.to_excel(writer, sheet_name="per_model_artist_by_target", index=False)
        res.to_excel(writer, sheet_name="per_image", index=False)

    print("Saved to:", str(out_dir))
    print("Interpretation: if the VQA question is asking presence of the negated concept, higher score (=more NO) is better.")

if __name__ == "__main__":
    main()
