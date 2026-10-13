import os
import glob
import pandas as pd

# =========================
# Paths
# =========================
results_dir = "/ssd/sy/NegatingDiffusion/Scoring/results/1_negation_coco"
label_xlsx  = "/ssd/sy/NegatingDiffusion/data/1_negation_coco.xlsx"
out_csv     = os.path.join(results_dir, "classwise_score_summary.csv")

# =========================
# Load label excel
# =========================
df_label = pd.read_excel(label_xlsx)

# 기대 컬럼: id, prompt, target
need_cols = {"prompt", "target"}
missing = need_cols - set(df_label.columns)
if missing:
    raise ValueError(f"Excel missing columns: {missing} | got={list(df_label.columns)}")

# prompt 중복이 있을 수 있어서(혹시라도) target이 동일하다고 가정하고 첫 값 사용
prompt_to_target = (
    df_label.dropna(subset=["prompt", "target"])
            .astype({"prompt": "string", "target": "string"})
            .drop_duplicates(subset=["prompt"], keep="first")
            .set_index("prompt")["target"]
)

# =========================
# Process each model csv
# =========================
rows = []
per_model_tables = {}

csv_paths = sorted(glob.glob(os.path.join(results_dir, "*.csv")))
if not csv_paths:
    raise RuntimeError(f"No CSV found under: {results_dir}")

for csv_path in csv_paths:
    model_name = os.path.splitext(os.path.basename(csv_path))[0]

    df = pd.read_csv(csv_path)

    # 기대 컬럼: text, score (그리고 model)
    if "text" not in df.columns or "score" not in df.columns:
        print(f"[SKIP] {model_name}: missing text/score columns")
        continue

    df["text"] = df["text"].astype("string")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    # target 붙이기: text == prompt
    df["target"] = df["text"].map(prompt_to_target)

    # 매칭 안된 것 체크(원하면 로그)
    n_all = len(df)
    n_matched = df["target"].notna().sum()
    if n_matched < n_all:
        print(f"[WARN] {model_name}: matched {n_matched}/{n_all} (unmatched {n_all - n_matched})")

    df_m = df.dropna(subset=["target", "score"]).copy()

    # class-wise mean
    g = (df_m.groupby("target")["score"]
              .agg(["mean", "count"])
              .reset_index()
              .rename(columns={"mean": "avg_score", "count": "n"}))

    g["model_csv"] = model_name
    per_model_tables[model_name] = g

    rows.append(g)

# =========================
# Save summary (long format)
# =========================
summary_long = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
summary_long = summary_long.sort_values(["model_csv", "target"])
summary_long.to_csv(out_csv, index=False)
print("saved ->", out_csv)

# =========================
# (Optional) Pivot: target x model
# =========================
out_pivot = os.path.join(results_dir, "classwise_score_pivot.csv")
if not summary_long.empty:
    pivot = summary_long.pivot_table(
        index="target",
        columns="model_csv",
        values="avg_score",
        aggfunc="mean"
    ).reset_index()
    pivot.to_csv(out_pivot, index=False)
    print("saved ->", out_pivot)
