# fid_by_model_artist.py
import os
os.environ["TORCH_FIDELITY_NUM_WORKERS"] = "0"  # torch-fidelity dataloader bug workaround

import glob
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import torch_fidelity

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}

def has_images(d: Path) -> bool:
    if not d.is_dir():
        return False
    for p in d.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            return True
    return False

def find_image_dir(root: Path) -> Path | None:
    # 1) images directly under root
    if has_images(root):
        return root

    # 2) common nested folder names
    for sub in ["images", "imgs", "samples", "results", "output", "outputs"]:
        cand = root / sub
        if has_images(cand):
            return cand

    # 3) one-level deep search
    for child in root.iterdir():
        if child.is_dir() and has_images(child):
            return child

    return None

def calc_fid(real_dir: str, fake_dir: str) -> float:
    metrics = torch_fidelity.calculate_metrics(
        input1=real_dir,
        input2=fake_dir,
        fid=True,
        kid=False,
        prc=False,
        cuda=True,
        verbose=False,
    )
    return float(metrics.get("frechet_inception_distance", float("nan")))

def load_existing(out_csv: Path) -> pd.DataFrame | None:
    if out_csv.is_file():
        try:
            return pd.read_csv(out_csv)
        except Exception:
            return None
    return None

def build_done_set(df: pd.DataFrame) -> set[str]:
    done = set()
    if df is None or len(df) == 0:
        return done
    cols = set(df.columns)
    needed = {"model", "artist", "fake_dir"}
    if not needed.issubset(cols):
        return done
    for _, r in df.iterrows():
        fid_ok = ("FID" in cols) and pd.notna(r.get("FID", None))
        err = str(r.get("error", "")).strip()
        if fid_ok and (err == "" or err.lower() == "nan"):
            done.add(f"{r['model']}|{r['artist']}|{r['fake_dir']}")
    return done

def write_summary(detail_df: pd.DataFrame, out_summary_csv: Path):
    if detail_df is None or len(detail_df) == 0:
        return
    if "FID" not in detail_df.columns:
        return

    df_ok = detail_df.dropna(subset=["FID"])
    if len(df_ok) == 0:
        return

    # model-level summary
    model_sum = (
        df_ok.groupby(["model"], as_index=False)
            .agg(FID_mean=("FID", "mean"), FID_std=("FID", "std"), n=("FID", "count"))
            .sort_values(["FID_mean"])
    )
    # artist-level summary (across models)
    artist_sum = (
        df_ok.groupby(["artist"], as_index=False)
            .agg(FID_mean=("FID", "mean"), FID_std=("FID", "std"), n=("FID", "count"))
            .sort_values(["FID_mean"])
    )

    out_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_summary_csv.with_suffix(".xlsx")) as w:
        model_sum.to_excel(w, index=False, sheet_name="by_model")
        artist_sum.to_excel(w, index=False, sheet_name="by_artist")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", required=True, help="root: <root>/<model>/<artist>/...")
    ap.add_argument("--real_dir", required=True, help="real images directory (must directly contain images)")
    ap.add_argument("--out_csv", required=True, help="detail csv")
    ap.add_argument("--out_summary", required=True, help="summary (xlsx will be written)")
    ap.add_argument("--overwrite", action="store_true", default=False)
    ap.add_argument("--save_every", type=int, default=1)
    ap.add_argument("--summary_every", type=int, default=20)
    args = ap.parse_args()

    root_dir = Path(args.root_dir)
    real_dir = Path(args.real_dir)
    out_csv = Path(args.out_csv)
    out_summary = Path(args.out_summary)

    if not root_dir.is_dir():
        raise FileNotFoundError(f"root_dir not found: {root_dir}")
    if not real_dir.is_dir():
        raise FileNotFoundError(f"real_dir not found: {real_dir}")
    if not has_images(real_dir):
        raise RuntimeError(f"real_dir must directly contain images: {real_dir}")

    existing_df = load_existing(out_csv)
    done_set = set()
    if not args.overwrite and existing_df is not None:
        done_set = build_done_set(existing_df)

    rows = []
    if existing_df is not None and len(existing_df) > 0:
        rows = existing_df.to_dict("records")

    model_names = [m for m in sorted(os.listdir(root_dir)) if (root_dir / m).is_dir()]
    total_runs = 0
    for m in model_names:
        mp = root_dir / m
        artist_names = [a for a in sorted(os.listdir(mp)) if (mp / a).is_dir()]
        total_runs += len(artist_names)

    n_run = 0
    n_done = 0

    for model in model_names:
        model_path = root_dir / model
        artist_names = [a for a in sorted(os.listdir(model_path)) if (model_path / a).is_dir()]

        for artist in artist_names:
            artist_path = model_path / artist
            img_dir = find_image_dir(artist_path)
            if img_dir is None:
                continue

            key = f"{model}|{artist}|{str(img_dir)}"
            if (not args.overwrite) and (key in done_set):
                n_done += 1
                print(f"[SKIP] {model}/{artist}")
                continue

            n_run += 1
            print(f"[RUN {n_run}/{total_runs}] model={model} artist={artist}")
            print(f"      fake_dir={img_dir}")

            row = dict(
                model=model,
                artist=artist,
                fake_dir=str(img_dir),
                real_dir=str(real_dir),
                ts=datetime.now().isoformat(timespec="seconds"),
            )

            try:
                fid = calc_fid(real_dir=str(real_dir), fake_dir=str(img_dir))
                row["FID"] = float(fid)
                row["error"] = ""
            except Exception as e:
                row["FID"] = float("nan")
                row["error"] = f"{type(e).__name__}: {e}"

            rows.append(row)

            # iterative save
            if args.save_every > 0 and (n_run % args.save_every == 0):
                df = pd.DataFrame(rows)
                df = df.sort_values(by=["model", "artist", "FID"], ascending=True, na_position="last")
                out_csv.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(out_csv, index=False)
                print(f"[SAVE] detail -> {out_csv}")

            # iterative summary
            if args.summary_every > 0 and (n_run % args.summary_every == 0):
                df = pd.DataFrame(rows)
                write_summary(df, out_summary)
                print(f"[SAVE] summary -> {out_summary.with_suffix('.xlsx')}")

    # final save
    df = pd.DataFrame(rows)
    if "FID" in df.columns:
        df = df.sort_values(by=["model", "artist", "FID"], ascending=True, na_position="last")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[DONE] detail saved: {out_csv}")

    write_summary(df, out_summary)
    print(f"[DONE] summary saved: {out_summary.with_suffix('.xlsx')}")

    if not args.overwrite:
        print(f"[INFO] skipped already-done: {n_done}")

if __name__ == "__main__":
    main()
