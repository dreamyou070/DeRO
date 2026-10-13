# run_pos_neg.py
# ------------------------------------------------------------
# Iterative FID runner for:
#   base_folder/{negation_mode,positive_mode}/{lr}/{neg_weight}/(images...)
#
# - Iterates all (mode, lr, neg_weight) leaf folders
# - Computes FID vs real_dir (torch-fidelity)
# - Writes/updates CSV iteratively (so you can monitor progress)
# - Can resume: skips rows already computed (unless --overwrite)
#
# NOTE:
# - FID does NOT use prompts (Excel not needed for FID).
# - torch-fidelity requires real_dir and fake_dir to DIRECTLY contain images.
#
# FIX:
# torch-fidelity DataLoader multiprocessing bug workaround
# (RuntimeError: Trying to resize storage that is not resizable)
# ------------------------------------------------------------
import os
os.environ["TORCH_FIDELITY_NUM_WORKERS"] = "0"

import argparse
from pathlib import Path
import pandas as pd
import torch_fidelity
from datetime import datetime

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


def list_leaf_runs(base_folder: Path, modes: list[str]) -> list[dict]:
    runs = []
    for mode in modes:
        mode_dir = base_folder / mode
        if not mode_dir.is_dir():
            continue
        for lr_dir in sorted([p for p in mode_dir.iterdir() if p.is_dir()]):
            lr = lr_dir.name
            for nw_dir in sorted([p for p in lr_dir.iterdir() if p.is_dir()]):
                neg_weight = nw_dir.name
                img_dir = find_image_dir(nw_dir)
                if img_dir is None:
                    continue
                runs.append(
                    dict(
                        mode=mode,
                        lr=lr,
                        neg_weight=neg_weight,
                        run_dir=str(nw_dir),
                        fake_dir=str(img_dir),
                    )
                )
    return runs


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


def make_key(row: dict) -> str:
    return f"{row['mode']}|{row['lr']}|{row['neg_weight']}|{row['fake_dir']}"


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
    needed = {"mode", "lr", "neg_weight", "fake_dir"}
    if not needed.issubset(cols):
        return done
    # mark as done if FID exists (not NaN) and no error
    for _, r in df.iterrows():
        fid_ok = ("FID" in cols) and pd.notna(r.get("FID", None))
        err = str(r.get("error", "")).strip()
        if fid_ok and (err == "" or err.lower() == "nan"):
            key = f"{r['mode']}|{r['lr']}|{r['neg_weight']}|{r['fake_dir']}"
            done.add(key)
    return done


def write_summary(detail_df: pd.DataFrame, out_summary_csv: Path):
    if detail_df is None or len(detail_df) == 0:
        return
    if "FID" not in detail_df.columns:
        return
    df_ok = detail_df.dropna(subset=["FID"])
    if len(df_ok) == 0:
        return
    summary = (
        df_ok.groupby(["mode", "lr", "neg_weight"], as_index=False)
            .agg(
                FID_mean=("FID", "mean"),
                FID_std=("FID", "std"),
                n=("FID", "count"),
            )
            .sort_values(["mode", "lr", "neg_weight"])
    )
    out_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_summary_csv, index=False)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_folder", type=str, required=True)
    ap.add_argument("--real_dir", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--out_summary_csv", type=str, required=True)

    # force exactly two modes (as you requested)
    ap.add_argument("--negation_mode", type=str, default="negation_mode",
                    help="folder name under base_folder for negation runs")
    ap.add_argument("--positive_mode", type=str, default="positive_mode",
                    help="folder name under base_folder for positive runs")

    # iterative behavior
    ap.add_argument("--overwrite", action="store_true", default=False,
                    help="recompute even if already in out_csv")
    ap.add_argument("--save_every", type=int, default=1,
                    help="write CSV every N runs")
    ap.add_argument("--summary_every", type=int, default=5,
                    help="write summary CSV every N runs")
    return ap.parse_args()


def main():
    args = parse_args()

    base_folder = Path(args.base_folder)
    real_dir = Path(args.real_dir)
    out_csv = Path(args.out_csv)
    out_summary = Path(args.out_summary_csv)

    if not base_folder.is_dir():
        raise FileNotFoundError(f"base_folder not found: {base_folder}")
    if not real_dir.is_dir():
        raise FileNotFoundError(f"real_dir not found: {real_dir}")
    if not has_images(real_dir):
        raise RuntimeError(f"real_dir must directly contain images: {real_dir}")

    modes = [args.negation_mode, args.positive_mode]

    # load existing results to resume
    existing_df = load_existing(out_csv)
    done_set = set()
    if not args.overwrite and existing_df is not None:
        done_set = build_done_set(existing_df)

    runs = list_leaf_runs(base_folder, modes=modes)
    if len(runs) == 0:
        raise RuntimeError(f"No valid runs found under {base_folder} with modes={modes}")

    rows = []
    if existing_df is not None and len(existing_df) > 0:
        # keep existing rows
        rows = existing_df.to_dict("records")

    n_total = len(runs)
    n_done = 0
    n_run = 0

    for idx, r in enumerate(runs, start=1):
        key = make_key(r)
        if (not args.overwrite) and (key in done_set):
            n_done += 1
            print(f"[SKIP {idx}/{n_total}] already done | {r['mode']} {r['lr']} {r['neg_weight']}")
            continue

        print(f"[RUN  {idx}/{n_total}] mode={r['mode']} lr={r['lr']} neg_weight={r['neg_weight']}")
        print(f"      fake_dir={r['fake_dir']}")

        row = dict(
            mode=r["mode"],
            lr=r["lr"],
            neg_weight=r["neg_weight"],
            run_dir=r["run_dir"],
            fake_dir=r["fake_dir"],
            real_dir=str(real_dir),
            ts=datetime.now().isoformat(timespec="seconds"),
        )

        try:
            fid = calc_fid(real_dir=str(real_dir), fake_dir=r["fake_dir"])
            row["FID"] = fid
            row["error"] = ""
        except Exception as e:
            row["FID"] = float("nan")
            row["error"] = f"{type(e).__name__}: {e}"

        rows.append(row)
        n_run += 1

        # iterative save
        if args.save_every > 0 and (n_run % args.save_every == 0):
            df = pd.DataFrame(rows)
            df = df.sort_values(by=["mode", "lr", "neg_weight", "FID"], ascending=True, na_position="last")
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv, index=False)
            print(f"[SAVE] detail -> {out_csv}")

        # iterative summary
        if args.summary_every > 0 and (n_run % args.summary_every == 0):
            df = pd.DataFrame(rows)
            write_summary(df, out_summary)
            print(f"[SAVE] summary -> {out_summary}")

    # final save
    df = pd.DataFrame(rows)
    if "FID" in df.columns:
        df = df.sort_values(by=["mode", "lr", "neg_weight", "FID"], ascending=True, na_position="last")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[DONE] detail saved: {out_csv}")

    write_summary(df, out_summary)
    print(f"[DONE] summary saved: {out_summary}")

    if not args.overwrite:
        print(f"[INFO] skipped already-done: {n_done}")


if __name__ == "__main__":
    main()
