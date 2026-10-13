#!/usr/bin/env bash

# ---------- user config ----------
ROOT_DIR=/MIR/nas1/dreamyou070/NegatingDiffusion/Result/Compare/1_negation_coco
REAL_DIR=/MIR/nas1/dreamyou070/hf_cache/annotations/val2017
DEVICE=cuda:0
DIMS=2048
BATCH=1
NUM_WORKERS=0
# --------------------------------
OUT_CSV=${ROOT_DIR}/fid_summary.csv
LOG_DIR=${ROOT_DIR}/_5_fid_logs
REAL_STATS=${ROOT_DIR}/real_stats_inception${DIMS}.npz

mkdir -p "${LOG_DIR}"

echo "[check] REAL_DIR=${REAL_DIR}"
echo "[check] sample files:"
ls -1 "${REAL_DIR}" | head -n 5

echo "[1/2] build real stats: ${REAL_STATS}"
if [[ ! -s "${REAL_STATS}" ]]; then
  python -m pytorch_fid --save-stats \
    --device "${DEVICE}" --dims "${DIMS}" \
    --batch-size "${BATCH}" --num-workers "${NUM_WORKERS}" \
    "${REAL_DIR}" "${REAL_STATS}"
fi

echo "[2/2] per-model fid -> ${OUT_CSV}"

ROOT_DIR="${ROOT_DIR}" REAL_STATS="${REAL_STATS}" OUT_CSV="${OUT_CSV}" LOG_DIR="${LOG_DIR}" \
DEVICE="${DEVICE}" DIMS="${DIMS}" BATCH="${BATCH}" NUM_WORKERS="${NUM_WORKERS}" \
python - <<'PY'
import os, re, csv, glob, subprocess, sys
from pathlib import Path

root_dir   = Path(os.environ["ROOT_DIR"])
real_stats = Path(os.environ["REAL_STATS"])
out_csv    = Path(os.environ["OUT_CSV"])
log_dir    = Path(os.environ["LOG_DIR"])
device     = os.environ["DEVICE"]
dims       = os.environ["DIMS"]
batch      = os.environ["BATCH"]
num_workers= os.environ["NUM_WORKERS"]

img_exts = (".png",".jpg",".jpeg",".bmp",".webp")

def find_images(d: Path):
    files = []
    for ext in img_exts:
        files += list(d.glob(f"*{ext}"))
    return files

def parse_fid(text: str):
    # 다양한 출력 포맷 대비: FID가 포함된 줄에서 숫자 하나 뽑기
    for line in text.splitlines()[::-1]:
        if "FID" in line:
            m = re.search(r"([0-9]+(?:\.[0-9]+)?)", line)
            if m:
                return float(m.group(1))
    return None

rows = []
subdirs = sorted([p for p in root_dir.iterdir() if p.is_dir()])

for d in subdirs:
    model = d.name
    imgs = find_images(d)
    if not imgs:
        rows.append([model, str(d), "", 0, "no_images"])
        print(model, "no_images")
        continue

    log_path = log_dir / f"{model}.log"
    cmd = [
        sys.executable, "-m", "pytorch_fid",
        "--device", device, "--dims", dims,
        "--batch-size", batch, "--num-workers", num_workers,
        str(real_stats), str(d)
    ]

    p = subprocess.run(cmd, capture_output=True, text=True)
    text = (p.stdout or "") + "\n" + (p.stderr or "")
    log_path.write_text(
        "CMD: " + " ".join(cmd) + "\n\n" +
        f"returncode: {p.returncode}\n\n" +
        text
    )

    fid = parse_fid(text) if p.returncode == 0 else None
    status = "ok" if (p.returncode == 0 and fid is not None) else ("fail" if p.returncode != 0 else "no_fid_line")
    rows.append([model, str(d), "" if fid is None else fid, len(imgs), status])
    print(model, fid, status)

with out_csv.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "dir", "fid", "num_images", "status"])
    w.writerows(rows)

print("saved", out_csv)
print("logs", log_dir)
PY

echo "done"
# export ROOT_DIR=/MIR/nas1/dreamyou070/NegatingDiffusion/Result/Compare/1_negation_coco
# export REAL_DIR=/MIR/nas1/dreamyou070/hf_cache/annotations/val2017
  #bash compute_fid_multi.sh