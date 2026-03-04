#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

PY_SCRIPT="./annotate/video_annotate.py"
# PY_SCRIPT="./annotate/video_read_coco.py"
PROFILE="s2c2"
OUT_SUBDIR_BY_DATE=true       # true: 输出按日期分子目录；false: 全部写到同一个 out_dir

# 1) 从 video_annotate.py 读取默认配置（BASE/CHUNK_REL/OUT_PREFIX）
eval "$(python ./annotate/video_annotate.py "$PROFILE" --show_config)"

echo "[INFO] PROFILE   = $PROFILE"
echo "[INFO] BASE      = $BASE"
echo "[INFO] CHUNK_REL = $CHUNK_REL"
echo "[INFO] OUT_PREFIX= $OUT_PREFIX"


# 2) 自动扫描日期目录（例如 2025.12.11）
date_dirs=( "$BASE"/2025.* "$BASE"/2026.* )
# date_dirs=("$BASE"/2026.* )
if (( ${#date_dirs[@]} == 0 )); then
  echo "[ERROR] No date dirs found under: $BASE (patterns: 2025.* 2026.*)"
  exit 1
fi

# 3) 遍历日期 → 遍历 mp4 → 调用 video_annotate.py
for dpath in "${date_dirs[@]}"; do
  d="$(basename "$dpath")"
  in_dir="$BASE/$d/$CHUNK_REL"

  if [[ ! -d "$in_dir" ]]; then
    echo "[WARN] Skip (no input dir): $in_dir"
    continue
  fi

  if [[ "$OUT_SUBDIR_BY_DATE" == true ]]; then
    out_dir="$OUT_PREFIX/$d"
  else
    out_dir="$OUT_PREFIX"
  fi
  mkdir -p "$out_dir"

  videos=("$in_dir"/*.mp4)
  if (( ${#videos[@]} == 0 )); then
    echo "[WARN] No mp4 in: $in_dir"
    continue
  fi

  echo "========== Date: $d =========="
  echo "Input : $in_dir"
  echo "Output: $out_dir"

  for video_path in "${videos[@]}"; do
    echo "=== Processing: $video_path"
    python "$PY_SCRIPT" "$PROFILE" \
      --video_path "$video_path" \
      --out_dir "$out_dir"
  done
done

echo "[DONE] Outputs under: $OUT_PREFIX"


for d in "${DATES[@]}"; do
  IN_DIR="${BASE}/${d}/${CHUNK_REL}"
  OUT_DIR="${OUT_PREFIX}/${d}"   # 关键：保留时间，直接拼在后面

  if [[ ! -d "$IN_DIR" ]]; then
    echo "[WARN] Skip: input dir not found: $IN_DIR"
    continue
  fi

  mkdir -p "$OUT_DIR"

  echo "========== Date: $d =========="
  echo "Input : $IN_DIR"
  echo "Output: $OUT_DIR"

  vids=("$IN_DIR"/*.mp4)
  if (( ${#vids[@]} == 0 )); then
    echo "[WARN] No mp4 found in $IN_DIR"
    continue
  fi

  for v in "${vids[@]}"; do
    echo "[RUN] $v"
    python "$PY_SCRIPT" \
    "$PROFILE" 
    --video_path "$v" \
  done
done

echo "All done."
