#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

ANNOTATE_MODULE="annotate.video_annotate"
RENDER_MODULE="annotate.video_render"

PROFILE="s2c2"
OUT_SUBDIR_BY_DATE=true       # true: 输出按日期分子目录；false: 全部写到同一个 out_dir
IS_RESUME=false
############# START resume control
# IS_RESUME=true
# RESUME_DATE="2025.12.02"
# START_BASENAME="episode_000021.mp4"
############ END resume control


# 1) 从 video_annotate.py 读取默认配置（BASE/CHUNK_REL/OUT_PREFIX）
eval "$(python -m "$ANNOTATE_MODULE" "$PROFILE" --show_config)"

echo "[INFO] PROFILE   = $PROFILE"
echo "[INFO] BASE      = $BASE"
echo "[INFO] CHUNK_REL = $CHUNK_REL"
echo "[INFO] OUT_PREFIX= $OUT_PREFIX"


# 2) 自动扫描日期目录（例如 2025.12.11）
date_dirs=( "$BASE"/2025.* "$BASE"/2026.* )
if (( ${#date_dirs[@]} == 0 )); then
  echo "[ERROR] No date dirs found under: $BASE (patterns: 2025.* 2026.*)"
  exit 1
fi

# 3) 遍历日期 → 遍历 mp4
run_phase() {
  local py_module="$1"
  echo
  echo "=============================="
  echo "[PHASE] Running: $py_module"
  echo "=============================="

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

    # 默认从头开始,只有打开 IS_RESUME 才启用 resume 逻辑
    started=1
    if [[ "$IS_RESUME" == true ]]; then
      if [[ -n "$RESUME_DATE" ]]; then
        if [[ "$d" == "$RESUME_DATE" ]]; then
          started=0
        fi
      else
        started=0
      fi
    fi

    for video_path in "${videos[@]}"; do

      base="$(basename "$video_path")"

      if (( started == 0 )); then
        if [[ "$base" == "$START_BASENAME" ]]; then
          started=1
        else
          continue
        fi
      fi

      echo "=== Processing: $py_module: $video_path"
      python -m "$py_module" "$PROFILE" \
        --video_path "$video_path" \
        --out_dir "$out_dir"
    done
  done
}
# phase 1: 先全部 annotate
run_phase "$ANNOTATE_MODULE"

# phase 2: 再全部 render
run_phase "$RENDER_MODULE"
echo "[DONE] Done"

