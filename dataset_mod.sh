#!/bin/bash
set -e
shopt -s nullglob

IMAGE_TEXT_DES=true

TEXT_VARIANT_NUM=9
TASK_TEXT="Place the bottle on the pad."

BASE_SRC="/home/grail/training_data/real_data/scenario_1"
BASE_DEST="./data/scenario_1_cam2_cam3"

CAM2_BASE="/home/grail/training_data/real_data/scenario_1_cam_2_marked_render"
CAM3_BASE="/home/grail/training_data/real_data/scenario_1_cam_3_marked_render"

# Loop through all date folders inside CAM2_BASE
DATE_DIR=( "$BASE_SRC"/2025.* "$BASE_SRC"/2026.* )

if (( ${#DATE_DIR[@]} == 0 )); then
  echo "No date dirs found under: $BASE_SRC (patterns: 2025.* 2026.*)"
  exit 1
fi

for DATE_FOLDER in "${DATE_DIR[@]}"; do
  [ -d "$DATE_FOLDER" ] || continue
  
  DATE_NAME=$(basename "$DATE_FOLDER")

  echo "======================================"
  echo "Processing date: $DATE_NAME"
  echo "======================================"

  SRC_DIR="$BASE_SRC/$DATE_NAME"
  DEST_DIR="$BASE_DEST/$DATE_NAME"
  CAM2_SRC="$CAM2_BASE/$DATE_NAME"
  CAM3_SRC="$CAM3_BASE/$DATE_NAME"

  echo "Step 0: copy the dataset"
  python ./dataset_mod/data/dataset_copy.py \
    --src-dir "$SRC_DIR" \
    --dest-dir "$DEST_DIR"

  echo "Step 1: video augmentation"
  python ./dataset_mod/video/video_aug.py \
    --dest-dir "$DEST_DIR" \
    --cam2-src "$CAM2_SRC" \
    --cam3-src "$CAM3_SRC"

  echo "Step 2: augment the parquet data"
  python ./dataset_mod/data/data_aug.py --dest-dir "$DEST_DIR"

  echo "Step 3: generate new episodes and tasks metadata"
  python ./dataset_mod/meta/gen_episode_task.py \
    --dest-dir "$DEST_DIR" \
    --task-text "$TASK_TEXT"

  echo "Step 4: modify the stats and info data"
  python ./dataset_mod/meta/mod_info_stats.py --dest-dir "$DEST_DIR"

  echo "Step 5: generate augmented task descriptions"
  python ./dataset_mod/text/text_gen.py \
    --dest-dir "$DEST_DIR" \
    --n-variants "$TEXT_VARIANT_NUM"

  echo "Step 6: move the generated text into dataset"
  python ./dataset_mod/text/apply_prompt_to_episodes.py \
    --dest-dir "$DEST_DIR"

done

if [ "$IMAGE_TEXT_DES" = true ]; then
    echo "Step 7: Text enhancement based on image annotation"
    python ./dataset_mod/text/text_mod.py \
      --base-dest "$BASE_DEST"
fi

echo "All dates processed successfully."