#!/usr/bin/env bash
set -euo pipefail

# Input dataset.  The output is created beside it automatically.
DATA_PATH="/home/grail/training_data/real_data/robotwin/stack_blocks_two_order-demo_clean-zoyi"
CAMERA="cam2"

# Multiple episodes can be written either as EPISODES=(44 89) or EPISODES=(44,89).
# Leave empty to process every episode in the selected camera directory.
# 9,75,83,86,87,90,94
# EPISODES=(8,9,15,21,28,35,47,49,53,94,97,99,)
EPISODES=()

# Each entry follows profile.py's explicit object format:
#   "OBJECT_ID|OBJECT_NAME|SAM3_PROMPT"
# small rectangular deck of playing cards on the table
OBJECTS=(
  "1|red block|red block"
  "2|green block|green block"
)

# Leave empty for: ${DATA_PATH}_sam3_${CAMERA}
OUTPUT_PATH=""
# Default behavior is to overwrite existing annotations.
# Set this to true only when you want to skip already annotated episodes.
SKIP_EXISTING=false

cmd=(
  python -m annotate.robotwin_video_adapter
  --data-path "$DATA_PATH"
  --camera "$CAMERA"
)

for object_spec in "${OBJECTS[@]}"; do
  IFS='|' read -r object_id object_name object_prompt <<< "$object_spec"
  cmd+=(--object "$object_id" "$object_name" "$object_prompt")
done

for episode_group in "${EPISODES[@]}"; do
  IFS=',' read -ra episode_values <<< "$episode_group"
  for episode in "${episode_values[@]}"; do
    [[ -n "$episode" ]] || continue
    cmd+=(--episode "$episode")
  done
done

if [[ -n "$OUTPUT_PATH" ]]; then
  cmd+=(--output-path "$OUTPUT_PATH")
fi
if [[ "$SKIP_EXISTING" == true ]]; then
  cmd+=(--skip-existing)
fi

echo "[INFO] DATA_PATH=$DATA_PATH"
echo "[INFO] CAMERA=$CAMERA"
echo "[INFO] OBJECTS=${OBJECTS[*]}"
echo "[INFO] EPISODES=${EPISODES[*]:-all}"
"${cmd[@]}"
