#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ROOT="/home/grail/training_data/real_data/stage2_augment"

# Edit these two values for the usual dataset run.
SCENARIO="scenario_23"
ANNOTATE_CAMS=("cam_2")

usage() {
  cat <<'EOF'
Usage:
  ./run_dataset_pipeline.sh [scenario] [annotate_cams]

Examples:
  ./run_dataset_pipeline.sh
  ./run_dataset_pipeline.sh 10 cam_2
  ./run_dataset_pipeline.sh scenario_10 cam_2,cam_3
EOF
}

normalize_scenario() {
  local scenario="$1"
  if [[ "$scenario" == scenario_* ]]; then
    printf '%s' "$scenario"
  else
    printf 'scenario_%s' "$scenario"
  fi
}

normalize_annotate_cam() {
  local camera="$1"
  camera="${camera// /}"
  camera="${camera//$'\t'/}"
  camera="${camera#observation.images.}"
  if [[ "$camera" =~ ^cam[0-9]+$ ]]; then
    printf 'cam_%s' "${camera#cam}"
  else
    printf '%s' "$camera"
  fi
}

sample_camera_from_annotate_cam() {
  local annotate_cam="$1"
  printf 'observation.images.%s' "${annotate_cam/_/}"
}

run_step() {
  printf '\n==> %s\n' "$*"
  "$@"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -gt 2 ]]; then
  usage >&2
  exit 1
fi

SCENARIO="$(normalize_scenario "${1:-$SCENARIO}")"

if [[ $# -ge 2 ]]; then
  IFS=',' read -r -a RAW_ANNOTATE_CAMS <<< "$2"
  ANNOTATE_CAMS=()
  for camera in "${RAW_ANNOTATE_CAMS[@]}"; do
    ANNOTATE_CAMS+=("$(normalize_annotate_cam "$camera")")
  done
else
  for index in "${!ANNOTATE_CAMS[@]}"; do
    ANNOTATE_CAMS[$index]="$(normalize_annotate_cam "${ANNOTATE_CAMS[$index]}")"
  done
fi

SAMPLE_CAMERA="$(sample_camera_from_annotate_cam "${ANNOTATE_CAMS[0]}")"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  printf 'error: OPENAI_API_KEY is required for generate_visual_prompt_augment.py\n' >&2
  exit 1
fi

printf 'Pipeline configuration:\n'
printf '  scenario: %s\n' "$SCENARIO"
printf '  annotate_cams: %s\n' "${ANNOTATE_CAMS[*]}"
printf '  sample_camera: %s\n' "$SAMPLE_CAMERA"

run_step "$PYTHON_BIN" "$SCRIPT_DIR/dataset_mod2/move_file.py" \
  --root "$ROOT" \
  --scenario "$SCENARIO" \
  --annotate-cams "${ANNOTATE_CAMS[@]}"

run_step "$PYTHON_BIN" "$SCRIPT_DIR/dataset_mod2/fix_coco_annotations.py" \
  --root "$ROOT" \
  --scenarios "$SCENARIO"

run_step "$PYTHON_BIN" "$SCRIPT_DIR/dataset_mod2/sample_video_frames.py" \
  --root "$ROOT" \
  --scenarios "$SCENARIO" \
  --camera-name "$SAMPLE_CAMERA"

run_step "$PYTHON_BIN" "$SCRIPT_DIR/dataset_mod2/generate_visual_prompt_augment.py" \
  --root "$ROOT" \
  --scenarios "$SCENARIO"

run_step "$PYTHON_BIN" "$SCRIPT_DIR/dataset_mod2/assign_prompt_augment.py" \
  --root "$ROOT" \
  --scenarios "$SCENARIO"

printf '\nPipeline completed.\n'
