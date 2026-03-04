from pathlib import Path
import argparse

import numpy as np

from sam3.model_builder import build_sam3_video_predictor
from coco_io import load_outputs_merged_from_coco_json
from video_annotate import build_parser, PROFILES, OBJ_ID, VideoPromptRunner, VideoMaskRenderer

RENDER_TAG = "render" # modify this

def build_render_parser() -> argparse.ArgumentParser:
    # 复用你已有的 profile/subcommand parser
    parser = build_parser()
    parser.add_argument(
        "--render_tag",
        type=str,
        default=RENDER_TAG,
        help="Tag used for rendered video filename.",
    )
    return parser

def main():
    predictor = build_sam3_video_predictor()
    args = build_render_parser().parse_args()
    spec = PROFILES[args.profile]

    video_path = args.video_path
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    stem = Path(args.video_path).stem
    out_coco_path = Path(args.out_dir) / f"{stem}.coco.json"
    
    out_dir = Path(args.out_dir).resolve()
    date_dir = out_dir.name          # 2025.12.11
    out_prefix = out_dir.parent      # .../scenario_2_cam2
    render_root = Path(f"{out_prefix}_{args.render_tag}")  # .../scenario_2_cam2_no_hand
    out_dir_render = render_root / date_dir                # .../scenario_2_cam2_no_hand/2025.12.11
    out_dir_render.mkdir(parents=True, exist_ok=True)
    
    out_video_path =  out_dir_render/ f"{stem}_{spec.out_tag}.mp4"

    runner = VideoPromptRunner(predictor, args, spec)
    video_frames = runner.load_video_frames_for_vis(video_path)

    outputs_merged = load_outputs_merged_from_coco_json(out_coco_path)

    # ---- KEEP only selected objects (by name) ----
    # example: exclude hand
    KEEP_NAMES = tuple(n for n in spec.objects if n != "hand")
    KEEP_IDS = tuple(OBJ_ID[n] for n in KEEP_NAMES)

    for frame_idx, out in outputs_merged.items():
        ids = out["out_obj_ids"]
        if ids is None or len(ids) == 0:
            continue

        # sanity
        if "out_boxes_xywh" in out and out["out_boxes_xywh"] is not None:
            assert out["out_boxes_xywh"].shape[0] == ids.shape[0]
        if "out_binary_masks" in out and out["out_binary_masks"] is not None:
            assert out["out_binary_masks"].shape[0] == ids.shape[0]

        keep = np.isin(ids, KEEP_IDS)
        out["out_obj_ids"] = ids[keep]
        out["out_boxes_xywh"] = out["out_boxes_xywh"][keep]
        out["out_binary_masks"] = out["out_binary_masks"][keep]
    # ---- end KEEP

    color_by_id = {OBJ_ID[name]: tuple(getattr(args, f"color_{name}")) for name in KEEP_NAMES}

    renderer = VideoMaskRenderer(
        video_frames_for_vis=video_frames,
        outputs_merged=outputs_merged,
        color_by_id=color_by_id,
        fps=30,
        alpha=0.18,
        thickness=2,
    )
    renderer.render(out_video_path)

if __name__ == "__main__":
    main()