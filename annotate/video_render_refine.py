from pathlib import Path
import argparse
import cv2
import numpy as np
import os

from sam3.model_builder import build_sam3_video_predictor
from annotate.utils.coco_io import load_outputs_merged_from_coco_json
from annotate.video_annotate_point_fast import build_parser
from annotate.video_annotate import PROFILES, OBJ_ID, VideoPromptRunner
from annotate.video_render import VideoMaskRenderer
from annotate.config.profile import DATA_ROOT


def main():
    predictor = build_sam3_video_predictor()
    args = build_parser().parse_args()
    spec = PROFILES[args.profile]

    video_path = Path(args.video_path)
    out_dir = Path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    refine_folder = out_dir / "refine_folder"
    new_coco_paths = sorted(list(refine_folder.glob("*.coco.json")))
    new_video_paths = [video_path.parent / p.name.replace(".coco.json", ".mp4") for p in new_coco_paths]

    for video_path in new_video_paths:
        out_coco_path = refine_folder /f"{video_path.stem}.coco.json"
        args.video_path = video_path
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

        color_by_id = {OBJ_ID[name]: spec.color(OBJ_ID[name]) for name in KEEP_NAMES}
        render_dir = DATA_ROOT / f"scenario_{spec.scenario}_cam_{spec.cam}_render" / out_dir.name / "refine_folder"
        render_dir.mkdir(parents=True, exist_ok=True)
        
        out_video_path = render_dir / video_path.name

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