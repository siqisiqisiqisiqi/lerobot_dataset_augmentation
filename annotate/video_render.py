from pathlib import Path
import argparse
import cv2
import numpy as np
import os

from sam3.model_builder import build_sam3_video_predictor
from annotate.utils.coco_io import load_outputs_merged_from_coco_json
from annotate.video_annotate import build_parser, PROFILES, OBJ_ID, VideoPromptRunner
from annotate.config.profile import DATA_ROOT

class VideoMaskRenderer:
    def __init__(
        self,
        video_frames_for_vis:list,
        outputs_merged: dict,
        color_by_id: dict[int, tuple[int, int, int]],
        fps: int = 30,
        alpha: float = 0.18,
        thickness: int = 2,
        fourcc: str = "mp4v",
    ):
        self.video_frames_for_vis = video_frames_for_vis
        self.outputs_merged = outputs_merged
        self.color_by_id = color_by_id  # BGR colors
        self.alpha = float(alpha)
        self.thickness = int(thickness)
        self.fps = int(fps)
        self.fourcc = fourcc
        self.H, self.W = self.video_frames_for_vis[0].shape[:2]

    def overlay_mask_bgr(
        self,
        frame_bgr, 
        mask_bool, 
        fill_color_bgr, 
        outline_color_bgr=None, 
    ):
        """frame_bgr: (H,W,3) uint8, mask_bool: (H,W) bool"""
        if mask_bool is None or mask_bool.size == 0:
            return frame_bgr
        m = (mask_bool.astype(np.uint8) * 255)  # uint8, values 0/255
        if m.max() == 0:
            return frame_bgr
        
        if outline_color_bgr is None:
            outline_color_bgr = fill_color_bgr

        overlay = frame_bgr.copy()
        overlay[m>0] = fill_color_bgr
        frame_bgr = cv2.addWeighted(overlay, self.alpha, frame_bgr, 1 - self.alpha, 0)

        res = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = res[0] if len(res) == 2 else res[1]
        cv2.drawContours(frame_bgr, contours, -1, outline_color_bgr, self.thickness)

        return frame_bgr

    def render(self, out_video_path):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(
            out_video_path, 
            fourcc, 
            self.fps, 
            (self.W, self.H),
        )
        assert vw.isOpened(), f"Failed to open VideoWriter: {out_video_path}"
        n_frames = len(self.video_frames_for_vis)
        for frame_idx in range(n_frames):
            # if frame_idx not in self.outputs_merged: 
            #     continue
            frame_bgr = cv2.cvtColor(self.video_frames_for_vis[frame_idx], cv2.COLOR_RGB2BGR) #original
            out = self.outputs_merged.get(frame_idx, None)
            if out is not None:
                masks = out.get("out_binary_masks", None)  # (3,H,W)
                obj_ids = out.get("out_obj_ids", None)
                if masks is not None and obj_ids is not None:
                    N = obj_ids.shape[0]
                    for i in range(N):
                        mask = masks[i]
                        obj_id = obj_ids[i]
                        if mask is None or mask.size == 0 or not mask.any():
                            continue
                        c = self.color_by_id.get(obj_id)
                        frame_bgr = self.overlay_mask_bgr(frame_bgr, mask, c)
            vw.write(frame_bgr)
        vw.release()
        print("Saved video to:", out_video_path)
        return


def main():
    predictor = build_sam3_video_predictor()
    args = build_parser().parse_args()
    spec = PROFILES[args.profile]

    BASE=DATA_ROOT/f"scenario_{spec.scenario}"
    CHUNK_REL=f"videos/chunk-{spec.chunk:03d}/observation.images.cam{spec.cam}"
    VIDEO_BASENAME = f"episode_{spec.episode:06d}.mp4"
    OUT_PREFIX=DATA_ROOT/f"scenario_{spec.scenario}_cam_{spec.cam}_annotate"

    # use .sh script
    if args.show_config:
        print(f'BASE="{BASE}"')
        print(f'CHUNK_REL="{CHUNK_REL}"')
        print(f"VIDEO_BASENAME={VIDEO_BASENAME}")
        print(f'OUT_PREFIX="{OUT_PREFIX}"')
        return
    # END use .sh script

    video_path = Path(args.video_path)
    out_dir = Path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    out_coco_path = out_dir / f"{video_path.stem}.coco.json"

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
    render_dir = DATA_ROOT / f"scenario_{spec.scenario}_cam_{spec.cam}_render" / out_dir.name
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