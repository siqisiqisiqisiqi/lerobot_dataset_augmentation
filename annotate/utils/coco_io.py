"""
Minimal COCO saver for ONE VIDEO (frame-indexed outputs)

Input:
  outputs_merged: dict[int frame_idx] -> dict with:
    - out_obj_ids:      (N,) int
    - out_boxes_xywh:   (N,4) float, NORMALIZED [cx, cy, w, h] in [0,1]
    - out_binary_masks: (N,H,W) bool
Optional (ignored):
    - out_probs, frame_stats, etc.

Output:
  One COCO JSON file:
    - images: one entry per frame (id = frame_idx)
    - annotations: one entry per instance per frame
      segmentation stored as RLE (pycocotools)
      bbox stored as COCO pixel xywh (top-left x,y + w,h)
      track_id stored as custom field to preserve out_obj_ids
"""

import json
import numpy as np
from pycocotools import mask as mask_utils


# ============================================================
# Part 1) Helpers: empty-mask check + bbox conversion
# ============================================================

def is_empty_mask(mask: np.ndarray) -> bool:
    """Your exact emptiness criterion."""
    return mask is None or mask.size == 0 or (not mask.any())


def norm_cxcywh_to_coco_xywh(norm_box, W: int, H: int):
    """
    Convert NORMALIZED [cx, cy, w, h] in [0,1]
    to COCO PIXEL [x_top_left, y_top_left, w, h].
    """
    cx, cy, bw, bh = [float(x) for x in norm_box]
    x = (cx - bw / 2.0) * W
    y = (cy - bh / 2.0) * H
    w = bw * W
    h = bh * H
    return [float(x), float(y), float(w), float(h)]


# ============================================================
# Part 2) Main function: save outputs_merged -> COCO JSON
# ============================================================

def save_outputs_merged_to_coco_json(
    outputs_merged: dict,
    out_json_path: str,
    video_name: str = "video",
    category_id: int = 1,
    category_name: str = "object",
):
    """
    Minimal COCO export:
      - masks -> RLE
      - boxes -> convert normalized cxcywh to pixel xywh
      - obj ids -> saved as "track_id" (custom)
    """

    # --------------------------------------------------------
    # Part 2.1) Determine frame order and infer (H,W)
    # --------------------------------------------------------
    frame_idxs = sorted(outputs_merged.keys())

    H = W = None
    for fi in frame_idxs:
        masks = outputs_merged[fi].get("out_binary_masks", None)
        if isinstance(masks, np.ndarray) and masks.ndim == 3 and masks.shape[0] > 0:
            H, W = int(masks.shape[1]), int(masks.shape[2])
            break

    if H is None or W is None:
        # raise ValueError(
        #     "Cannot infer H,W because all frames have 0 masks. "
        #     "If you still want to save, hardcode H,W and modify this function."
        # )
        H, W = 360, 640

    # --------------------------------------------------------
    # Part 2.2) Build the minimal COCO skeleton
    # --------------------------------------------------------
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": int(category_id), "name": category_name}],
    }

    # --------------------------------------------------------
    # Part 2.3) Fill images + annotations
    # --------------------------------------------------------
    ann_id = 1

    for fi in frame_idxs:
        out = outputs_merged[fi]

        # ---- (a) Add one COCO "image" per frame ----
        image_id = int(fi)
        coco["images"].append({
            "id": image_id,
            "file_name": f"{video_name}_frame_{fi:06d}.jpg",  # can be any placeholder
            "height": H,
            "width": W,
            "frame_index": int(fi),  # custom helper field
        })

        # ---- (b) Read required fields (skip frame if missing) ----
        obj_ids = out.get("out_obj_ids", None)
        boxes   = out.get("out_boxes_xywh", None)
        masks   = out.get("out_binary_masks", None)

        if not isinstance(masks, np.ndarray) or masks.ndim != 3 or masks.shape[0] == 0:
            continue

        N = masks.shape[0]

        # Safety: if obj_ids/boxes are missing or shape mismatch, we still try
        has_ids = isinstance(obj_ids, np.ndarray) and obj_ids.shape[0] == N
        has_box = isinstance(boxes, np.ndarray) and boxes.shape[0] == N and boxes.shape[1] == 4

        for i in range(N):
            m = masks[i]
            if is_empty_mask(m):
                continue

            # ---- (c) Encode mask as COCO RLE ----
            # pycocotools expects uint8 and Fortran-order array
            m_u8 = np.asfortranarray(m.astype(np.uint8))
            rle = mask_utils.encode(m_u8)              # counts is bytes
            rle["counts"] = rle["counts"].decode("ascii")  # JSON needs str

            # ---- (d) Convert bbox: normalized cxcywh -> pixel xywh ----
            if has_box:
                bbox = norm_cxcywh_to_coco_xywh(boxes[i], W=W, H=H)
            else:
                bbox = [0.0, 0.0, 0.0, 0.0]  # fallback

            # ---- (e) Minimal annotation entry ----
            ann = {
                "id": ann_id,
                "image_id": image_id,
                "category_id": int(category_id),
                "segmentation": rle,
                "bbox": bbox,
                "iscrowd": 0,

                # custom field: preserve your object id (tracking id)
                "track_id": int(obj_ids[i]) if has_ids else -1,
            }
            coco["annotations"].append(ann)
            ann_id += 1

    # --------------------------------------------------------
    # Part 2.4) Write JSON to disk
    # --------------------------------------------------------
    with open(out_json_path, "w") as f:
        json.dump(coco, f)

    print(f"[COCO SAVED] {out_json_path}")
    print(f"  images: {len(coco['images'])}")
    print(f"  annotations: {len(coco['annotations'])}")


# ============================================================
# Part 3) Example call (paste after you have outputs_merged)
# ============================================================
# save_outputs_merged_to_coco_json(
#     outputs_merged=outputs_merged,  # <-- must exist in your runtime
#     out_json_path="/tmp/merged_outs.coco.json",
#     video_name="episode_000186_cam2",
#     category_id=1,
#     category_name="object",
# )


import json
import numpy as np
from pycocotools import mask as mask_utils

# ============================================================
# Part 1) bbox 反转换：COCO 像素 xywh -> 归一化 cxcywh
# ============================================================
def coco_xywh_to_norm_cxcywh(bbox_xywh, W: int, H: int):
    """
    Convert COCO PIXEL [x_top_left, y_top_left, w, h]
    to NORMALIZED [cx, cy, w, h] in [0,1] (same style as your original out_boxes_xywh).
    """
    x, y, w, h = [float(v) for v in bbox_xywh]
    cx = (x + w / 2.0) / W
    cy = (y + h / 2.0) / H
    bw = w / W
    bh = h / H
    return [float(cx), float(cy), float(bw), float(bh)]


# ============================================================
# Part 2) Main function: load COCO JSON -> outputs_merged
# ============================================================
def load_outputs_merged_from_coco_json(coco_json_path: str):
    """
    Return:
      outputs_merged: dict[int frame_idx] -> {
        'out_obj_ids': (N,) int32
        'out_boxes_xywh': (N,4) float32, NORMALIZED cxcywh
        'out_binary_masks': (N,H,W) bool
      }
    Notes:
      - Uses image['id'] as frame_idx (matches how you saved).
      - If a frame has no annotations, returns N=0 arrays for that frame.
    """

    # --------------------------------------------------------
    # Part 2.1) Read JSON
    # --------------------------------------------------------
    with open(coco_json_path, "r") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])

    # --------------------------------------------------------
    # Part 2.2) Build image_id -> (H,W) and frame_idx list
    # --------------------------------------------------------
    # You saved image_id = frame_idx, but we still handle frame_index if present.
    img_meta = {}
    frame_idxs = []
    for img in images:
        image_id = int(img["id"])
        fi = int(img.get("frame_index", image_id))
        H = int(img["height"])
        W = int(img["width"])
        img_meta[image_id] = {"frame_idx": fi, "H": H, "W": W}
        frame_idxs.append(fi)

    # If images list is empty, nothing to load
    if not img_meta:
        return {}

    # --------------------------------------------------------
    # Part 2.3) Group annotations by image_id
    # --------------------------------------------------------
    anns_by_image = {}
    for ann in annotations:
        image_id = int(ann["image_id"])
        anns_by_image.setdefault(image_id, []).append(ann)

    # --------------------------------------------------------
    # Part 2.4) Reconstruct outputs_merged frame by frame
    # --------------------------------------------------------
    outputs_merged = {}

    # Sort by frame_idx for stable order
    # (We sort images by their stored frame_idx)
    image_ids_sorted = sorted(img_meta.keys(), key=lambda iid: img_meta[iid]["frame_idx"])

    for image_id in image_ids_sorted:
        meta = img_meta[image_id]
        fi = meta["frame_idx"]
        H = meta["H"]
        W = meta["W"]

        anns = anns_by_image.get(image_id, [])

        obj_ids = []
        boxes_norm = []
        masks = []

        for ann in anns:
            # ---- (a) track_id -> out_obj_ids ----
            obj_ids.append(int(ann.get("track_id", -1)))

            # ---- (b) bbox(pixel xywh) -> normalized cxcywh ----
            bbox_xywh = ann.get("bbox", [0, 0, 0, 0])
            boxes_norm.append(coco_xywh_to_norm_cxcywh(bbox_xywh, W=W, H=H))

            # ---- (c) RLE -> mask(H,W) bool ----
            rle = ann["segmentation"]

            # If counts was saved as str (we did), pycocotools can decode directly.
            m = mask_utils.decode(rle)  # uint8 array (H,W)
            masks.append(m.astype(bool))

        if len(masks) == 0:
            outputs_merged[fi] = {
                "out_obj_ids": np.zeros((0,), dtype=np.int32),
                "out_boxes_xywh": np.zeros((0, 4), dtype=np.float32),
                "out_binary_masks": np.zeros((0, H, W), dtype=bool),
            }
        else:
            outputs_merged[fi] = {
                "out_obj_ids": np.array(obj_ids, dtype=np.int32),
                "out_boxes_xywh": np.array(boxes_norm, dtype=np.float32),
                "out_binary_masks": np.stack(masks, axis=0),  # (N,H,W)
            }

    return outputs_merged



# ============================================================
# Part 3) Example call 
# ============================================================
# from coco_io
# outputs_merged = load_outputs_merged_from_coco_json("/tmp/merged_outs.coco.json")
