import os
import json
import argparse
from pathlib import Path

from annotate.config.profile import OBJ, OBJ_ID, DATA_ROOT, PROFILES
from annotate.utils.sam3_util import VideoPointAnnotator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    sub = parser.add_subparsers(dest="profile", required=True)

    for key, spec in PROFILES.items():
        sp = sub.add_parser(key)
        sp.add_argument("--video_path", type=str, default=DATA_ROOT/f"scenario_{spec.scenario}/{spec.date_dir}/videos/chunk-{spec.chunk:03d}/observation.images.cam{spec.cam}/episode_{spec.episode:06d}.mp4")
        sp.add_argument("--out_dir", type=str, default=DATA_ROOT/f"scenario_{spec.scenario}_cam_{spec.cam}_annotate/{spec.date_dir}")
        sp.add_argument("--refine_episode", type=list, default=spec.refine_episode)
        sp.add_argument(
            "--show_config", 
            action="store_true",
            help="Print derived paths (BASE/CHUNK_REL/OUT_PREFIX) and exit")
    return parser

def point_generation(args, spec, video_path):
    num = int(video_path.stem.split('_')[1])
    prompt_db = {"profile": args.profile,
                 "episode": num,
                 "objects": []}

    ORDER = list(spec.objects)
    objs = []
    for name in ORDER:
        obj_id = OBJ_ID[name]
        prompt = spec.prompt(obj_id)
        frame_index = spec.frame(obj_id)
        if not prompt:   # 可选：允许某个对象不做
            continue
        objs.append(dict(
            obj_id=obj_id,
            prompt=prompt,
            frame_index=frame_index,
        ))
    for cfg in objs:
        frame_index = cfg.get("frame_index", 0)
        obj_id = cfg['obj_id']
        name = OBJ[obj_id]["name"]
        annotator = VideoPointAnnotator(video_path, frame_index, name)
        points, labels = annotator.run()
        object_instance = {"obj_id": cfg["obj_id"],
                           "prompt": cfg["prompt"],
                           "frame_index": cfg["frame_index"],
                           "points": points.tolist(),
                           "labels": labels.tolist(),}
        prompt_db["objects"].append(object_instance)
    return prompt_db

def main():

    args = build_parser().parse_args()
    spec = PROFILES[args.profile]

    out_dir = Path(args.out_dir)/"refine_folder"
    video_path = Path(args.video_path)
    os.makedirs(out_dir, exist_ok=True)
    refine_path = out_dir / f"refine_point.json"
    

    refine_episode = args.refine_episode
    if type(refine_episode) == int: refine_episode = [refine_episode]
    new_paths = [
        video_path.with_name(f"episode_{i:06d}.mp4")
        for i in refine_episode
    ]
    prompt_dataset = []
    for path in new_paths:
        prompt_db = point_generation(args, spec, path)
        prompt_dataset.append(prompt_db)

    with open(refine_path, "w") as f:
        json.dump(prompt_dataset, f, indent=2)

if __name__ == "__main__":
    main()