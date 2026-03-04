#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd
import re

EP_RE = re.compile(r"episode_(\d+)")

def ep_idx(path: Path) -> int:
    m = EP_RE.search(path.name)
    if not m:
        raise RuntimeError(f"Bad filename (need episode_XXXXXX): {path.name}")
    return int(m.group(1))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest-dir", default="./data/scenario_1_cam2_cam3",
                        help='the augment data root')
    args = parser.parse_args()
    data_root = Path(args.dest_dir)/"data/chunk-000"

    files = sorted(data_root.glob("*.parquet"))
    if not files:
        raise RuntimeError(f"No parquet files found in: {data_root}")

    # base offsets (same logic as your original code)
    last_file = files[-1]
    max_episode = ep_idx(last_file)

    # read only the index column to get last row index
    df_last_index = pd.read_parquet(last_file, columns=["index"])
    last_index = int(df_last_index["index"].iloc[-1])

    episode_offset = max_episode + 1
    index_offset = last_index + 1

    for f in files:
        df = pd.read_parquet(f)
        new_episode = ep_idx(f) + episode_offset

        df["episode_index"] = new_episode
        df["index"] = df["index"] + index_offset

        out = data_root / f"episode_{new_episode:06d}.parquet"
        df.to_parquet(out)

    print("Complete the data augmentation.")

if __name__ == "__main__":
    main()