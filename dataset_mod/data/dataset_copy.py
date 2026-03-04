#!/usr/bin/env python3
import shutil
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--dest-dir", required=True, help='destination path')
parser.add_argument("--src-dir", required=True, help='source path')
args = parser.parse_args()
dest_dir = Path(args.dest_dir)
src_dir = Path(args.src_dir)

print(dest_dir)
print(src_dir)
shutil.copytree(src_dir, dest_dir)
print("Dataset copied successfully.")