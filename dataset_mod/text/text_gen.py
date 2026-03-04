#!/usr/bin/env python3
import os
import json
import time
import argparse
from pathlib import Path
from typing import List, Set

from openai import OpenAI


# fixed config (no CLI args)
MODEL = "gpt-5.2"
TEMPERATURE = 0.9
MAX_RETRIES = 5


# -------------------------
# Helpers
# -------------------------
def read_existing_task_indices(out_path: Path) -> Set[int]:
    if not out_path.exists():
        return set()
    done = set()
    with out_path.open("r") as f:
        for line in f:
            try:
                obj = json.loads(line)
                done.add(int(obj["task_index"]))
            except:
                continue
    return done


def jsonl_iter(path: Path):
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_text(s: str) -> str:
    return " ".join(s.strip().split())


def paraphrase_instruction(
    client: OpenAI,
    instruction: str,
    n: int,
) -> List[str]:

    schema = {
        "type": "object",
        "properties": {
            "variants": {
                "type": "array",
                "minItems": n,
                "maxItems": n,
                "items": {"type": "string"},
            }
        },
        "required": ["variants"],
        "additionalProperties": False,
    }

    prompt = f"""
                Generate {n} different natural-language instructions that mean the same thing.

                Rules:
                - Keep the exact same intent and physical meaning.
                - Vary wording (verbs, objects, prepositions, structure).
                - Do NOT add extra constraints, extra steps, or extra objects not implied.
                - Return JSON only, matching the provided schema.

                Instruction: {instruction!r}
            """.strip()

    for attempt in range(MAX_RETRIES):

        try:

            r = client.responses.create(
                model=MODEL,
                input=prompt,
                temperature=TEMPERATURE,
                text={"format": {"type": "json_schema", "name": "aug_opts", "schema": schema}},
            )

            data = json.loads(r.output_text)

            variants = []
            seen = set()

            for v in data["variants"]:

                v = normalize_text(v)

                if v and v not in seen:
                    variants.append(v)
                    seen.add(v)

            if len(variants) == n:
                return variants

        except Exception as e:

            time.sleep(2 ** attempt)

    raise RuntimeError("paraphrase failed")


# -------------------------
# Main
# -------------------------
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dest-dir", default="/home/grail/training_data/real_data/scenario_1")

    parser.add_argument("--n-variants", type=int, default=9)

    args = parser.parse_args()


    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")


    dest_dir = Path(args.dest_dir)

    task_file = dest_dir / "meta/tasks.jsonl"

    out_file = dest_dir / "meta/prompt_augment.jsonl"


    client = OpenAI(api_key=api_key)

    done = read_existing_task_indices(out_file)

    out_file.parent.mkdir(parents=True, exist_ok=True)


    with out_file.open("a") as f:

        for obj in jsonl_iter(task_file):

            task_index = int(obj["task_index"])

            if task_index in done:
                continue

            prompt = normalize_text(obj["task"])

            variants = paraphrase_instruction(
                client,
                prompt,
                args.n_variants
            )

            record = {
                "task_index": task_index,
                "task_des": [prompt] + variants
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            f.flush()

            print("Done task", task_index)


    print("completed")


if __name__ == "__main__":

    main()