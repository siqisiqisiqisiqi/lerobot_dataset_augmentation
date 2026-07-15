from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any

from openai import OpenAI


# DEFAULT_SCENARIOS = ("scenario_9", "scenario_10")
DEFAULT_SCENARIOS = ("scenario_10", )
DEFAULT_N_VARIANTS = 10
DEFAULT_MODEL = "gpt-5.4"
DEFAULT_TEMPERATURE = 0.9
DEFAULT_IMAGE_DETAIL = "low"
DEFAULT_OUTPUT_NAME = "prompt_augment_visual.jsonl"
MAX_RETRIES = 5
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False))
            file.write("\n")
    tmp_path.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False))
        file.write("\n")


def iter_meta_dirs(
    root: Path,
    scenario_names: tuple[str, ...],
    dates: set[str] | None,
) -> list[tuple[str, str, Path]]:
    meta_dirs: list[tuple[str, str, Path]] = []

    for scenario_name in scenario_names:
        scenario_dir = root / scenario_name
        if not scenario_dir.exists():
            continue

        for date_dir in sorted(path for path in scenario_dir.iterdir() if path.is_dir()):
            if dates is not None and date_dir.name not in dates:
                continue

            meta_dir = date_dir / "meta"
            if (meta_dir / "tasks.jsonl").exists():
                meta_dirs.append((scenario_name, date_dir.name, meta_dir))

    return meta_dirs


def resolve_metadata_image_path(root: Path, sample_dir: Path, image_text: str) -> Path:
    path = Path(image_text)
    if path.is_absolute():
        return path

    root_relative = root / path
    if root_relative.exists():
        return root_relative

    return sample_dir / path.name


def read_sampled_images(root: Path, scenario_name: str, date_name: str) -> list[Path]:
    sample_dir = root / f"{scenario_name}_sampled_image" / date_name
    if not sample_dir.exists():
        raise FileNotFoundError(
            f"Could not find {sample_dir}. Run sample_video_frames.py first."
        )

    metadata_path = sample_dir / "sampled_frames.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

        sampled_frames = sorted(
            metadata.get("sampled_frames", []),
            key=lambda item: int(item.get("sample_number", 0)),
        )
        image_paths = [
            resolve_metadata_image_path(root, sample_dir, frame["image"])
            for frame in sampled_frames
        ]
    else:
        image_paths = sorted(
            path for path in sample_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
        )

    missing_images = [path for path in image_paths if not path.exists()]
    if missing_images:
        raise FileNotFoundError(f"Missing sampled image files: {missing_images}")
    if not image_paths:
        raise FileNotFoundError(f"No sampled image files found in {sample_dir}")

    return image_paths


def encode_image_data_url(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/jpeg"

    with image_path.open("rb") as file:
        encoded = base64.b64encode(file.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def read_existing_prompt_augment(path: Path) -> dict[int, list[str]]:
    if not path.exists():
        return {}

    prompt_map: dict[int, list[str]] = {}
    for row in read_jsonl(path):
        task_index = int(row["task_index"])
        prompt_map[task_index] = [normalize_text(text) for text in row["task_des"]]
    return prompt_map


def build_multimodal_content(
    instruction: str,
    image_paths: list[Path],
    n_variants: int,
    image_detail: str,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": f"""
You are generating robot task prompt augmentations.

Original task instruction: {instruction!r}

The following {len(image_paths)} images are ordered frames sampled from one head-camera video of the task.
Use the images only to ground the meaning of the original task and disambiguate the objects/actions.

Generate {n_variants} different natural-language instructions that mean the same thing.

Rules:
- Keep the same physical intent as the original task and the observed frames.
- Vary wording, such as verbs, objects, prepositions, and sentence structure.
- Keep the original sentence as one candidate.
- Wrap the pick object and place target with angle brackets "<>", including the original sentence.
- Do not mention the video, frames, images, camera, robot, scene layout, colors, timestamps, or visual details unless they are essential to the task meaning.
- Do not add extra constraints, extra steps, or extra objects not implied by the task and frames.
- Return JSON only, matching the provided schema.
""".strip(),
        }
    ]

    for index, image_path in enumerate(image_paths, start=1):
        content.append(
            {
                "type": "input_text",
                "text": f"Frame {index} of {len(image_paths)}:",
            }
        )
        content.append(
            {
                "type": "input_image",
                "image_url": encode_image_data_url(image_path),
                "detail": image_detail,
            }
        )

    return content


def paraphrase_instruction_with_images(
    client: OpenAI,
    instruction: str,
    image_paths: list[Path],
    n_variants: int,
    model: str,
    temperature: float,
    image_detail: str,
) -> list[str]:
    schema = {
        "type": "object",
        "properties": {
            "variants": {
                "type": "array",
                "minItems": n_variants,
                "maxItems": n_variants,
                "items": {"type": "string"},
            }
        },
        "required": ["variants"],
        "additionalProperties": False,
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = client.responses.create(
                model=model,
                input=[
                    {
                        "role": "user",
                        "content": build_multimodal_content(
                            instruction=instruction,
                            image_paths=image_paths,
                            n_variants=n_variants,
                            image_detail=image_detail,
                        ),
                    }
                ],
                temperature=temperature,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "visual_prompt_variants",
                        "schema": schema,
                    }
                },
            )
            data = json.loads(response.output_text)

            variants: list[str] = []
            seen = {instruction}
            for variant in data["variants"]:
                variant = normalize_text(variant)
                if variant and variant not in seen:
                    variants.append(variant)
                    seen.add(variant)

            if len(variants) == n_variants:
                return variants
        except Exception:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2**attempt)

    raise RuntimeError(f"Could not generate {n_variants} variants for {instruction!r}")


def build_record(
    task_index: int,
    task: str,
    variants: list[str],
) -> dict[str, Any]:
    return {
        "task_index": task_index,
        "task_des": variants,
    }


def generate_for_meta_dir(
    meta_dir: Path,
    image_paths: list[Path],
    client: OpenAI,
    n_variants: int,
    model: str,
    temperature: float,
    image_detail: str,
    output_name: str,
    overwrite: bool,
) -> int:
    task_rows = read_jsonl(meta_dir / "tasks.jsonl")
    output_path = meta_dir / output_name
    generated_count = 0

    if overwrite:
        records = []
        existing_prompt_map: dict[int, list[str]] = {}
    else:
        records = []
        existing_prompt_map = read_existing_prompt_augment(output_path)

    for task_row in task_rows:
        task_index = int(task_row["task_index"])
        task = normalize_text(task_row["task"])

        if task_index in existing_prompt_map:
            continue

        variants = paraphrase_instruction_with_images(
            client=client,
            instruction=task,
            image_paths=image_paths,
            n_variants=n_variants,
            model=model,
            temperature=temperature,
            image_detail=image_detail,
        )
        record = build_record(
            task_index=task_index,
            task=task,
            variants=variants,
        )

        if overwrite:
            records.append(record)
        else:
            append_jsonl(output_path, record)

        generated_count += 1

    if overwrite:
        write_jsonl(output_path, records)

    return generated_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate visually grounded prompt augmentations from meta/tasks.jsonl "
            "and scenario_x_sampled_image/date frames."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Repo/data root containing scenario folders and sampled-image folders.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=list(DEFAULT_SCENARIOS),
        help="Scenario folders to process.",
    )
    parser.add_argument(
        "--dates",
        nargs="*",
        help="Optional date folder names to process, such as 2025.12.01 2026.01.15.",
    )
    parser.add_argument(
        "--n-variants",
        type=int,
        default=DEFAULT_N_VARIANTS,
        help="Number of visual-text paraphrases to generate for each task.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument(
        "--image-detail",
        choices=("low", "high", "auto"),
        default=DEFAULT_IMAGE_DETAIL,
        help="Vision detail level for sampled frames.",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help="Output JSONL filename inside each meta folder.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate and replace the output file for selected meta folders.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)
    dates = set(args.dates) if args.dates else None
    meta_dirs = iter_meta_dirs(args.root, tuple(args.scenarios), dates)
    generated_count = 0

    for scenario_name, date_name, meta_dir in meta_dirs:
        image_paths = read_sampled_images(
            root=args.root,
            scenario_name=scenario_name,
            date_name=date_name,
        )
        count = generate_for_meta_dir(
            meta_dir=meta_dir,
            image_paths=image_paths,
            client=client,
            n_variants=args.n_variants,
            model=args.model,
            temperature=args.temperature,
            image_detail=args.image_detail,
            output_name=args.output_name,
            overwrite=args.overwrite,
        )
        generated_count += count
        print(
            f"Processed {meta_dir} with {len(image_paths)} images: "
            f"generated {count} task prompt records"
        )

    print(
        f"Completed {len(meta_dirs)} meta folders. "
        f"Generated {generated_count} visual prompt records."
    )


if __name__ == "__main__":
    main()
