"""Add canonical object/target tags to RobotWin task text.

The original metadata files are never modified. By default this writes:

    meta/tasks_tagged.jsonl
    meta/episodes_tagged.jsonl
    meta/task_tagging_report.json

Only the first occurrence of the movable object and the first occurrence of
the basket are wrapped with ``<>`` in each task sentence.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


CARD_PATTERNS = [
    r"\bplayingcards?\b",
    r"\bplaying\s+cards?\b",
    r"\b(?:box|pack|container|carton|case|holder)\s+with\s+cards?\b",
    r"\b(?:box|pack|container|carton|case|holder)\s+(?:for|holding)\s+(?:playing\s*)?cards?\b",
    r"\bcards?\s+(?:inside|box|storage|pack|container|case)\b",
    r"\bcard\s+storage\b",
]

TOYCAR_PATTERNS = [
    r"\btoycar\b",
    r"\btoy\s+car\b",
    r"\b(?:pink|blue|green|black|plastic|small|lightweight|rolling|miniature)\s+car\b",
    r"\bcar\s+with\b",
    r"\bcar\s+for\s+kids\b",
]

BASKET_PATTERN = r"\bbasket\b"


def compile_patterns(patterns: list[str]) -> re.Pattern[str]:
    return re.compile("(?:" + "|".join(patterns) + ")", re.IGNORECASE)


CARD_RE = compile_patterns(CARD_PATTERNS)
TOYCAR_RE = compile_patterns(TOYCAR_PATTERNS)
BASKET_RE = re.compile(BASKET_PATTERN, re.IGNORECASE)


def tag_first(text: str, pattern: re.Pattern[str], tag: str) -> tuple[str, bool]:
    """Replace only the first matched phrase with one canonical tag."""
    tagged, count = pattern.subn(tag, text, count=1)
    return tagged, count > 0


def normalize_task(text: str) -> tuple[str, dict[str, Any]]:
    """Tag one task and return the transformed text plus an audit record."""
    card_match = CARD_RE.search(text)
    toycar_match = TOYCAR_RE.search(text)
    object_matches = [
        ("playingcards", card_match),
        ("toycar", toycar_match),
    ]
    found_objects = [name for name, match in object_matches if match is not None]

    result = text
    object_name = None
    object_match = None
    if len(found_objects) == 1:
        object_name = found_objects[0]
        object_match = dict(object_matches)[object_name]
        object_pattern = CARD_RE if object_name == "playingcards" else TOYCAR_RE
        result, _ = tag_first(result, object_pattern, f"<{object_name}>")

    result, basket_found = tag_first(result, BASKET_RE, "<basket>")
    status = "ok"
    if len(found_objects) == 0:
        status = "missing_object"
    elif len(found_objects) > 1:
        status = "ambiguous_object"
    elif not basket_found:
        status = "missing_basket"

    return result, {
        "status": status,
        "object": object_name,
        "object_match": object_match.group(0) if object_match else None,
        "basket_found": basket_found,
        "source": text,
        "normalized": result,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_tasks(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output = []
    report = []
    for row in rows:
        updated = dict(row)
        normalized, audit = normalize_task(str(row["task"]))
        updated["task"] = normalized
        output.append(updated)
        report.append({"file": "tasks.jsonl", "task_index": row.get("task_index"), **audit})
    return output, report


def normalize_episodes(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output = []
    report = []
    for row in rows:
        updated = dict(row)
        tasks = row.get("tasks")
        if not isinstance(tasks, list):
            output.append(updated)
            report.append(
                {
                    "file": "episodes.jsonl",
                    "episode_index": row.get("episode_index"),
                    "status": "invalid_tasks_field",
                }
            )
            continue

        normalized_tasks = []
        for task in tasks:
            normalized, audit = normalize_task(str(task))
            normalized_tasks.append(normalized)
            report.append(
                {
                    "file": "episodes.jsonl",
                    "episode_index": row.get("episode_index"),
                    **audit,
                }
            )
        updated["tasks"] = normalized_tasks
        output.append(updated)
    return output, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create tagged RobotWin task metadata without changing originals."
    )
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--tasks-output", default="tasks_tagged.jsonl")
    parser.add_argument("--episodes-output", default="episodes_tagged.jsonl")
    parser.add_argument("--report-output", default="task_tagging_report.json")
    parser.add_argument(
        "--overwrite-generated",
        action="store_true",
        help="Allow replacing the generated output files, never the originals.",
    )
    args = parser.parse_args()

    meta_dir = args.dataset_path.expanduser().resolve() / "meta"
    tasks_path = meta_dir / "tasks.jsonl"
    episodes_path = meta_dir / "episodes.jsonl"
    output_paths = [
        meta_dir / args.tasks_output,
        meta_dir / args.episodes_output,
        meta_dir / args.report_output,
    ]
    if not tasks_path.exists() or not episodes_path.exists():
        raise FileNotFoundError(f"Expected tasks.jsonl and episodes.jsonl in {meta_dir}")
    if not args.overwrite_generated:
        existing = [path for path in output_paths if path.exists()]
        if existing:
            raise FileExistsError(
                "Generated output already exists; use --overwrite-generated: "
                + ", ".join(str(path) for path in existing)
            )

    tagged_tasks, task_report = normalize_tasks(read_jsonl(tasks_path))
    tagged_episodes, episode_report = normalize_episodes(read_jsonl(episodes_path))
    report = task_report + episode_report

    write_jsonl(output_paths[0], tagged_tasks)
    write_jsonl(output_paths[1], tagged_episodes)
    output_paths[2].write_text(
        json.dumps(
            {
                "dataset_path": str(args.dataset_path.expanduser().resolve()),
                "patterns": {
                    "cards": CARD_PATTERNS,
                    "toycar": TOYCAR_PATTERNS,
                    "basket": BASKET_PATTERN,
                },
                "summary": {
                    "records": len(report),
                    "ok": sum(item.get("status") == "ok" for item in report),
                    "non_ok": sum(item.get("status") != "ok" for item in report),
                },
                "records": report,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )

    non_ok = [item for item in report if item.get("status") != "ok"]
    print(f"Wrote {output_paths[0]}")
    print(f"Wrote {output_paths[1]}")
    print(f"Wrote {output_paths[2]}")
    print(f"Records: {len(report)}, ok: {len(report) - len(non_ok)}, non-ok: {len(non_ok)}")
    for item in non_ok:
        print(
            f"[{item.get('status')}] "
            f"{item.get('file')} index="
            f"{item.get('task_index', item.get('episode_index'))}: "
            f"{item.get('source', '')}"
        )


if __name__ == "__main__":
    main()
