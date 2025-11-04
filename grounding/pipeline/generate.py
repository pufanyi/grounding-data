from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from typing import NamedTuple

from ..datasets.data import GroundingData
from ..format.detect_multi_formatter import DetectMultiFormatter
from ..format.detect_multi_object_formatter import DetectMultiObjectFormatter
from ..format.detect_single_formatter import DetectSingleFormatter


class FormatterSpec(NamedTuple):
    name: str
    formatter: object
    output_path: Path


TARGET_COUNT = 1_000_000


def _root_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    root = _root_dir()
    dataset_path = root / "data" / "result_dataset.jsonl"
    formatter_specs = [
        FormatterSpec(
            "detect_single",
            DetectSingleFormatter(),
            root / "data" / "detect_single.jsonl",
        ),
        FormatterSpec(
            "detect_multi", DetectMultiFormatter(), root / "data" / "detect_multi.jsonl"
        ),
        FormatterSpec(
            "detect_multi_object",
            DetectMultiObjectFormatter(),
            root / "data" / "detect_multi_object.jsonl",
        ),
    ]

    counts: dict[str, int] = {spec.name: 0 for spec in formatter_specs}
    targets: dict[str, int] = {spec.name: TARGET_COUNT for spec in formatter_specs}
    used_items: set[str] = set()

    with ExitStack() as stack:
        handles = {
            spec.name: stack.enter_context(spec.output_path.open("w"))
            for spec in formatter_specs
        }

        with dataset_path.open("r") as source_file:
            for line in source_file:
                data = GroundingData.model_validate_json(line)
                unique_id = f"{data.source_dataset}:{data.source_id}"
                if unique_id in used_items:
                    continue

                for spec in formatter_specs:
                    if counts[spec.name] >= targets[spec.name]:
                        continue
                    formatter = spec.formatter
                    if formatter.check_eligible(data):
                        formatted = formatter.format(data)
                        handles[spec.name].write(formatted.model_dump_json() + "\n")
                        counts[spec.name] += 1
                        used_items.add(unique_id)
                        break

                if all(counts[name] >= targets[name] for name in counts):
                    break

    missing = {
        name: targets[name] - counts[name]
        for name in counts
        if counts[name] < targets[name]
    }
    if missing:
        raise RuntimeError(
            "Unable to satisfy target counts: "
            + json.dumps(missing, ensure_ascii=False)
        )

    for spec in formatter_specs:
        print(
            f"Generated {counts[spec.name]} examples "
            f"for {spec.name} -> {spec.output_path}"
        )


if __name__ == "__main__":
    main()
