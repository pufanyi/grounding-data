from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, Iterator, Sequence, Tuple

DATASET_NAMES = [
    "detect_single",
    "detect_multi",
    "detect_multi_object",
]

OPTION_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _load_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            yield json.loads(stripped)


def _ensure_list(image_field: str | Sequence[str]) -> list[str]:
    if isinstance(image_field, str):
        return [image_field]
    return list(image_field)


def _shuffle_choices(
    choices: Sequence[str], rng: random.Random
) -> Tuple[list[Tuple[str, str]], str]:
    if len(choices) > len(OPTION_LETTERS):
        raise ValueError("Too many choices to assign unique letters")

    indexed = list(enumerate(choices))
    rng.shuffle(indexed)

    option_pairs: list[Tuple[str, str]] = []
    correct_letter: str | None = None

    for position, (original_index, choice_text) in enumerate(indexed):
        letter = OPTION_LETTERS[position]
        option_pairs.append((letter, choice_text))
        if original_index == 0:
            correct_letter = letter

    if correct_letter is None:
        raise ValueError("Unable to determine correct answer after shuffling")

    return option_pairs, correct_letter


def _letter_instruction(option_pairs: Sequence[Tuple[str, str]]) -> str:
    letters = "/".join(letter for letter, _ in option_pairs[:4])
    return "A/B/C/D" if len(option_pairs) == 4 else letters


def _template_block(question: str, option_pairs: Sequence[Tuple[str, str]]) -> str:
    instruction = _letter_instruction(option_pairs)
    lines = [
        question.rstrip(),
        "",
        "Options:",
    ]
    lines.extend(f"{letter}. {text}" for letter, text in option_pairs)
    lines.extend(
        [
            "",
            f"Please answer the question with a single {instruction} directly.",
        ]
    )
    return "\n".join(lines)


def _template_inline(question: str, option_pairs: Sequence[Tuple[str, str]]) -> str:
    instruction = _letter_instruction(option_pairs)
    lines = [
        question.rstrip(),
        "",
        f"Choose the best option and respond with a single letter ({instruction}).",
        f"Please answer the question with a single {instruction} directly.",
        "",
        "Options:",
    ]
    lines.extend(f"{letter}) {text}" for letter, text in option_pairs)
    return "\n".join(lines)


def _template_steps(question: str, option_pairs: Sequence[Tuple[str, str]]) -> str:
    instruction = _letter_instruction(option_pairs)
    lines = [
        question.rstrip(),
        "",
        "Consider these candidates:",
    ]
    lines.extend(f"- Option {letter}: {text}" for letter, text in option_pairs)
    lines.extend(
        [
            "",
            f"Please answer the question with a single {instruction} directly with no extra text.",
        ]
    )
    return "\n".join(lines)


TEMPLATES = [_template_block, _template_inline, _template_steps]


def _convert_record(record: dict, rng: random.Random) -> dict:
    choices = record.get("choices")
    if not choices:
        raise ValueError("Record is missing 'choices'")
    if len(choices) < 2:
        raise ValueError("Need at least two choices to build a multiple-choice prompt")

    option_pairs, correct_letter = _shuffle_choices(choices, rng)
    question = record.get("question", "").strip()

    template = rng.choice(TEMPLATES)
    human_prompt = template(question, option_pairs)

    raw_image = record.get("image", [])
    image_list = _ensure_list(raw_image)

    raw_id = record.get("source_id")
    converted_id: int | str
    if isinstance(raw_id, int):
        converted_id = raw_id
    else:
        try:
            converted_id = int(str(raw_id))
        except (TypeError, ValueError):
            converted_id = str(raw_id)

    return {
        "image": image_list,
        "id": converted_id,
        "conversations": [
            {"from": "human", "value": human_prompt},
            {"from": "gpt", "value": correct_letter},
        ],
        "image_count": len(image_list),
    }


def _convert_file(path: Path, rng: random.Random) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    output_path = path.with_name(f"{path.stem}_chat.jsonl")

    with output_path.open("w") as output_handle:
        for record in _load_jsonl(path):
            converted = _convert_record(record, rng)
            output_handle.write(json.dumps(converted) + "\n")

    return output_path


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert grounding data to chat format.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Optional random seed for deterministic shuffling.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    rng = random.Random(args.seed)

    root = Path(__file__).resolve().parents[2]
    data_dir = root / "data"

    for name in DATASET_NAMES:
        input_path = data_dir / f"{name}.jsonl"
        output_path = _convert_file(input_path, rng)
        print(f"Converted {input_path} -> {output_path}")


if __name__ == "__main__":
    main()
