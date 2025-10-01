from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import NamedTuple

DATASET_NAMES = [
    "detect_single",
    "detect_multi",
    "detect_multi_object",
]

OPTION_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


class InstructionContext(NamedTuple):
    letters: tuple[str, ...]
    descriptor: str
    lowercase_descriptor: str


class ResponseStyle(NamedTuple):
    instruction_builder: Callable[[InstructionContext], list[str]]
    answer_builder: Callable[[str], str]


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
) -> tuple[list[tuple[str, str]], str]:
    if len(choices) > len(OPTION_LETTERS):
        raise ValueError("Too many choices to assign unique letters")

    indexed = list(enumerate(choices))
    rng.shuffle(indexed)

    option_pairs: list[tuple[str, str]] = []
    correct_letter: str | None = None

    for position, (original_index, choice_text) in enumerate(indexed):
        letter = OPTION_LETTERS[position]
        option_pairs.append((letter, choice_text))
        if original_index == 0:
            correct_letter = letter

    if correct_letter is None:
        raise ValueError("Unable to determine correct answer after shuffling")

    return option_pairs, correct_letter


def _letter_instruction(option_pairs: Sequence[tuple[str, str]]) -> str:
    letters = [letter for letter, _ in option_pairs]
    if len(letters) == 4:
        return "A/B/C/D"
    return "/".join(letters)


def _instruction_context(option_pairs: Sequence[tuple[str, str]]) -> InstructionContext:
    letters = tuple(letter for letter, _ in option_pairs)
    descriptor = _letter_instruction(option_pairs)
    lowercase_descriptor = "/".join(letter.lower() for letter in letters)
    return InstructionContext(letters, descriptor, lowercase_descriptor)


def _template_block(
    question: str,
    option_pairs: Sequence[tuple[str, str]],
    instruction_lines: Sequence[str],
) -> str:
    lines = [
        question.rstrip(),
        "",
        "Options:",
    ]
    lines.extend(f"{letter}. {text}" for letter, text in option_pairs)
    if instruction_lines:
        lines.append("")
        lines.extend(instruction_lines)
    return "\n".join(lines)


def _template_inline(
    question: str,
    option_pairs: Sequence[tuple[str, str]],
    instruction_lines: Sequence[str],
) -> str:
    lines = [question.rstrip()]
    if instruction_lines:
        lines.append("")
        lines.extend(instruction_lines)
    lines.append("")
    lines.append("Options:")
    rendered_options = " ".join(f"{letter}) {text}" for letter, text in option_pairs)
    lines.append(rendered_options)
    return "\n".join(lines)


def _template_steps(
    question: str,
    option_pairs: Sequence[tuple[str, str]],
    instruction_lines: Sequence[str],
) -> str:
    lines = [
        question.rstrip(),
        "",
        "Consider these candidates:",
    ]
    lines.extend(f"- Option {letter}: {text}" for letter, text in option_pairs)
    if instruction_lines:
        lines.append("")
        lines.extend(instruction_lines)
    return "\n".join(lines)


TEMPLATES = [_template_block, _template_inline, _template_steps]


def _style_letter_only(ctx: InstructionContext) -> list[str]:
    return [
        "Choose the best option from the list.",
        f"Respond with only the letter ({ctx.descriptor}).",
    ]


def _style_answer_prefix(ctx: InstructionContext) -> list[str]:
    return [
        "Select the most accurate choice.",
        f"Reply using the exact format `Answer: X` where X is one of {ctx.descriptor}.",
        "No additional text is allowed.",
    ]


def _style_final_answer(ctx: InstructionContext) -> list[str]:
    return [
        "Pick the option that fits best.",
        f"Return it as `Final answer: X` using one of {ctx.descriptor}.",
    ]


def _style_lowercase(ctx: InstructionContext) -> list[str]:
    return [
        "Determine the correct candidate.",
        f"Respond in lowercase using only {ctx.lowercase_descriptor}.",
    ]


def _style_option_prefix(ctx: InstructionContext) -> list[str]:
    return [
        "Choose the most suitable option.",
        f"Reply as `Option X` where X is one of {ctx.descriptor}.",
        "Do not include any other words.",
    ]


RESPONSE_STYLES = [
    ResponseStyle(_style_letter_only, lambda letter: letter),
    ResponseStyle(_style_answer_prefix, lambda letter: f"Answer: {letter}"),
    ResponseStyle(_style_final_answer, lambda letter: f"Final answer: {letter}"),
    ResponseStyle(_style_lowercase, lambda letter: letter.lower()),
    ResponseStyle(_style_option_prefix, lambda letter: f"Option {letter}"),
]


def _convert_record(record: dict, rng: random.Random) -> dict:
    choices = record.get("choices")
    if not choices:
        raise ValueError("Record is missing 'choices'")
    if len(choices) < 2:
        raise ValueError("Need at least two choices to build a multiple-choice prompt")

    option_pairs, correct_letter = _shuffle_choices(choices, rng)
    question = record.get("question", "").strip()

    context = _instruction_context(option_pairs)
    style = rng.choice(RESPONSE_STYLES)
    instruction_lines = style.instruction_builder(context)
    if not instruction_lines:
        raise ValueError("Instruction builder returned no content")

    template = rng.choice(TEMPLATES)
    human_prompt = template(question, option_pairs, instruction_lines)
    assistant_value = style.answer_builder(correct_letter)

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
            {"from": "gpt", "value": assistant_value},
        ],
        "image_count": len(image_list),
    }


def _convert_file(path: Path, output_dir: Path, rng: random.Random) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    output_path = output_dir / f"{path.stem}_chat.jsonl"

    with output_path.open("w") as output_handle:
        for record in _load_jsonl(path):
            converted = _convert_record(record, rng)
            output_handle.write(json.dumps(converted) + "\n")

    return output_path


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert grounding data to chat format."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for deterministic shuffling.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    rng = random.Random(args.seed)

    root = Path(__file__).resolve().parents[2]
    data_dir = root / "data"
    output_dir = root / "final_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in DATASET_NAMES:
        input_path = data_dir / f"{name}.jsonl"
        output_path = _convert_file(input_path, output_dir, rng)
        print(f"Converted {input_path} -> {output_path}")


if __name__ == "__main__":
    main()
