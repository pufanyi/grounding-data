import random
from collections.abc import Sequence

from ..datasets.data import GroundingData
from ..utils.bbox import random_bbox
from .formatter import Formatter, SITEData

TEMPLATE = """Please detect all instances of the following categories in this image: {categories}. For each detected object, provide the output in the format category:[x1, y1, x2, y2]. Here, [x1, y1] represent the top-left coordinates and [x2, y2] the bottom-right coordinates within a normalized range of 0 to 1, where [0, 0] is the top-left corner and [1, 1] is the bottom-right corner of the image."""


def _round_bbox(bbox: Sequence[float]) -> list[float]:
    return [round(coord, 3) for coord in bbox]


def _signature(entries: Sequence[tuple[str, Sequence[float]]]) -> tuple[tuple[str, tuple[float, float, float, float]], ...]:
    signature = []
    for name, bbox in entries:
        rounded = tuple(round(coord, 3) for coord in bbox)
        signature.append((name, rounded))
    return tuple(sorted(signature))


def _format_entries(entries: Sequence[tuple[str, Sequence[float]]]) -> str:
    parts: list[str] = []
    for name, bbox in entries:
        rounded = _round_bbox(bbox)
        parts.append(f"{name}: [{rounded[0]:.3f}, {rounded[1]:.3f}, {rounded[2]:.3f}, {rounded[3]:.3f}]")
    return ", ".join(parts)


def _clone_entries(entries: Sequence[tuple[str, Sequence[float]]]) -> list[tuple[str, list[float]]]:
    return [(name, list(bbox)) for name, bbox in entries]


class DetectMultiObjectFormatter(Formatter):
    def __init__(self) -> None:
        super().__init__("detect_multi_object")

    def check_eligible(self, data: GroundingData) -> bool:
        multi_objs = [bboxes for bboxes in data.objs.values() if len(bboxes) > 1]
        return len(multi_objs) >= 2

    def format(self, data: GroundingData) -> SITEData:
        multi_objs = {name: bboxes for name, bboxes in data.objs.items() if len(bboxes) > 1}
        target_names = list(multi_objs)

        max_categories = min(4, len(target_names))
        sample_size = random.randint(2, max_categories) if max_categories > 2 else max_categories
        chosen_names = random.sample(target_names, sample_size)

        target_entries: list[tuple[str, list[float]]] = []
        for name in chosen_names:
            bboxes = multi_objs[name]
            take = min(len(bboxes), 3)
            selected = random.sample(bboxes, take)
            for bbox in selected:
                target_entries.append((name, list(bbox)))

        prompt = TEMPLATE.format(categories=", ".join(chosen_names))

        extra_pool: list[tuple[str, list[float]]] = []
        for name, bboxes in data.objs.items():
            for bbox in bboxes:
                extra_pool.append((name, list(bbox)))

        distractors: list[list[tuple[str, list[float]]]] = []
        used_signatures = {_signature(target_entries)}

        def add_candidate(candidate: list[tuple[str, list[float]]]) -> None:
            if not candidate:
                return
            signature = _signature(candidate)
            if signature not in used_signatures:
                distractors.append(candidate)
                used_signatures.add(signature)

        attempts = 0
        while len(distractors) < 3 and attempts < 100:
            candidate = self._mutate_entries(target_entries, extra_pool)
            add_candidate(candidate)
            attempts += 1

        while len(distractors) < 3:
            candidate = self._fallback_entries(target_entries)
            add_candidate(candidate)

        choices = [_format_entries(target_entries)]
        sampled = random.sample(distractors, 3)
        choices.extend([_format_entries(entries) for entries in sampled])

        return SITEData.model_validate(
            {
                "source_dataset": data.source_dataset,
                "source_id": data.source_id,
                "image": data.image,
                "question": prompt,
                "choices": choices,
                "question_type": self.name,
            }
        )

    def _mutate_entries(
        self,
        base_entries: Sequence[tuple[str, Sequence[float]]],
        extra_pool: Sequence[tuple[str, Sequence[float]]],
    ) -> list[tuple[str, list[float]]]:
        entries = _clone_entries(base_entries)
        operations = ["perturb"]
        if len(entries) > 1:
            operations.append("drop")
            operations.append("swap")
        if extra_pool:
            operations.append("add")

        op_count = random.randint(1, 2)
        for _ in range(op_count):
            if not operations:
                break
            op = random.choice(operations)
            if op == "perturb" and entries:
                idx = random.randrange(len(entries))
                _, ref_bbox = random.choice(base_entries)
                entries[idx] = (entries[idx][0], random_bbox(list(ref_bbox)))
            elif op == "drop" and len(entries) > 1:
                idx = random.randrange(len(entries))
                entries.pop(idx)
            elif op == "swap" and len(entries) > 1:
                i, j = random.sample(range(len(entries)), 2)
                name_i, bbox_i = entries[i]
                name_j, bbox_j = entries[j]
                entries[i] = (name_j, bbox_i)
                entries[j] = (name_i, bbox_j)
            elif op == "add" and extra_pool:
                name, bbox = random.choice(extra_pool)
                entries.append((name, list(bbox)))
        random.shuffle(entries)
        return entries

    def _fallback_entries(self, base_entries: Sequence[tuple[str, Sequence[float]]]) -> list[tuple[str, list[float]]]:
        entries = _clone_entries(base_entries)
        idx = random.randrange(len(entries))
        name, bbox = entries[idx]
        entries[idx] = (name, random_bbox(list(bbox)))
        random.shuffle(entries)
        return entries
