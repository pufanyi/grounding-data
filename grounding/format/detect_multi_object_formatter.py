import random
from collections.abc import Sequence

from ..datasets.data import GroundingData
from .formatter import Formatter, SITEData

TEMPLATE = """\
Please detect all instances of the following categories in this image: \
{categories}. For each detected object, provide the output in the format \
category:[x1, y1, x2, y2]. Here, [x1, y1] represent the top-left coordinates \
and [x2, y2] the bottom-right coordinates within a normalized range of 0 to 1, \
where [0, 0] is the top-left corner and [1, 1] is the bottom-right corner of \
the image.\
"""


def _round_bbox(bbox: Sequence[float]) -> list[float]:
    return [round(float(coord), 3) for coord in bbox]


def _signature(
    entries: Sequence[tuple[str, Sequence[float]]],
) -> tuple[tuple[str, tuple[float, float, float, float]], ...]:
    signature = []
    for name, bbox in entries:
        rounded = tuple(_round_bbox(bbox))
        signature.append((name, tuple(rounded)))
    return tuple(sorted(signature))


def _format_entries(entries: Sequence[tuple[str, Sequence[float]]]) -> str:
    parts: list[str] = []
    for name, bbox in entries:
        rounded = _round_bbox(bbox)
        parts.append(
            f"{name}: [{rounded[0]:.3f}, {rounded[1]:.3f}, "
            f"{rounded[2]:.3f}, {rounded[3]:.3f}]"
        )
    return ", ".join(parts)


def _clone_entries(
    entries: Sequence[tuple[str, Sequence[float]]],
) -> list[tuple[str, list[float]]]:
    return [(name, list(bbox)) for name, bbox in entries]


def _jitter_bbox(bbox: Sequence[float], max_delta: float = 0.05) -> list[float]:
    x1, y1, x2, y2 = map(float, bbox)
    dx1 = random.uniform(-max_delta, max_delta)
    dy1 = random.uniform(-max_delta, max_delta)
    dx2 = random.uniform(-max_delta, max_delta)
    dy2 = random.uniform(-max_delta, max_delta)
    nx1 = max(0.0, min(1.0, x1 + dx1))
    ny1 = max(0.0, min(1.0, y1 + dy1))
    nx2 = max(0.0, min(1.0, x2 + dx2))
    ny2 = max(0.0, min(1.0, y2 + dy2))
    if nx1 > nx2:
        nx1, nx2 = nx2, nx1
    if ny1 > ny2:
        ny1, ny2 = ny2, ny1
    return _round_bbox((nx1, ny1, nx2, ny2))


def _aggressive_variant(
    entries: Sequence[tuple[str, Sequence[float]]], max_delta: float
) -> list[tuple[str, list[float]]]:
    variant = _clone_entries(entries)
    for idx, (name, bbox) in enumerate(variant):
        variant[idx] = (name, _jitter_bbox(bbox, max_delta=max_delta))
    random.shuffle(variant)
    return variant


def _label_swap_variant(
    entries: Sequence[tuple[str, Sequence[float]]],
) -> list[tuple[str, list[float]]]:
    """
    Generate a variant by swapping labels between two bboxes with different categories.
    """
    variant = _clone_entries(entries)
    if len(variant) < 2:
        return variant

    # Find pairs with different categories
    different_category_pairs = []
    for i in range(len(variant)):
        for j in range(i + 1, len(variant)):
            if variant[i][0] != variant[j][0]:  # Different category names
                different_category_pairs.append((i, j))

    if not different_category_pairs:
        # If all entries have the same category, just shuffle
        random.shuffle(variant)
        return variant

    # Swap labels for 1-2 random pairs
    num_swaps = min(random.randint(1, 2), len(different_category_pairs))
    swap_pairs = random.sample(different_category_pairs, num_swaps)

    for i, j in swap_pairs:
        # Swap the category names while keeping bboxes
        name_i, bbox_i = variant[i]
        name_j, bbox_j = variant[j]
        variant[i] = (name_j, bbox_i)
        variant[j] = (name_i, bbox_j)

    random.shuffle(variant)
    return variant


class DetectMultiObjectFormatter(Formatter):
    def __init__(self) -> None:
        super().__init__("detect_multi_object")

    def check_eligible(self, data: GroundingData) -> bool:
        return data.num_objs >= 2

    def format(self, data: GroundingData) -> SITEData:
        candidate_objs = {name: bboxes for name, bboxes in data.objs.items() if bboxes}
        target_names = list(candidate_objs)

        max_categories = min(6, len(target_names))
        sample_size = (
            random.randint(2, max_categories) if max_categories > 2 else max_categories
        )
        chosen_names = random.sample(target_names, sample_size)

        target_entries: list[tuple[str, list[float]]] = []
        for name in chosen_names:
            bboxes = candidate_objs[name]
            take = min(len(bboxes), 3)
            selected = random.sample(bboxes, take)
            for bbox in selected:
                target_entries.append((name, list(bbox)))

        prompt = TEMPLATE.format(categories=", ".join(chosen_names))

        category_pool: dict[str, list[list[float]]] = {
            name: [list(bbox) for bbox in bboxes]
            for name, bboxes in data.objs.items()
            if bboxes
        }
        extra_pool: list[tuple[str, list[float]]] = [
            (name, bbox) for name, boxes in category_pool.items() for bbox in boxes
        ]

        # Strategy: 80% label swap, 20% mutation
        # Each distractor is generated using random.choice based on these probabilities
        distractors: list[list[tuple[str, list[float]]]] = []
        used_signatures = {_signature(target_entries)}
        target_len = len(target_entries)

        def add_candidate(candidate: list[tuple[str, list[float]]]) -> bool:
            if len(candidate) != target_len:
                return False
            signature = _signature(candidate)
            if signature not in used_signatures:
                distractors.append(candidate)
                used_signatures.add(signature)
                return True
            return False

        # Generate 3 distractors using weighted random choice (80% swap, 20% mutate)
        attempts = 0
        while len(distractors) < 3 and attempts < 300:
            # 80% probability for label swap, 20% for mutation
            if random.random() < 0.8:
                candidate = _label_swap_variant(target_entries)
            else:
                candidate = self._mutate_entries(
                    target_entries, category_pool, extra_pool
                )
            add_candidate(candidate)
            attempts += 1

        # Additional fallback with gentle mutation
        gentle_attempts = 0
        while len(distractors) < 3 and gentle_attempts < 100:
            candidate = self._fallback_entries(target_entries, category_pool)
            if len(candidate) == target_len:
                signature = _signature(candidate)
                if signature not in used_signatures:
                    distractors.append(candidate)
                    used_signatures.add(signature)
            gentle_attempts += 1

        force_delta = 0.12
        while len(distractors) < 3 and force_delta <= 0.24:
            candidate = _aggressive_variant(target_entries, max_delta=force_delta)
            if len(candidate) == target_len:
                signature = _signature(candidate)
                if signature not in used_signatures:
                    distractors.append(candidate)
                    used_signatures.add(signature)
            force_delta += 0.04

        heavy_attempts = 0
        while len(distractors) < 3 and heavy_attempts < 20:
            candidate = _aggressive_variant(target_entries, max_delta=0.3)
            signature = _signature(candidate)
            if signature not in used_signatures:
                distractors.append(candidate)
                used_signatures.add(signature)
            heavy_attempts += 1

        if not distractors:
            candidate = _aggressive_variant(target_entries, max_delta=0.3)
            distractors.append(candidate)
            used_signatures.add(_signature(candidate))

        selected_distractors = (
            random.sample(distractors, 3) if len(distractors) >= 3 else distractors[:]
        )
        selected_signatures = {_signature(entries) for entries in selected_distractors}
        selected_signatures.add(_signature(target_entries))

        fill_attempts = 0
        while len(selected_distractors) < 3 and fill_attempts < 50:
            generated = _aggressive_variant(target_entries, max_delta=0.3)
            signature = _signature(generated)
            if signature in selected_signatures:
                fill_attempts += 1
                continue
            selected_distractors.append(generated)
            selected_signatures.add(signature)

        while len(selected_distractors) < 3:
            generated = _aggressive_variant(target_entries, max_delta=0.35)
            selected_distractors.append(generated)
            selected_signatures.add(_signature(generated))

        choices = [_format_entries(target_entries)]
        choices.extend(_format_entries(entries) for entries in selected_distractors[:3])

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
        category_pool: dict[str, list[Sequence[float]]],
        extra_pool: Sequence[tuple[str, Sequence[float]]],
    ) -> list[tuple[str, list[float]]]:
        entries = _clone_entries(base_entries)
        target_len = len(entries)
        if target_len == 0:
            return entries
        change_fraction = random.uniform(0.3, 0.6)
        change_count = max(1, min(target_len, int(round(change_fraction * target_len))))
        for idx in random.sample(range(target_len), change_count):
            name, bbox = entries[idx]
            action = random.random()
            pool = category_pool.get(name, [])
            alternatives = [candidate for candidate in pool if candidate != bbox]
            if action < 0.5 and alternatives:
                entries[idx] = (name, list(random.choice(alternatives)))
                continue
            if action < 0.85:
                entries[idx] = (name, _jitter_bbox(bbox))
            else:
                if extra_pool:
                    alt_name, alt_bbox = random.choice(extra_pool)
                    if alt_name == name and alternatives:
                        entries[idx] = (name, list(random.choice(alternatives)))
                    else:
                        entries[idx] = (alt_name, _jitter_bbox(alt_bbox))
                else:
                    entries[idx] = (name, _jitter_bbox(bbox))
        if target_len > 1 and random.random() < 0.4:
            i, j = random.sample(range(target_len), 2)
            entries[i], entries[j] = entries[j], entries[i]
        random.shuffle(entries)
        return entries

    def _fallback_entries(
        self,
        base_entries: Sequence[tuple[str, Sequence[float]]],
        category_pool: dict[str, list[Sequence[float]]],
    ) -> list[tuple[str, list[float]]]:
        entries = _clone_entries(base_entries)
        if not entries:
            return entries
        idx = random.randrange(len(entries))
        name, bbox = entries[idx]
        pool = category_pool.get(name, [])
        alternatives = [candidate for candidate in pool if candidate != bbox]
        if alternatives and random.random() < 0.7:
            entries[idx] = (name, list(random.choice(alternatives)))
        else:
            entries[idx] = (name, _jitter_bbox(bbox, max_delta=0.08))
        if len(entries) > 1 and random.random() < 0.3:
            j = random.randrange(len(entries))
            if j != idx:
                entries[idx], entries[j] = entries[j], entries[idx]
        random.shuffle(entries)
        return entries
