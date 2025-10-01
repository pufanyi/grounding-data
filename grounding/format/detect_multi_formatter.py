import json
import random

from ..datasets.data import GroundingData
from ..utils.bbox import random_bbox
from .formatter import Formatter, SITEData

TEMPLATE = """\
Please detect all regions corresponding to the {obj_name} in this image. \
Please provide the bounding box coordinates for the described objects using \
the format [x1, y1, x2, y2]. Here, [x1, y1] represent the top-left coordinates \
and [x2, y2] the bottom-right coordinates within a normalized range of 0 to 1, \
where [0, 0] is the top-left corner and [1, 1] is the bottom-right corner of \
the image. There are {count} bounding boxes to report.\
"""


def _signature(
    bboxes: list[list[float]],
) -> tuple[tuple[float, float, float, float], ...]:
    return tuple(tuple(value for value in bbox) for bbox in bboxes)


class DetectMultiFormatter(Formatter):
    def __init__(self) -> None:
        super().__init__("detect_multi")

    def check_eligible(self, data: GroundingData) -> bool:
        if any(len(bboxes) > 1 for bboxes in data.objs.values()):
            return True
        return random.random() < 0.1

    def format(self, data: GroundingData) -> SITEData:
        multi_objs = [
            (name, bboxes) for name, bboxes in data.objs.items() if len(bboxes) > 1
        ]
        if len(multi_objs) == 0:
            obj_name, target_bboxes = random.choice(list(data.objs.items()))
        else:
            obj_name, target_bboxes = random.choice(multi_objs)
        target_len = len(target_bboxes)

        cleaned_obj_name = obj_name
        for start_str in ["a ", "an ", "the "]:
            if cleaned_obj_name.startswith(start_str):
                cleaned_obj_name = cleaned_obj_name[len(start_str) :]
                break

        prompt = TEMPLATE.format(obj_name=cleaned_obj_name, count=target_len)

        distractors: list[list[list[float]]] = []
        used_signatures = {_signature(target_bboxes)}

        def add_candidate(candidate: list[list[float]]) -> None:
            if len(candidate) != target_len:
                return
            signature = _signature(candidate)
            if signature not in used_signatures:
                distractors.append(candidate)
                used_signatures.add(signature)

        for name, bboxes in data.objs.items():
            if name == obj_name:
                continue
            add_candidate(bboxes)

        attempts = 0
        while len(distractors) < 3 and attempts < 50:
            keep_count = random.randint(1, target_len - 1) if target_len > 1 else 1
            keep_indices = random.sample(range(target_len), keep_count)
            candidate = [target_bboxes[idx] for idx in keep_indices]
            while len(candidate) < target_len:
                candidate.append(random_bbox(random.choice(target_bboxes)))
            add_candidate(candidate)
            attempts += 1

        while len(distractors) < 8:
            candidate = [
                random_bbox(random.choice(target_bboxes)) for _ in range(target_len)
            ]
            add_candidate(candidate)

        choices = [target_bboxes]
        choices.extend(random.sample(distractors, 3))
        str_choices = [json.dumps(choice) for choice in choices]

        return SITEData.model_validate(
            {
                "source_dataset": data.source_dataset,
                "source_id": data.source_id,
                "image": data.image,
                "question": prompt,
                "choices": str_choices,
                "question_type": self.name,
            }
        )
