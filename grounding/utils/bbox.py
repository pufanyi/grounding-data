import random

import numpy as np


def in_box(bbox: list[float], larger_bbox: list[float]) -> bool:
    return (
        bbox[0] >= larger_bbox[0]
        and bbox[1] >= larger_bbox[1]
        and bbox[2] <= larger_bbox[2]
        and bbox[3] <= larger_bbox[3]
    )


def random_bbox(original_bbox: list[float]) -> list[float]:
    probs = [0.6, 0.3, 0.1]
    probs = np.cumsum(probs)
    rand_choice = random.random()
    if rand_choice < probs[0]:
        bbox = [
            original_bbox[0] + random.uniform(-0.1, 0.1),
            original_bbox[1] + random.uniform(-0.1, 0.1),
            original_bbox[2] + random.uniform(-0.1, 0.1),
            original_bbox[3] + random.uniform(-0.1, 0.1),
        ]
    elif rand_choice < probs[1]:
        bbox = [
            original_bbox[0] + random.uniform(-0.3, 0.3),
            original_bbox[1] + random.uniform(-0.3, 0.3),
            original_bbox[2] + random.uniform(-0.3, 0.3),
            original_bbox[3] + random.uniform(-0.3, 0.3),
        ]
    else:
        bbox = [
            random.uniform(0, 1),
            random.uniform(0, 1),
            random.uniform(0, 1),
            random.uniform(0, 1),
        ]

    bbox = [max(0, min(1, round(coord, 3))) for coord in bbox]
    if bbox[0] > bbox[2]:
        bbox[0], bbox[2] = bbox[2], bbox[0]
    if bbox[1] > bbox[3]:
        bbox[1], bbox[3] = bbox[3], bbox[1]

    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    if (
        area < 0.01
        or area > 0.9
        or in_box(bbox, original_bbox)
        or in_box(original_bbox, bbox)
    ):
        return random_bbox(original_bbox)

    return bbox
