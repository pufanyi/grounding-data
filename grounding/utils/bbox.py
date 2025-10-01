import random


def random_bbox(original_bbox: list[float]) -> list[float]:
    if random.random() < 0.9:
        bbox = [
            original_bbox[0] + random.uniform(-0.1, 0.1),
            original_bbox[1] + random.uniform(-0.1, 0.1),
            original_bbox[2] + random.uniform(-0.1, 0.1),
            original_bbox[3] + random.uniform(-0.1, 0.1),
        ]
    else:
        bbox = [
            random.uniform(0, 1),
            random.uniform(0, 1),
            random.uniform(0, 1),
            random.uniform(0, 1),
        ]
    bbox = [max(0, min(1, coord)) for coord in bbox]
    if bbox[0] > bbox[2]:
        bbox[0], bbox[2] = bbox[2], bbox[0]
    if bbox[1] > bbox[3]:
        bbox[1], bbox[3] = bbox[3], bbox[1]
    return bbox
