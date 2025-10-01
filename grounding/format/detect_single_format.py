import random

from ..datasets.data import GroundingData
from .format import SITEData

TEMPLATE = """\
Please detect the {obj_name} in this image and represent them \
using a single bounding box. Please provide the bounding box coordinates for \
the described object or area using the format [x1, y1, x2, y2]. Here, [x1, y1] \
represent the top-left coordinates and [x2, y2] the bottom-right coordinates \
within a normalized range of 0 to 1, where [0, 0] is the top-left corner and \
[1, 1] is the bottom-right corner of the image.\
"""


class DetectSingleFormat:
    def __init__(self):
        super().__init__("detect_single")

    def check_eligible(self, data: GroundingData) -> bool:
        for obj in data.objs.values():
            if len(obj) == 1:
                return True
        return False

    def format(self, data: GroundingData) -> SITEData:
        all_single_objs = []
        for key, bboxes in data.objs.items():
            if len(bboxes) == 1:
                all_single_objs.append((key, bboxes[0]))
        obj_name, bbox = random.choice(all_single_objs)
        input_text = TEMPLATE.format(obj_name=obj_name)
        choices = [str(bbox)]
        return input_text, choices
