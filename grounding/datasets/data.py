import json
import os
import re
from collections.abc import Iterable
from pathlib import Path

from loguru import logger
from pydantic import BaseModel
from tqdm import tqdm


class BBoxObj(BaseModel):
    bbox: list[float]
    label: str


_REF_PATTERN = re.compile(
    r"<ref>(?P<label>.+?)</ref>\s*(?P<bboxes>\[\[.*?\]\])", re.DOTALL
)


class GroundingData(BaseModel):
    image: str
    objs: list[BBoxObj]


class GroundingDataset:
    def __init__(
        self,
        data_path: str | os.PathLike | list[str | os.PathLike],
        max_items: int | None = None,
    ):
        if isinstance(data_path, str | os.PathLike):
            data_path = Path(data_path)
            if data_path.is_file():
                self.data_paths = [data_path]
            else:
                self.data_paths = [Path(path) for path in data_path.glob("*.jsonl")]
        else:
            self.data_paths = [Path(path) for path in data_path]

        self.max_items = max_items
        self.data = []
        for data_path in tqdm(self.data_paths, desc="Loading data"):
            with data_path.open("r") as f:
                self.data.extend([json.loads(line) for line in f])

        if self.max_items is not None:
            self.data = self.data[: self.max_items]

    def parse_item(self, item: dict) -> GroundingData:
        assert len(item["image"]) == 1, "Only one image is supported"
        image = item["image"][0]
        objs: list[BBoxObj] = []
        for conv in item["conversations"]:
            if conv["from"] == "gpt":
                conv_str = conv["value"]
                for match in _REF_PATTERN.finditer(conv_str):
                    label = match.group("label").strip()
                    bboxes = json.loads(match.group("bboxes"))
                    for bbox in bboxes:
                        objs.append(
                            BBoxObj(bbox=[float(coord) for coord in bbox], label=label)
                        )
        return GroundingData(image=image, objs=objs)

    def parse_data(self) -> Iterable[GroundingData]:
        for item in tqdm(self.data):
            result = self.parse_item(item)
            if len(result.objs) == 0:
                logger.warning(
                    f"No objects found in item {item['id']}, "
                    f"the conversation is {item['conversations']}"
                )
                continue
            yield result


if __name__ == "__main__":
    dataset = GroundingDataset(
        "/mnt/aigc/users/pufanyi/workspace/playground/grouding/data/jsonl",
        # max_items=100,
    )
    for data in dataset.parse_data():
        # logger.info(data)
        pass
