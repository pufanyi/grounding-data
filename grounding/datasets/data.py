import json
import os
import re
from collections.abc import Iterable
from pathlib import Path

from loguru import logger
from pydantic import BaseModel
from tqdm import tqdm

_REF_PATTERN = re.compile(
    r"<ref>(?P<label>.+?)</ref>\s*(?P<bboxes>\[\[.*?\]\])", re.DOTALL
)


class GroundingData(BaseModel):
    source_dataset: str
    source_id: str
    image: str
    objs: dict[str, list[list[float]]]
    num_objs: int
    num_bbox: int


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
                self.data.extend(
                    [{"data": json.loads(line), "source": data_path.name} for line in f]
                )

        if self.max_items is not None:
            self.data = self.data[: self.max_items]

    def parse_item(self, item: dict) -> GroundingData:
        assert len(item["data"]["image"]) == 1, "Only one image is supported"
        image = item["data"]["image"][0]
        objs: dict[str, list[list[float]]] = {}
        for conv in item["data"]["conversations"]:
            if conv["from"] == "gpt":
                conv_str = conv["value"]
                for match in _REF_PATTERN.finditer(conv_str):
                    label = match.group("label").strip().lower()
                    bboxes = json.loads(match.group("bboxes"))
                    if label not in objs:
                        objs[label] = []
                    for bbox in bboxes:
                        objs[label].append([float(coord) for coord in bbox])
        return GroundingData.model_validate(
            {
                "image": image,
                "objs": objs,
                "source_dataset": item["source"],
                "source_id": str(item["data"]["id"]),
                "num_objs": len(objs),
                "num_bbox": sum(len(bboxes) for bboxes in objs.values()),
            }
        )

    def parse_data(self) -> Iterable[GroundingData]:
        for item in tqdm(self.data):
            result = self.parse_item(item)
            if result.num_objs == 0:
                logger.warning(
                    f"No objects found in item"
                    f" {item['data']['id']} from {item['source']},"
                    f" the conversation is {item['data']['conversations']}"
                )
                continue
            yield result


if __name__ == "__main__":
    dataset = GroundingDataset(
        "/mnt/aigc/users/pufanyi/workspace/playground/grouding/data/jsonl",
    )
    with open("data/result_dataset.jsonl", "w") as f:
        for data in dataset.parse_data():
            f.write(data.model_dump_json() + "\n")
