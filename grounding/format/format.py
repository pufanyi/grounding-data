from abc import ABC, abstractmethod

from pydantic import BaseModel

from ..datasets.data import GroundingData


class SITEData(BaseModel):
    source_dataset: str
    source_id: str
    image: str
    conversation: list[dict]
    question_type: str


class Format(ABC):
    @abstractmethod
    def format(self, data: GroundingData) -> SITEData:
        raise NotImplementedError
