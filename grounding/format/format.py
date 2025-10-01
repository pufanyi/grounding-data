from abc import ABC, abstractmethod

from pydantic import BaseModel

from ..datasets.data import GroundingData


class SITEData(BaseModel):
    source_dataset: str
    source_id: str
    image: str
    question: str
    choices: list[str]  # The first choice is the answer
    question_type: str


class Formater(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def check_eligible(self, data: GroundingData) -> bool:
        raise NotImplementedError

    @abstractmethod
    def format(self, data: GroundingData) -> SITEData:
        raise NotImplementedError
