from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError
