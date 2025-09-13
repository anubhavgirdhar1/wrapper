from abc import ABC, abstractmethod

class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass