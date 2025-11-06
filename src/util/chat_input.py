from typing import Any, List
from pydantic import BaseModel


class IdsInput(BaseModel):
    input_ids: List[int] = []


class ModelInput(BaseModel):
    text: str = ""

    def tokenize(self, subject: Any) -> List[int]:
        return subject.tokenize(self.text)
