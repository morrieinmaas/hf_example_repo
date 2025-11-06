from typing import Any, Dict
from pydantic import BaseModel


class ModelActivations(BaseModel):
    layers: Dict[int, Dict[str, Any]] = {}
    unembed_in_BTD: Any = None
    unembed_out_BTV: Any = None
