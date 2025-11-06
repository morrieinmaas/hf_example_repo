from typing import Any, List, NamedTuple
import numpy as np

# Type aliases
NDIntArray = np.ndarray  # type: ignore
NDFloatArray = np.ndarray  # type: ignore


# Simple placeholder for GenerateOutput
class GenerateOutput(NamedTuple):  # type: ignore
    output_ids: Any
    log_probs: Any
    tokenwise_log_probs: Any
    output_strings: List[str]


# Simple placeholder for TopKResult
class TopKResult(NamedTuple):  # type: ignore
    indices: List[int]
    probabilities: List[float]
