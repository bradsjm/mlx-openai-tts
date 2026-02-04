from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

import numpy as np

class TTSResult(Protocol):
    audio: np.ndarray

class TTSModel(Protocol):
    sample_rate: int | float | None

    def generate(self, *, text: str, voice: str, speed: float) -> Iterable[TTSResult]: ...

def load_model(model_path: str, *, strict: bool = ...) -> TTSModel: ...
