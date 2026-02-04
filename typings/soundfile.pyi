from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from typing import IO

import numpy as np

def write(
    file: str | PathLike[str] | PathLike[bytes] | IO[bytes],
    data: np.ndarray | Sequence[float] | Sequence[Sequence[float]],
    samplerate: int,
    format: str | None = ...,
    subtype: str | None = ...,
) -> None: ...
