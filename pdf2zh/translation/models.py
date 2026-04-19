from __future__ import annotations

from typing import NamedTuple


class Task(NamedTuple):
    target: dict  # element or cell dict to write into
    write_key: str
    text: str
    id: str
