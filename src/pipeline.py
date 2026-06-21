"""Top-level orchestration (step 6). Ties all stages together; writes RunLog."""
from __future__ import annotations

from src.schemas import RunLog


def run() -> RunLog:
    raise NotImplementedError


if __name__ == "__main__":
    run()
