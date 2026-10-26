"""Console UTF-8 (tránh lỗi in tiếng Việt khi stdout bị chuyển hướng trên Windows) + logging."""
from __future__ import annotations

import logging
import sys


def setup_console(level: int = logging.INFO) -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
