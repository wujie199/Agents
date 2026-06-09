# -*- coding: utf-8 -*-
"""跨进程文件锁（L1 热记忆多实例安全写）。"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def file_lock(path: Path, *, exclusive: bool = False) -> Iterator[None]:
    """POSIX fcntl 文件锁；不支持时无锁直通。"""
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = lock_path.open("a+")
    try:
        try:
            import fcntl

            mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            fcntl.flock(fh.fileno(), mode)
        except (ImportError, OSError, AttributeError):
            pass
        yield
    finally:
        try:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except (ImportError, OSError, AttributeError):
            pass
        fh.close()
        if exclusive and lock_path.exists() and lock_path.stat().st_size == 0:
            try:
                lock_path.unlink(missing_ok=True)
            except OSError:
                pass
