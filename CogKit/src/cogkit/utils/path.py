# -*- coding: utf-8 -*-


import os
from pathlib import Path


def resolve_path(pth: str | Path) -> Path:
    return Path(pth).expanduser().resolve()


def mkdir(dir_pth: str | Path) -> Path:
    pth = resolve_path(dir_pth)
    if pth.is_file():
        err_msg = f"Path '{os.fspath(dir_pth)}' is a regular file."
        raise ValueError(err_msg)

    if not pth.is_dir():
        pth.mkdir(parents=True, exist_ok=False)
    return pth
