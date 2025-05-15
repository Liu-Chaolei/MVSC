# -*- coding: utf-8 -*-


import logging

LOG_FORMAT = "%(asctime)s [%(levelname)s] [%(name)s:%(funcName)s:%(lineno)d] %(message)s"

_MAX_PARTS = 2


def set_log_level(verbosity: int) -> int:
    if verbosity == 1:
        return logging.INFO
    if verbosity > 1:
        return logging.DEBUG
    return logging.WARNING


def get_logger(module: str) -> logging.Logger:
    parts = module.split(".")
    name = module
    if len(parts) > _MAX_PARTS:
        name = ".".join(parts[:2])
    return logging.getLogger(name)
