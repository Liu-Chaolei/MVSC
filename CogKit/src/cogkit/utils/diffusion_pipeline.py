# -*- coding: utf-8 -*-


import os
from pathlib import Path
from typing import TypedDict

from diffusers import DiffusionPipeline
from huggingface_hub import hf_hub_download

from cogkit.logging import get_logger
from cogkit.utils.path import resolve_path

_logger = get_logger(__name__)


class PipelineMeta(TypedDict):
    cls_name: str | None


def _get_config_file(model_id_or_path: str) -> Path:
    if os.path.isdir(model_id_or_path):
        return resolve_path(model_id_or_path) / DiffusionPipeline.config_name

    return resolve_path(
        hf_hub_download(model_id_or_path, DiffusionPipeline.config_name, force_download=True)
    )


def get_pipeline_meta(model_id_or_path: str) -> PipelineMeta:
    config_file = _get_config_file(model_id_or_path)
    _logger.info(
        "Found the configuration file of the `DiffusionPipeline`: %s",
        os.fspath(config_file),
    )
    config_dct = DiffusionPipeline._dict_from_json_file(config_file)
    return PipelineMeta(cls_name=config_dct.get("_class_name", None))
