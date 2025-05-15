import torch
from diffusers import DiffusionPipeline
from cogkit.utils import load_lora_checkpoint


def load_pipeline(
    model_id_or_path: str,
    lora_model_id_or_path: str | None = None,
    transformer_path: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> DiffusionPipeline:
    pipeline = DiffusionPipeline.from_pretrained(model_id_or_path, torch_dtype=dtype)
    if transformer_path is not None:
        pipeline.transformer.save_config(transformer_path)
        pipeline.transformer = pipeline.transformer.from_pretrained(transformer_path)
    if lora_model_id_or_path is not None:
        load_lora_checkpoint(pipeline, lora_model_id_or_path)
    return pipeline
