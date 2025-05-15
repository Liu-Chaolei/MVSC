# -*- coding: utf-8 -*-


from diffusers.loaders import CogVideoXLoraLoaderMixin, CogView4LoraLoaderMixin


def load_lora_checkpoint(
    pipeline: CogVideoXLoraLoaderMixin | CogView4LoraLoaderMixin,
    lora_model_id_or_path: str,
    lora_scale: float = 1.0,
) -> None:
    pipeline.load_lora_weights(lora_model_id_or_path, lora_scale=lora_scale)
    # pipeline.fuse_lora(components=["transformer"], lora_scale=lora_scale)


def unload_lora_checkpoint(
    pipeline: CogVideoXLoraLoaderMixin | CogView4LoraLoaderMixin,
) -> None:
    pipeline.unload_lora_weights()
