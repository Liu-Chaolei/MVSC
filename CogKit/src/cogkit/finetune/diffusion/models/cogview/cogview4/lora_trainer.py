# -*- coding: utf-8 -*-


from typing import Any, Tuple

import torch
from diffusers import (
    AutoencoderKL,
    CogView4Pipeline,
    CogView4Transformer2DModel,
    FlowMatchEulerDiscreteScheduler,
)
from PIL import Image
from transformers import AutoTokenizer, GlmForCausalLM
from typing_extensions import override

from cogkit.finetune import register
from cogkit.finetune.diffusion.schemas import DiffusionComponents
from cogkit.finetune.diffusion.trainer import DiffusionTrainer
from cogkit.finetune.utils import unwrap_model


class Cogview4Trainer(DiffusionTrainer):
    UNLOAD_LIST = ["text_encoder", "vae"]
    MAX_TTOKEN_LENGTH = 224

    @override
    def load_components(self) -> DiffusionComponents:
        components = DiffusionComponents()
        model_path = str(self.args.model_path)

        components.pipeline_cls = CogView4Pipeline

        components.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")

        components.text_encoder = GlmForCausalLM.from_pretrained(
            model_path, subfolder="text_encoder"
        )

        components.transformer = CogView4Transformer2DModel.from_pretrained(
            model_path, subfolder="transformer"
        )

        components.vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")

        components.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )
        return components

    @override
    def initialize_pipeline(self) -> CogView4Pipeline:
        pipe = CogView4Pipeline(
            tokenizer=self.components.tokenizer,
            text_encoder=self.components.text_encoder,
            vae=self.components.vae,
            transformer=unwrap_model(self.accelerator, self.components.transformer),
            scheduler=self.components.scheduler,
        )
        return pipe

    @override
    def encode_text(self, prompt: str) -> torch.Tensor:
        prompt_token_ids = self.components.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.MAX_TTOKEN_LENGTH,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).input_ids
        prompt_embedding = self.components.text_encoder(
            prompt_token_ids.to(self.accelerator.device), output_hidden_states=True
        ).hidden_states[-2][0]
        # shape of prompt_embedding: [sequence length(self.MAX_TTOKEN_LENGTH), embedding dimension(4096)]
        return prompt_embedding

    @override
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        vae = self.components.vae
        image = image.to(vae.device, dtype=vae.dtype)
        latent_dist = vae.encode(image).latent_dist
        latent = latent_dist.sample() * vae.config.scaling_factor
        return latent

    @override
    def collate_fn(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        ret = {"prompt": [], "prompt_embedding": [], "image": [], "encoded_image": []}

        for sample in samples:
            prompt = sample.get("prompt", None)
            prompt_embedding = sample.get("prompt_embedding", None)
            image = sample.get("image", None)
            encoded_image = sample.get("encoded_image", None)

            ret["prompt"].append(prompt)
            ret["prompt_embedding"].append(prompt_embedding)
            # image and encoded_image maybe None during validation
            if image is not None:
                ret["image"].append(image)
            if encoded_image is not None:
                ret["encoded_image"].append(encoded_image)

        ret["prompt_embedding"] = torch.stack(ret["prompt_embedding"])
        ret["encoded_image"] = torch.stack(ret["encoded_image"]) if ret["encoded_image"] else None

        prompts = [sample["prompt"] for sample in samples if "prompt" in sample]
        attention_mask = self.components.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.MAX_TTOKEN_LENGTH,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).attention_mask
        ret["attention_mask"] = attention_mask

        # shape of prompt_embedding: [batch_size, max_sequence_length(self.MAX_TTOKEN_LENGTH), embedding_dim(4096)]
        assert ret["attention_mask"].shape == ret["prompt_embedding"].shape[:2]

        return ret

    @override
    def compute_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        batch_size, text_seqlen, text_embedding_dim = batch["prompt_embedding"].shape
        prompt_embeds = batch["prompt_embedding"]
        latent = batch["encoded_image"]

        batch_size, num_channels, height, width = latent.shape
        image_height, image_width = self.state.train_resolution
        vae_scale_factor = 8
        image_seq_len = (
            (image_height // vae_scale_factor) * (image_width // vae_scale_factor)
        ) // (self.state.transformer_config.patch_size**2)

        text_attention_mask = batch["attention_mask"].float()

        # prepare timesteps
        m = (image_seq_len / self.components.scheduler.config.base_image_seq_len) ** 0.5
        mu = (
            m * self.components.scheduler.config.max_shift
            + self.components.scheduler.config.base_shift
        )
        self.components.scheduler.set_timesteps(
            self.components.scheduler.config.num_train_timesteps,
            mu=mu,
            device=self.accelerator.device,
        )
        timestep = torch.randint(
            0,
            self.components.scheduler.config.num_train_timesteps,
            (1,),
            device=self.accelerator.device,
        ).long()

        noise = torch.randn_like(latent)
        model_input, model_label = self.add_noise(latent, noise, timestep[0])
        original_size = torch.tensor(
            [[image_height, image_width] for _ in range(batch_size)],
            dtype=latent.dtype,
            device=self.accelerator.device,
        )
        target_size = torch.tensor(
            [[image_height, image_width] for _ in range(batch_size)],
            dtype=latent.dtype,
            device=self.accelerator.device,
        )
        crop_coords = torch.tensor(
            [[0, 0] for _ in range(batch_size)], dtype=latent.dtype, device=self.accelerator.device
        )

        noise_pred_cond = self.components.transformer(
            hidden_states=model_input,
            encoder_hidden_states=prompt_embeds,
            timestep=timestep,
            original_size=original_size,
            target_size=target_size,
            crop_coords=crop_coords,
            return_dict=False,
            attention_mask=text_attention_mask,
        )[0]

        loss = torch.mean((noise_pred_cond - model_label) ** 2, dim=(1, 2, 3))
        loss = loss.mean()

        return loss

    def add_noise(
        self, latent: torch.Tensor, noise: torch.Tensor, timestep: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to the latent vector based on the timestep.

        Args:
            latent (torch.Tensor): The latent vector to add noise to.
            noise (torch.Tensor): The noise tensor to add.
            timestep (torch.LongTensor): The current timestep.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The noisy latent vector that will be input to the model and the model label.
        """
        num_train_timesteps = self.components.scheduler.config.num_train_timesteps
        # note: sigmas in scheduler is arranged in reversed order
        scale_factor = self.components.scheduler.sigmas[num_train_timesteps - timestep]
        model_input = latent * (1 - scale_factor) + noise * scale_factor
        model_label = noise - latent
        return model_input, model_label

    @override
    def validation_step(
        self, eval_data: dict[str, Any], pipe: CogView4Pipeline
    ) -> list[tuple[str, Image.Image | list[Image.Image]]]:
        """
        Return the data that needs to be saved. For images, the data format is PIL
        """
        prompt = eval_data["prompt"]
        _ = eval_data["prompt_embedding"]

        image_generate = pipe(
            height=self.state.train_resolution[0],
            width=self.state.train_resolution[1],
            prompt=prompt,
            # prompt_embeds=prompt_embedding,
            generator=self.state.generator,
        ).images[0]
        return [("text", prompt), ("image", image_generate)]


register("cogview4-6b", "lora", Cogview4Trainer)
