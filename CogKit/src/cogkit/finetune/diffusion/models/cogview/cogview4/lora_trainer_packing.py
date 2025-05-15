# -*- coding: utf-8 -*-


from typing import Any, Tuple

import torch
from typing_extensions import override

from cogkit.finetune import register
from cogkit.finetune.diffusion.trainer import DiffusionState
from cogkit.finetune.utils import (
    expand_list,
    process_latent_attention_mask,
    process_prompt_attention_mask,
)

from .lora_trainer import Cogview4Trainer


class Cogview4LoraPackingTrainer(Cogview4Trainer):
    DOWNSAMPLER_FACTOR = 8

    @override
    def _init_state(self) -> DiffusionState:
        patch_size = self.components.transformer.config.patch_size
        height, width = self.args.train_resolution
        sample_height, sample_width = (
            height // self.DOWNSAMPLER_FACTOR,
            width // self.DOWNSAMPLER_FACTOR,
        )
        max_vtoken_length = sample_height * sample_width // patch_size**2
        training_seq_length = max_vtoken_length + self.MAX_TTOKEN_LENGTH

        return DiffusionState(
            weight_dtype=self.get_training_dtype(),
            train_resolution=self.args.train_resolution,
            max_vtoken_length=max_vtoken_length,
            training_seq_length=training_seq_length,
        )

    @override
    def sample_to_length(self, sample: dict[str, Any]) -> int:
        # shape of image_latent: [num_channels, height, width]
        image_latent = sample["encoded_image"]
        prompt_embedding = sample["prompt_embedding"]
        assert image_latent.ndim == 3
        assert prompt_embedding.ndim == 2
        num_channels, latent_height, latent_width = image_latent.shape

        patch_size = self.components.transformer.config.patch_size
        paded_width = latent_width + latent_width % patch_size
        paded_height = latent_height + latent_height % patch_size
        latent_length = paded_height * paded_width // patch_size**2
        if latent_length > self.state.max_vtoken_length:
            raise ValueError(
                f"latent_length {latent_length} is greater than max_vtoken_length {self.state.max_vtoken_length}, "
                f"which means there is at least one sample in the batch has resolution greater than "
                f"{self.args.train_resolution[0]}x{self.args.train_resolution[1]}"
            )

        assert (
            self.MAX_TTOKEN_LENGTH + self.state.max_vtoken_length == self.state.training_seq_length
        )
        assert latent_length + prompt_embedding.shape[0] <= self.state.training_seq_length

        return latent_length + prompt_embedding.shape[0]

    @override
    def collate_fn_packing(self, samples: list[dict[str, list[Any]]]) -> dict[str, Any]:
        """
        Note: This collate_fn is for the training dataloader.
        For validation, you should use the collate_fn from Cogview4Trainer.
        """
        batched_data = {
            "prompt_embedding": None,
            "encoded_image": None,
            "attention_mask": {
                "batch_flag": None,
                "text_embedding_attn_mask": None,
                "latent_embedding_attn_mask": None,
            },
            "pixel_mask": None,
        }
        samples = expand_list(samples)

        prompt_embedding, prompt_attention_mask = process_prompt_attention_mask(
            self.components.tokenizer,
            samples["prompt"],
            samples["prompt_embedding"],
            self.MAX_TTOKEN_LENGTH,
        )

        patch_size = self.components.transformer.config.patch_size
        padded_latent, vtoken_attention_mask, pixel_mask = process_latent_attention_mask(
            samples["encoded_image"], patch_size
        )

        # Store in batched_data
        batched_data["prompt_embedding"] = prompt_embedding
        batched_data["attention_mask"]["text_embedding_attn_mask"] = prompt_attention_mask
        batched_data["encoded_image"] = padded_latent
        batched_data["attention_mask"]["latent_embedding_attn_mask"] = vtoken_attention_mask
        batched_data["pixel_mask"] = pixel_mask

        return batched_data

    @override
    def compute_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        patch_size = self.components.transformer.config.patch_size
        prompt_embeds = batch["prompt_embedding"]
        latent = batch["encoded_image"]
        batch_size, text_seqlen, text_embedding_dim = prompt_embeds.shape
        batch_size, num_channels, height, width = latent.shape

        attn_mask = batch["attention_mask"]
        latent_attention_mask = attn_mask["latent_embedding_attn_mask"].float()
        latent_attention_mask_1d = latent_attention_mask.reshape(batch_size, -1)
        vtoken_seq_len = torch.sum(latent_attention_mask_1d != -1, dim=1)
        latent_shape = []

        for i, data in enumerate(latent_attention_mask):
            row_indices = torch.where(data[:, 0] == -1)[0]
            if len(row_indices) > 0:
                num_rows = row_indices[0].item()
            else:
                num_rows = data.shape[0]

            col_indices = torch.where(data[0, :] == -1)[0]
            if len(col_indices) > 0:
                num_cols = col_indices[0].item()
            else:
                num_cols = data.shape[1]
            latent_shape.append((num_rows, num_cols))
        latent_shape = torch.tensor(latent_shape)
        original_shape = latent_shape * self.DOWNSAMPLER_FACTOR * patch_size
        assert torch.equal(
            vtoken_seq_len.cpu(),
            torch.prod((original_shape / (self.DOWNSAMPLER_FACTOR * patch_size)), dim=1),
        )

        # prepare sigmas
        scheduler = self.components.scheduler
        sigmas = torch.linspace(
            scheduler.sigma_min,
            scheduler.sigma_max,
            scheduler.config.num_train_timesteps,
            device=self.accelerator.device,
        )

        m = (vtoken_seq_len / scheduler.config.base_image_seq_len) ** 0.5
        mu = m * scheduler.config.max_shift + scheduler.config.base_shift
        mu = mu.unsqueeze(1)
        sigmas = mu / (mu + (1 / sigmas - 1))
        sigmas = torch.flip(sigmas, dims=[1])
        sigmas = torch.cat([sigmas, torch.zeros((batch_size, 1), device=sigmas.device)], dim=1)
        self.components.scheduler.sigmas = sigmas

        timestep = torch.randint(
            0,
            scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.accelerator.device,
        )

        noise = torch.randn_like(latent)
        model_input, model_label = self.add_noise(latent, noise, timestep)
        original_size = torch.tensor(
            original_shape,
            dtype=latent.dtype,
            device=self.accelerator.device,
        )
        target_size = torch.tensor(
            original_shape,
            dtype=latent.dtype,
            device=self.accelerator.device,
        )
        crop_coords = torch.tensor(
            [[0, 0] for _ in range(batch_size)], dtype=latent.dtype, device=self.accelerator.device
        )

        # FIXME: add attn support for cogview4 transformer
        noise_pred_cond = self.components.transformer(
            hidden_states=model_input,
            encoder_hidden_states=prompt_embeds,
            timestep=timestep,
            original_size=original_size,
            target_size=target_size,
            crop_coords=crop_coords,
            return_dict=False,
            attention_mask=attn_mask,
        )[0]

        pixel_mask = batch["pixel_mask"]
        pixel_mask[pixel_mask == 0] = 1
        pixel_mask[pixel_mask == -1] = 0
        loss = torch.mean(((noise_pred_cond - model_label) ** 2) * pixel_mask, dim=(1, 2, 3))
        loss = loss.mean()

        return loss

    @override
    def add_noise(
        self, latent: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to the latent vector based on the timestep.

        Args:
            latent (torch.Tensor): The latent vector to add noise to.
            noise (torch.Tensor): The noise tensor to add.
            timestep (torch.Tensor): The current timestep.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The noisy latent vector that will be input to the model and the model label.
        """
        num_train_timesteps = self.components.scheduler.config.num_train_timesteps
        # note: sigmas in scheduler is arranged in reversed order
        index = num_train_timesteps - timestep
        scale_factor = (
            torch.gather(self.components.scheduler.sigmas, dim=1, index=index.unsqueeze(1))
            .squeeze(1)
            .view(-1, 1, 1, 1)
            .to(latent.device)
        )

        model_input = latent * (1 - scale_factor) + noise * scale_factor
        model_label = noise - latent
        return model_input, model_label


register("cogview4-6b", "lora-packing", Cogview4LoraPackingTrainer)
