from typing import List, Tuple

import torch
from transformers import AutoTokenizer

from .filters import MeanFilter


def process_prompt_attention_mask(
    tokenizer: AutoTokenizer,
    prompt: List[str],
    prompt_embedding: List[torch.Tensor],
    max_ttoken_length: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Process prompt attention mask for training.

    Args:
        tokenizer: HuggingFace AutoTokenizer instance
        prompt: List of text prompts
        prompt_embedding: List of prompt embeddings
        max_ttoken_length: Maximum text token length

    Returns:
        A tuple of (prompt_embedding, prompt_attention_mask)

    Attention mask values:
        -1: Padding tokens (tokens to be ignored)
        0: Tokens belonging to the first micro-batch (when enable packing)
    """
    # Tokenize the prompt in the batch-level
    tokenized_prompt = tokenizer(
        prompt,
        padding=True,
        max_length=max_ttoken_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    prompt_attention_mask = tokenized_prompt.attention_mask
    num_samples = len(prompt)

    ### Retrieve the unpadded value from prompt_embedding, then pad it to the max length in the batch
    for idx, embedding in enumerate(prompt_embedding):
        prompt_embedding[idx] = embedding[torch.where(prompt_attention_mask[idx] == 1)[0]]

    token_length_list = [embedding.shape[0] for embedding in prompt_embedding]
    max_seqlen = max(token_length_list)
    assert max_seqlen == prompt_attention_mask.shape[1]

    prompt_embedding = torch.nn.utils.rnn.pad_sequence(
        prompt_embedding, batch_first=True, padding_value=0, padding_side="left"
    )

    assert prompt_embedding.shape[0] == prompt_attention_mask.shape[0] == num_samples
    assert prompt_embedding.shape[1] == prompt_attention_mask.shape[1] == max_seqlen

    ### Construct the prompt attention mask
    prompt_attention_mask[
        prompt_attention_mask == 0
    ] = -1  # -1 means padding token type (tokens to be ignored)
    prompt_attention_mask[prompt_attention_mask == 1] = 0  # 0 means tokens belong to micro-batch0

    return prompt_embedding, prompt_attention_mask


def process_latent_attention_mask(
    encoded_images: List[torch.Tensor], patch_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Process latent attention mask for training.

    Args:
        encoded_images: List of encoded image latents with shape [num_channels, height, width]
                        where each image might have different height and width
        patch_size: Patch size for the transformer

    Returns:
        A tuple of (padded_latent, vtoken_attention_mask)

    Attention mask values:
        -1: Padding tokens (tokens to be ignored)
        0: Tokens belonging to the first micro-batch (when enable packing)
    """
    num_samples = len(encoded_images)
    num_latent_channel = encoded_images[0].shape[0]

    # Calculate maximum dimensions for padding
    max_latent_height = max([img.shape[1] for img in encoded_images])
    max_latent_width = max([img.shape[2] for img in encoded_images])

    # Ensure dimensions are divisible by patch_size
    max_latent_height += max_latent_height % patch_size
    max_latent_width += max_latent_width % patch_size

    # Create padded latent tensor and pixel mask
    padded_latent = torch.zeros(
        num_samples,
        num_latent_channel,
        max_latent_height,
        max_latent_width,
        dtype=torch.float32,
    )
    pixel_mask = (
        torch.ones(
            num_samples,
            num_latent_channel,
            max_latent_height,
            max_latent_width,
            dtype=torch.float32,
        )
        * -1
    )

    # Fill padded latent and set mask values
    for idx, latent in enumerate(encoded_images):
        padded_latent[idx, :, : latent.shape[1], : latent.shape[2]] = latent
        # 0 means this pixel belongs to micro-batch0
        pixel_mask[idx, :, : latent.shape[1], : latent.shape[2]] = 0

    # Ensure dimensions are divisible by patch_size
    assert max_latent_height % patch_size == 0 and max_latent_width % patch_size == 0

    # Apply mean filter to create vtoken attention mask
    mean_filter = MeanFilter(kernel_size=patch_size, in_channels=num_latent_channel)
    vtoken_attention_mask = mean_filter(pixel_mask).squeeze(1)  # remove channel dimension

    # 0 means this vtoken belongs to micro-batch0
    vtoken_attention_mask[vtoken_attention_mask != -1] = 0

    return padded_latent, vtoken_attention_mask, pixel_mask
