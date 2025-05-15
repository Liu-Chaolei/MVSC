# -*- coding: utf-8 -*-


import torch


def rand_generator(seed: int | None = None) -> torch.Generator:
    if seed is None:
        return torch.Generator()
    return torch.Generator().manual_seed(seed)
