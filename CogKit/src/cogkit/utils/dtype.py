# -*- coding: utf-8 -*-


import torch


def cast_to_torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype

    match dtype:
        case "bfloat16":
            return torch.bfloat16
        case "float16":
            return torch.float16
        case _:
            err_msg = f"Unknown data type: {dtype}"
            raise ValueError(err_msg)
