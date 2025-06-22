import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm
from compressai.datasets import RawVideoSequence, VideoFormat
from typing import Any, Dict, List, Tuple, Union
from compressai.transforms.functional import (
    rgb2ycbcr,
    ycbcr2rgb,
    yuv_420_to_444,
    yuv_444_to_420,
)

Frame = Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, ...]]

# test code / using example
def to_tensors(
    frame: Tuple[np.ndarray, np.ndarray, np.ndarray],
    max_value: int = 1,
    device: str = "cpu",
) -> Frame:
    return tuple(
        torch.from_numpy(np.true_divide(c, max_value, dtype=np.float32)).to(device)
        for c in frame
    )


def convert_yuv420_to_rgb(
    frame: Tuple[np.ndarray, np.ndarray, np.ndarray], device: torch.device, max_val: int
) -> Tensor:
    # yuv420 [0, 2**bitdepth-1] to rgb 444 [0, 1] only for now
    out = to_tensors(frame, device=str(device), max_value=max_val)
    out = yuv_420_to_444(
        tuple(c.unsqueeze(0).unsqueeze(0) for c in out), mode="bicubic"  # type: ignore
    )
    return ycbcr2rgb(out)  # type: ignore

def calculate_fvd(
    org_frames: Frame,
    rec_frames: Tensor,
    device: str = "cpu",
    max_val: int = 255,
    method='styleganv',
    only_final=True,
):

    if method == 'styleganv':
        from fvd.styleganv.fvd import get_fvd_feats, frechet_distance, load_i3d_pretrained
    elif method == 'videogpt':
        from fvd.videogpt.fvd import load_i3d_pretrained, frechet_distance
        from fvd.videogpt.fvd import get_fvd_logits as get_fvd_feats

    # change org_frames from compressai Frame type to Tensor, and change org/rec to RGB
    org_rgbs=[]
    rec_rgbs=[]
    for i in range (len(org_frames)):
        org_rgb = convert_yuv420_to_rgb(
            org_frames[i], device, max_val
        )  # ycbcr2rgb(yuv_420_to_444(org_frame, mode="bicubic"))  # type: ignore
        org_rgbs.append((org_rgb * max_val).clamp(0, max_val).round())
        rec_rgbs.append((rec_frames[i] * max_val).clamp(0, max_val).round())

    org_rgbs = org_rgbs.unsqueeze(0)
    rec_rgbs = rec_rgbs.unsqueeze(0)
    print("calculate_fvd...")

    # org/rec_rgbs [batch_size=1, timestamps, channel, h, w]
    
    assert org_rgbs.shape == rec_rgbs.shape
    assert org_rgbs.shape[2] >= 10, "for calculate FVD, each clip_timestamp must >= 10"

    i3d = load_i3d_pretrained(device=device)
    fvd_results = []

    if only_final:
        # videos_clip [batch_size, channel, timestamps, h, w]
        videos_clip1 = org_rgbs
        videos_clip2 = rec_rgbs

        # get FVD features
        feats1 = get_fvd_feats(videos_clip1, i3d=i3d, device=device)
        feats2 = get_fvd_feats(videos_clip2, i3d=i3d, device=device)

        # calculate FVD
        fvd_results.append(frechet_distance(feats1, feats2))
    else:
        # for calculate FVD, each clip_timestamp must >= 10
        for clip_timestamp in tqdm(range(10, videos1.shape[-3]+1)):
            # videos_clip [batch_size, channel, timestamps[:clip], h, w]
            videos_clip1 = videos1[:, :, : clip_timestamp]
            videos_clip2 = videos2[:, :, : clip_timestamp]

            # get FVD features
            feats1 = get_fvd_feats(videos_clip1, i3d=i3d, device=device)
            feats2 = get_fvd_feats(videos_clip2, i3d=i3d, device=device)
        
            # calculate FVD when timestamps[:clip]
            fvd_results.append(frechet_distance(feats1, feats2))

    return fvd_results

# CUDA_VISIBLE_DEVICES=2 python src/metrics/calculate_fvd.py
def main():
    device = torch.device("cuda")
    # device = torch.device("cpu")
    org_seq = RawVideoSequence.from_file('/data/ssd/liuchaolei/video_datasets/UVG/yuv/Beauty_1920x1080_120fps_420_8bit_YUV.yuv')
    x_recs=[]
    for i in range (len(org_seq)):
        org_rgb = convert_yuv420_to_rgb(org_seq[i], device, 255)
        x_recs.append(org_rgb)

    result = calculate_fvd(org_seq, x_recs, device, method='videogpt', only_final=False)
    print("[fvd-videogpt ]", result)

    result = calculate_fvd(org_seq, x_recs, device, method='styleganv', only_final=False)
    print("[fvd-styleganv]", result)

if __name__ == "__main__":
    main()
