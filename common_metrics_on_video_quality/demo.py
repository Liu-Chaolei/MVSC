import torch
import numpy as np
import cv2
import json
import os

from calculate_fvd import calculate_fvd
from calculate_psnr import calculate_psnr
from calculate_ssim import calculate_ssim
from calculate_lpips import calculate_lpips
from utils.yuv_to_tensor import yuv_to_tensor
from utils.mp4_to_tensor import mp4_to_tensor, batch_mp4_to_tensor

# ps: pixel value should be in [0, 1]!

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
video_path1 = '/data/ssd/liuchaolei/results/MVSC/ori_videos_mp4/UVG_crop/Beauty_720x480_120fps_8bit_4201.mp4'
video_path2 = '/data/ssd/liuchaolei/results/MVSC/enhance_videos_mp4/UVG_crop/Beauty_720x480_120fps_8bit_420.mp4'

_, ext = os.path.splitext(video_path1)
if os.path.isfile(video_path1):
    if ext == ".mp4":
        video1, VIDEO_LENGTH, WIDTH, HEIGHT = mp4_to_tensor(video_path1, device)
        video2, VIDEO_LENGTH, WIDTH, HEIGHT = mp4_to_tensor(video_path2, device)
        video1 = video1.unsqueeze(0)
        video2 = video2.unsqueeze(0)
    elif ext == ".yuv":
        video1 = yuv_to_tensor(video_path1, WIDTH, HEIGHT, VIDEO_LENGTH)
        video2 = yuv_to_tensor(video_path2, WIDTH, HEIGHT, VIDEO_LENGTH)
        WIDTH = 720
        HEIGHT = 480
        VIDEO_LENGTH = 100
    else:
        raise NotImplementedError(f"Unsupported video input format, only support mp4 or image sequence")
elif os.path.isdir(video_path1):
    video1, VIDEO_LENGTH, WIDTH, HEIGHT = batch_mp4_to_tensor(video_path1, device)
    video2, VIDEO_LENGTH, WIDTH, HEIGHT = batch_mp4_to_tensor(video_path2, device)



video1 = video1.to(device)
video2 = video2.to(device)

result = {}
only_final = False
only_final = True
result['fvd'] = calculate_fvd(video1, video2, device, method='styleganv', only_final=only_final)
# result['fvd'] = calculate_fvd(video1, video2, device, method='videogpt', only_final=only_final)
result['ssim'] = calculate_ssim(video1, video2, only_final=only_final)
result['psnr'] = calculate_psnr(video1, video2, only_final=only_final)
result['lpips'] = calculate_lpips(video1, video2, device, only_final=only_final)
print(json.dumps(result, indent=4))