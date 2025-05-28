import os
import cv2
import torch
import numpy as np
from tqdm import tqdm  # 进度条库

def mp4_to_tensor(video_path, device='cpu'):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    # 获取视频元数据
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 初始化张量
    video_tensor = []
    
    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # 转换为RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 归一化并转换为PyTorch张量
        frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
        frame_tensor = frame_tensor.permute(2, 0, 1)  # HWC → CHW
        
        video_tensor.append(frame_tensor)
    
    cap.release()
    
    # 堆叠帧并调整设备
    video_tensor = torch.stack(video_tensor).to(device)
    return video_tensor, total_frames, width, height


def batch_mp4_to_tensor(folder_path, device='cpu'):
    """
    批量处理文件夹中的 MP4 文件，返回堆叠后的张量及元数据列表
    """
    video_tensors = []
    # metadata_list = []  # 存储每个视频的 (fps, total_frames, width, height)

    # 遍历文件夹中的所有文件
    for filename in tqdm(os.listdir(folder_path), desc="Processing videos"):
        if not filename.endswith('.mp4'):
            continue
            
        file_path = os.path.join(folder_path, filename)
        try:
            # 调用原有单文件处理函数
            video_tensor, total_frames, width, height = mp4_to_tensor(file_path, device)

            video_tensors.append(video_tensor)
            # metadata_list.append({
            #     'filename': filename,
            #     'fps': cap.get(cv2.CAP_PROP_FPS),  # 需在 mp4_to_tensor 中返回 cap
            #     'total_frames': total_frames,
            #     'resolution': (width, height)
            # })
            
        except Exception as e:
            print(f"跳过文件 {filename}: {str(e)}")
            continue

    # 堆叠所有视频张量到首维
    if video_tensors:
        stacked_tensor = torch.stack(video_tensors, dim=0)  # Shape: (N, T, C, H, W)
        stacked_tensor = stacked_tensor.to(device)
        return stacked_tensor, total_frames, width, height
    else:
        raise ValueError("未找到有效的视频文件")
