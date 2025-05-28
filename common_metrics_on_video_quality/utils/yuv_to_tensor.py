import cv2
import torch
import numpy as np

def read_yuv_file(file_path, width, height):
    with open(file_path, 'rb') as f:
        # 读取Y分量（大小为 width*height）
        y_data = np.frombuffer(f.read(width*height), dtype=np.uint8)
        # 读取UV分量（大小为 width*height//2，NV12中UV交替存储）
        uv_data = np.frombuffer(f.read(width*height//2), dtype=np.uint8)
    return y_data, uv_data

def yuv_to_rgb(y_data, uv_data, width, height):
    # 重塑Y和UV为二维数组
    y_img = y_data.reshape((height, width))
    uv_img = uv_data.reshape((height//2, width//2, 2))  # NV12的UV分量为交错存储
    
    # 上采样UV分量到原始分辨率
    uv_img_upsampled = cv2.resize(uv_img, (width, height), interpolation=cv2.INTER_LINEAR)
    
    # 合并为YUV格式并转换为RGB
    yuv_img = np.stack([y_img, uv_img_upsampled[...,0], uv_img_upsampled[...,1]], axis=-1)
    rgb_img = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR_NV12)
    return rgb_img

def yuv_to_tensor(file_path, width, height, num_frames):
    # 读取YUV数据
    y_data, uv_data = read_yuv_file(file_path, width, height)
    
    # 初始化视频张量
    videos = []
    
    for frame_idx in range(num_frames):
        # 提取单帧YUV数据（需根据视频帧存储方式调整索引）
        y_frame = y_data[frame_idx*width : (frame_idx+1)*width]
        uv_frame = uv_data[frame_idx*(width//2)*(height//2) : (frame_idx+1)*(width//2)*(height//2)]
        
        # 转换为RGB
        rgb_frame = yuv_to_rgb(y_frame, uv_frame, width, height)
        
        # 归一化到[0,1]并转换为PyTorch张量
        tensor_frame = torch.from_numpy(rgb_frame).float() / 255.0
        tensor_frame = tensor_frame.permute(2, 0, 1)  # 转换为 C H W
        
        videos.append(tensor_frame)
    
    # 堆叠所有帧为视频张量
    video_tensor = torch.stack(videos)  # Shape: [帧数, 3, H, W]
    return video_tensor.unsqueeze(0)  # 添加视频维度，Shape: [1, 帧数, 3, H, W]