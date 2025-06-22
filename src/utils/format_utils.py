# # support frames[B, C, H, W] to PIL
# def toPIL(video):
#     # change video into [B, T, C, H, W] type
#     tensor_list = [video[key] for key in sorted(video.keys())]
#     videos = torch.stack(tensor_list, dim=1)

#     videos_pil = []
#     batch_size = videos.size(0)
#     for b in range(batch_size):
#         video_tensor = videos[b] # (T, C, H, W)
#         n_frames = video_tensor.size(0)
#         pil_images = []
#         to_pil = ToPILImage()

#         for i in range(n_frames):
#             frame = video_tensor[i]  # 提取单帧 (C, H, W)
#             pil_image = to_pil(frame)
#             pil_images.append(pil_image)

#         videos_pil.append(pil_images)
#     return videos_pil # [B, T]:PIL images, a list of PIL Image

# def to_Tensor(video_generates):
#     video_tensors = []
#     batch_size = video_generates.size(0)
#     for b in range(batch_size):
#         video_generate = video_generates[b]
#         if isinstance(video_generate[0], np.ndarray):
#             video_np = np.stack(video_generate, axis=0)  # [T, H, W, C] #TODO:make sure the range is right
#         elif isinstance(video_generate[0], Image.Image):
#             video_np = np.stack([np.array(frame) for frame in video_generate], axis=0)
        
#         video_tensor = torch.from_numpy(video_np).float()  # 转为 float32
#         video_tensor = video_tensor.permute(3, 0, 1, 2)    # [C, T, H, W]
#     video_tensors.append(video_tensor)
#     return video_tensors # [B, C, T, H, W]

from PIL import Image
import torch
from torchvision import transforms

def pil_to_tensor(pil_image: Image.Image, device: str) -> torch.Tensor:
    """
    将PIL图像转换为PyTorch Tensor
    参数:
        pil_image: PIL.Image对象 (RGB格式)
    返回:
        torch.Tensor (形状为 [C, H, W], 值范围 [0.0, 1.0])
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = transforms.ToTensor()
    return transform(pil_image).to(device)

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    将PyTorch Tensor转换为PIL图像
    参数:
        tensor: torch.Tensor (形状需为 [C, H, W] 或 [H, W, C])
    返回:
        PIL.Image对象 (RGB格式)
    """
    # 自动处理维度顺序 [7](@ref)
    if tensor.dim() == 4:  # 处理batch维度 [6](@ref)
        tensor = tensor.squeeze(0)
    if tensor.shape[2] == 3:  # 检查是否为CHW格式 [7](@ref)
        tensor = tensor.permute(1, 2, 0)
    
    transform = transforms.ToPILImage()
    return transform(tensor.cpu().detach())