# PIL → Tensor 转换
from PIL import Image
from utils.format_utils import pil_to_tensor, tensor_to_pil
pil_img = Image.open("/home/liuchaolei/MVSC/debug/im001.png").convert("RGB")
tensor_img = pil_to_tensor(pil_img)
print(f"Tensor形状: {tensor_img.shape}")  # 输出: torch.Size([3, H, W])

# Tensor → PIL 转换
new_pil_img = tensor_to_pil(tensor_img)
new_pil_img.save("/home/liuchaolei/MVSC/debug/converted_image.jpg")