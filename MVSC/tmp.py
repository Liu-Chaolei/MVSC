import torch

# 示例输入张量：[B=2, C=3, T=3, H=224, W=224]
tensor_5d = torch.randn(2, 3, 3, 224, 224)

# 沿 T 维度分割并压缩
split_tensors = torch.split(tensor_5d, 1, dim=2)
split_tensors = [t.squeeze(2) for t in split_tensors]

# 构建字典
tensor_dict = {str(i): t for i, t in enumerate(split_tensors)}

# 验证形状
for key, value in tensor_dict.items():
    print(f"Key {key}: Shape {value.shape}")  
    # 输出示例：Key 0 → [2, 3, 224, 224]