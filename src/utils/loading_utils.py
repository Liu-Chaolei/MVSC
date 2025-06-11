import os
from PIL import Image

def load_image_sequence(folder_path: str) -> list:
    """
    加载文件夹中按数字顺序命名的 PNG 图片为 PIL 图像列表。

    Args:
        folder_path (str): 图片文件夹路径。

    Returns:
        list: 包含所有图片的 PIL.Image 对象列表。
    """
    # 获取文件夹中所有 PNG 文件
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
    
    # 按文件名中的前缀排序（例如 im000.png, im001.png）
    image_files.sort(key=lambda x: x.split('.')[0])  # 提取文件名前缀部分排序
    
    # 加载所有图片到列表
    images = []
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        try:
            img = Image.open(img_path)
            images.append(img)
        except Exception as e:
            print(f"无法加载图片 {img_path}: {e}")
    
    return images

def main():
    folder_path = "path/to/your/images"
    image_list = load_image_sequence(folder_path)
    print(f"加载了 {len(image_list)} 张图片")

if __name__=="__main__":
    main()
