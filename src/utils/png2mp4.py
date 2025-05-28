import cv2
import os
import subprocess

def png_to_mp4(input_dir, output_path, fps=30):
    """
    将PNG图像序列转换为MP4视频
    参数：
        input_dir: 图像所在目录（文件名需按数字排序，如frame_0000.png, frame_0001.png）
        output_path: 输出视频路径（如output.mp4）
        fps: 视频帧率（默认30）
    """
    # 获取所有PNG文件并排序
    frames = sorted(
        [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.png')],
        key=lambda x: int(''.join(filter(str.isdigit, x)))
    )

    if not frames:
        raise FileNotFoundError("No PNG files found in the directory")

    # 读取第一帧获取尺寸
    sample = cv2.imread(frames[0])
    h, w = sample.shape[:2]

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # H.264编码
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame_path in frames:
        # 读取图像并写入视频
        img = cv2.imread(frame_path)
        writer.write(img)
    
    writer.release()
    print(f"Video saved to {output_path}")

def png_to_mp4_multi(input_dir, output_path, fps):
    # 获取帧列表
    frames = sorted(
        [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.png')],
        key=lambda x: int(''.join(filter(str.isdigit, os.path.splitext(x)[0])))
    )
    
    if not frames:
        raise FileNotFoundError("No PNG files found")

    # 使用FFmpeg管道处理（绕过OpenCV编码限制）
    cmd = [
        'ffmpeg',
        '-y',  # 覆盖输出
        '-framerate', fps,  # 根据实际帧率调整
        '-i', os.path.join(input_dir, 'frame_%04d.png'),
        '-c:v', 'libx264',
        '-crf', '0',  # 无损模式
        '-preset', 'ultrafast',  # 加速编码
        output_path
    ]
    subprocess.run(cmd, check=True)


def main():
    png_to_mp4(
        input_dir="/data/ssd/liuchaolei/results/MVSC/rec_videos/UVG_crop/2/Beauty_720x480_120fps_8bit_420",
        output_path="/data/ssd/liuchaolei/results/MVSC/rec_videos_mp4/UVG_crop/2/Beauty_720x480_120fps_8bit_420.mp4",
        fps=50
    )


if __name__=="__main__":
    main()
