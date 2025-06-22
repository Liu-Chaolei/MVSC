import av
import numpy as np

# 输入输出设置
input_path = "input.yuv"  # 未压缩视频源（如YUV）
output_path = "output_hevc.mp4"
resolution = (1920, 1080)  # 视频分辨率
frame_rate = 30            # 帧率
pix_fmt = "yuv420p"       # 像素格式

# 创建输出容器并添加 HEVC 视频流
container = av.open(output_path, mode="w")
stream = container.add_stream("hevc", rate=frame_rate)
stream.width = resolution[0]
stream.height = resolution[1]
stream.pix_fmt = pix_fmt
stream.options = {
    "profile": "main",    # HEVC 配置（main/main10）
    "crf": "23",          # 质量控制（0-51，值越低质量越高）
    "preset": "medium",   # 编码速度/质量平衡（ultrafast/slow等）
    "tune": "psnr",       # 优化目标（psnr/ssim/zerolatency）
    "x265-params": "keyint=60:min-keyint=30"  # 自定义参数（如GOP长度）
}

# 编码过程（模拟输入帧）
for i in range(300):  # 编码300帧
    # 生成模拟YUV数据（实际应从文件/摄像头读取）
    y_data = np.random.randint(0, 256, (resolution[1], resolution[0]), dtype=np.uint8)
    uv_data = np.random.randint(0, 256, (resolution[1]//2, resolution[0]//2), dtype=np.uint8)
    frame = av.VideoFrame.from_ndarray(
        [y_data, uv_data, uv_data], 
        format=pix_fmt
    )
    
    # 编码并写入容器
    for packet in stream.encode(frame):
        container.mux(packet)

# 刷新编码器并写入剩余数据
for packet in stream.encode():
    container.mux(packet)
container.close()