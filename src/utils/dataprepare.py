from moviepy.editor import VideoFileClip

video = VideoFileClip("input.mp4")
audio = video.audio
audio.write_audiofile("output_audio.mp3")  # 保存为MP3/WAV格式[1,3,4](@ref)