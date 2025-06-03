import io

import argparse
import numpy as np
import torch
from decord import cpu, VideoReader, bridge
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_video(video_data, strategy='chat'):
    bridge.set_bridge('torch')
    mp4_stream = video_data
    num_frames = 24
    decord_vr = VideoReader(io.BytesIO(mp4_stream), ctx=cpu(0))

    frame_id_list = None
    total_frames = len(decord_vr)
    if strategy == 'base':
        clip_end_sec = 60
        clip_start_sec = 0
        start_frame = int(clip_start_sec * decord_vr.get_avg_fps())
        end_frame = min(total_frames,
                        int(clip_end_sec * decord_vr.get_avg_fps())) if clip_end_sec is not None else total_frames
        frame_id_list = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
    elif strategy == 'chat':
        timestamps = decord_vr.get_frame_timestamp(np.arange(total_frames))
        timestamps = [i[0] for i in timestamps]
        max_second = round(max(timestamps)) + 1
        frame_id_list = []
        for second in range(max_second):
            closest_num = min(timestamps, key=lambda x: abs(x - second))
            index = timestamps.index(closest_num)
            frame_id_list.append(index)
            if len(frame_id_list) >= num_frames:
                break

    video_data = decord_vr.get_batch(frame_id_list)
    video_data = video_data.permute(3, 0, 1, 2)
    return video_data


def generate_text(
    prompt: str = "Please describe this video in detail.",
    temperature: float = 0.1,
    model_path: str = "THUDM/cogvlm2-llama3-caption",
    video_path: str = None,
    video = None,
    device: str = "cpu"
):
    strategy = 'chat'

    TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[
        0] >= 8 else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=TORCH_TYPE,
        trust_remote_code=True
    ).eval().to(device)

    if video_path and video is None:
        video_data = open(video_path, 'rb').read()
        video = load_video(video_data, strategy=strategy)
        videos = video.unsqueeze(0)
    elif video and video_path is None:
        video = video
        # change video into [B, C, T, H, W] type
        tensor_list = [video[key] for key in sorted(video.keys())]
        videos = torch.stack(tensor_list, dim=2)
    else:
        raise Exception("Invalid Video Input")

    responses=[]
    batch_size = videos.size(0)
    for b in range(batch_size):
        video = videos[b]
        history = []
        query = prompt
        inputs = model.build_conversation_input_ids(
            tokenizer=tokenizer,
            query=query,
            images=[video],
            history=history,
            template_version=strategy
        )
        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
            'images': [[inputs['images'][0].to('cuda').to(TORCH_TYPE)]],
        }
        gen_kwargs = {
            "max_new_tokens": 2048,
            "pad_token_id": 128002,
            "top_k": 1,
            "do_sample": False,
            "top_p": 0.1,
            "temperature": temperature,
        }
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        responses.append(response)
    return responses

# python src/utils/generate_text.py
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    video_path = '/home/liuchaolei/MVSC/src/test.mp4'
    model_path: str = "THUDM/cogvlm2-llama3-caption"
    caption = generate_text(video_path=video_path, model_path=model_path, device=device)
    print(caption)

if __name__=='__main__':
    main()