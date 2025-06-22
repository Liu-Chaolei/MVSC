import time
import torch
from PIL import Image
from src_inference.pipeline import FluxPipeline
from src_inference.lora_helper import set_single_lora

def clear_cache(transformer):
    for name, attn_processor in transformer.attn_processors.items():
        attn_processor.bank_kv.clear()

# Initialize model
device = "cuda"
base_path = "/data/ssd/liuchaolei/models/FLUX.1-dev"
pipe = FluxPipeline.from_pretrained(base_path, torch_dtype=torch.bfloat16).to("cuda")

# Load OmniConsistency model
set_single_lora(pipe.transformer, 
                "/data/ssd/liuchaolei/models/OmniConsistency/OmniConsistency.safetensors", 
                lora_weights=[1], cond_size=512)

# Load external LoRA
pipe.unload_lora_weights()
pipe.load_lora_weights("/data/ssd/liuchaolei/models/OmniConsistency/LoRAs", 
                       weight_name="lora_name.safetensors")

image_path1 = "/data/ssd/liuchaolei/video_datasets/UVG/sequences/Beauty/im001.png"
prompt = "A woman with blonde hair and green eyes is featured against a black background, her makeup accentuated by smoky eyeshadow and bold red lipstick. She wears simple stud earrings and a black top, exuding confidence and allure. As time passes, her expression shifts to one of serene confidence, with her gaze directed away, suggesting introspection. The lighting softly highlights her features, maintaining a warm ambiance. Her look is consistent, with the addition of subtle pink lipstick, and her hair appears windswept, adding dynamism to her poised demeanor."

subject_images = []
spatial_image = [Image.open(image_path1).convert("RGB")]

width, height = 1920, 1080

start_time = time.time()

image = pipe(
    prompt,
    height=height,
    width=width,
    guidance_scale=3.5,
    num_inference_steps=25,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(5),
    spatial_images=spatial_image,
    subject_images=subject_images,
    cond_size=512,
).images[0]

end_time = time.time()
elapsed_time = end_time - start_time
print(f"code running time: {elapsed_time} s")

# Clear cache after generation
clear_cache(pipe.transformer)

image.save("results/output.png")
