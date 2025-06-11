import torch
from diffusers.utils import load_image
from diffusers import FluxControlNetModel
from diffusers.pipelines import FluxControlNetPipeline

def clear_cache(transformer):
    for name, attn_processor in transformer.attn_processors.items():
        attn_processor.bank_kv.clear()

def generate_image(
    prompt: str = "",
    dtype: torch.dtype = torch.bfloat16,
    controlnet_path: str = "jasperai/Flux.1-dev-Controlnet-Upscaler",
    basenet_path: str = "black-forest-labs/FLUX.1-dev",
    output_path: str = "results/output.png",
    image = None
    image_path: str = "",
    num_inference_steps: int = 28,
    guidance_scale: float = 6.0,
):
    # Load pipeline
    controlnet = FluxControlNetModel.from_pretrained(controlnet_path, torch_dtype=dtype)
    pipe = FluxControlNetPipeline.from_pretrained(basenet_path, controlnet=controlnet, torch_dtype=dtype)
    pipe.to("cuda")

    control_image = load_image(image_path)
    w, h = control_image.size

    # Upscale x1
    control_image = control_image.resize((w, h))

    image = pipe(
        prompt=prompt, 
        control_image=control_image,
        controlnet_conditioning_scale=0.6,
        num_inference_steps=num_inference_steps, 
        guidance_scale=3.5,
        height=control_image.size[1],
        width=control_image.size[0]
    ).images[0]
    image

    # Clear cache after generation
    clear_cache(pipe.transformer)

    image.save(output_path)

# CUDA_VISIBLE_DEVICES=3 python src/utils/generate_image2.py
def main():
    controlnet_path = "/data/ssd/liuchaolei/models/Flux.1-dev-Controlnet-Upscaler"
    basenet_path = "/data/ssd/liuchaolei/models/FLUX.1-dev"
    image_path = '/data/ssd/liuchaolei/video_datasets/UVG/sequences/Beauty/im001.png'
    generate_image()

if __name__=='__main__':
    main()