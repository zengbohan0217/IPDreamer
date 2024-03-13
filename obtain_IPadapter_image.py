import os
import torch
import argparse
from PIL import Image
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionInpaintPipelineLegacy,
    DDIMScheduler,
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
    AutoencoderKL
    )
from guidance.ip_adapter import IPAdapter


def obtain_images(prompt, out_path):
    device = 'cuda'
    model_id_or_path = "stabilityai/stable-diffusion-2-1"

    # obtain foreground
    pipe = StableDiffusionPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16, safety_checker=None)
    pipe = pipe.to(device)
    front_prompt = "front of" + prompt
    images = pipe(front_prompt).images
    # images[0].save(os.path.join(out_path, f"{prompt}.png"))
    images[0].save(os.path.join(out_path, f"reference.png"))


def obtain_img_ip_adapter(reference_mask_path, out_path, text):
    device = 'cuda'
    vae_model_path = "stabilityai/sd-vae-ft-mse"
    image_encoder_path = "IP-Adapter/models/image_encoder/"
    ip_ckpt = "IP-Adapter/models/ip-adapter_sd15.bin"
    controlnet_scribble_path = 'lllyasviel/control_v11p_sd15_scribble'
    base_model_path = 'runwayml/stable-diffusion-v1-5'

    controlnet = ControlNetModel.from_pretrained(controlnet_scribble_path, torch_dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_path, controlnet=controlnet, torch_dtype=torch.float16, vae=vae, safety_checker=None
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)

    image = Image.open(os.path.join(out_path, f"reference.png"))
    g_image = Image.open(reference_mask_path)

    images = ip_model.generate(pil_image=image, num_samples=1, num_inference_steps=50, seed=42, image=g_image, prompt=text)
    images[0].save(os.path.join(out_path, f"reference.png"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', type=str, default='./data', help="reference image path for ip adapter")
    parser.add_argument('--text', type=str, default='a hamburger', help="text prompt")
    parser.add_argument('--refer_img_path', type=str, default=None)
    parser.add_argument('--back_img_path', type=str, default=None)
    parser.add_argument('--scribble_img_path', type=str, default=None)
    
    opt = parser.parse_args()

    os.makedirs(opt.out_path, exist_ok=True)
    obtain_images(prompt=opt.text, out_path=opt.out_path)
    if opt.scribble_img_path is not None:
        obtain_img_ip_adapter(out_path=opt.out_path, reference_mask_path=opt.scribble_img_path, text=opt.text)
