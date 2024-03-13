import torch
import torch.nn as nn

import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd

from PIL import Image
from diffusers import DDIMScheduler, AutoencoderKL
from .ip_adapter import IPAdapter, StableDiffusionImg2ImgPipeline
from .ip_adapter.utils import is_torch2_available
if is_torch2_available:
    from .ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from .ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor
from typing import List


class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None


class OffsetModel(nn.Module):
    def __init__(self):
        super(OffsetModel, self).__init__()
        self.offset = nn.Parameter(torch.zeros(2, 81, 768))
        self.pred_model = nn.Sequential(
                                nn.Linear(4, 64),
                                nn.ReLU(),
                                nn.Conv1d(10, 64, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv1d(64, 81, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Linear(64, 128),
                                nn.ReLU(),
                                nn.Linear(128, 768),
                                nn.ReLU(),
                            )
    
    def zero_module(module):
        """
        Zero out the parameters of a module and return it.
        """
        for p in module.parameters():
            p.detach().zero_()
        return module

    def positional_encoding(self, x, num_encoding_functions=10):
        # x is a tensor of size [B, 1]
        x = x.unsqueeze(-1)
        device, dtype = x.device, x.dtype
        scales = torch.linspace(0., num_encoding_functions - 1, num_encoding_functions, device=device, dtype=dtype)
        encodings = x * scales.unsqueeze(0)
        encodings = torch.cat((torch.sin(encodings), torch.cos(encodings)), dim=-1)
        return encodings

    def forward(self, thetas, phis, radius=None):
        thetas_emb = self.positional_encoding(thetas, num_encoding_functions=10)
        phis_emb = self.positional_encoding(phis, num_encoding_functions=10)
        emb = torch.cat([thetas_emb, phis_emb], dim=-1)
        return self.pred_model(emb)


class ip_adapter(nn.Module):
    def __init__(self, device, t_range=[0.02, 0.98], **kwargs):
        super().__init__()
        base_model_path = "runwayml/stable-diffusion-v1-5"
        vae_model_path = "stabilityai/sd-vae-ft-mse"
        image_encoder_path = "IP-Adapter/models/image_encoder/"
        ip_ckpt = "IP-Adapter/models/ip-adapter_sd15.bin"

        noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
        vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            scheduler=noise_scheduler,
            vae=vae,
            feature_extractor=None,
            safety_checker=None
        )

        self.to_PIL = T.ToPILImage()
        self.to_tensor = T.ToTensor()
        self.device = device
        self.ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)
        self.num_train_timesteps = self.ip_model.pipe.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.ip_model.pipe.scheduler.alphas_cumprod.to(self.device) # for convenience

        # Add the offset to the image prompt, for adapting the geometry and texture information
        self.offset_model = OffsetModel().to(self.device)

        self.aug_clip = T.Compose([
            T.Resize((224, 224)),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    
    @torch.no_grad()
    def obtain_ref_gt(self, refer_img, render_img, num_samples, strength=0.6):
        img = self.ip_model.generate(pil_image=refer_img, num_samples=num_samples, num_inference_steps=50, seed=42, \
                                     image=render_img, strength=strength)
        return img
    
    def calcul_ref_gt(self, pred_rgb, ref_gt_img):
        image_tensor = torch.stack([self.to_tensor(img) for img in ref_gt_img], dim=0).to(pred_rgb.device)
        dist = torch.mean((image_tensor - pred_rgb)**2, dim=(1,2,3))
        _, min_bs = torch.min(dist, dim=0)
        min_bs = min_bs.item()
        return image_tensor[min_bs:min_bs+1, ...]
 
    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.ip_model.pipe.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.ip_model.pipe.vae.config.scaling_factor

        return latents

    def get_prompt(
            self,
            refer_img,
            num_samples,
            prompt=None,
            negative_prompt=None,
            image_prompt_delta=None,
            ):

        if isinstance(refer_img, Image.Image):
            num_prompts = 1
        else:
            num_prompts = len(refer_img)
        
        if prompt is None:
            prompt = "best quality, high quality"
        else:
            prompt = prompt + ", best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
            
        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.ip_model.get_image_embeds(refer_img)
        if image_prompt_delta != None:
            image_prompt_embeds = image_prompt_embeds + image_prompt_delta
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds = self.ip_model.pipe._encode_prompt(
                prompt, device=self.device, num_images_per_prompt=num_samples, do_classifier_free_guidance=True, negative_prompt=negative_prompt)
            negative_prompt_embeds_, prompt_embeds_ = prompt_embeds.chunk(2)
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)
        
        prompt_embeds = self.ip_model.pipe._encode_prompt(
            prompt=None,
            device=self.device,
            num_images_per_prompt=num_samples,
            do_classifier_free_guidance=True,
            negative_prompt=None,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        return prompt_embeds

    def train_step_with_refgt_iter(self, refer_img, pred_rgb, num_samples=10, **kwargs):
        # only support batch size=1
        render_img = self.to_PIL(pred_rgb.squeeze(0))
        ref_gt_img = self.obtain_ref_gt(refer_img, render_img, num_samples)
        ref_gt_img[0].save("ip_adapter_mid_result.png")
        # pred_ref_gt = self.to_tensor(ref_gt_img).unsqueeze(0)
        pred_ref_gt = self.calcul_ref_gt(pred_rgb, ref_gt_img)
        loss = 0.8 * F.mse_loss(pred_rgb, pred_ref_gt.to(pred_rgb.device))
        return loss

    def IP_align_loss(self, refer_img, pred_rgd):

        # obtain reference image prompt embedding
        image_prompt_embeds_refer, _ = self.ip_model.get_image_embeds(refer_img)

        # obtain rendering image prompt embedding
        clip_image_embeds = self.ip_model.image_encoder(self.aug_clip(pred_rgd)).image_embeds
        image_prompt_embeds_pred = self.ip_model.image_proj_model(clip_image_embeds)

        # loss = 1 - F.cosine_similarity(image_prompt_embeds_refer, image_prompt_embeds_pred)
        loss = torch.pow((image_prompt_embeds_pred - image_prompt_embeds_refer.detach()), 2)
        return loss.mean()

    def train_step(self, refer_img, pred_rgb, guidance_scale=100, as_latent=False, num_samples=1, grad_scale=1, img_prompt_embeddings=None, text_prompt=None,
                   thetas=None, phis=None):

        # obtain image prompt
        if img_prompt_embeddings == None:
            img_prompt_embeddings = self.get_prompt(refer_img, num_samples, prompt=text_prompt)     # size [2, 81, 768]

        # if (thetas is not None) and (phis is not None):
        #     offset = self.offset_model(thetas, phis)
        #     img_prompt_embeddings = img_prompt_embeddings + offset

        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False) * 2 - 1
        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, (latents.shape[0],), dtype=torch.long, device=self.device)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.ip_model.pipe.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)
            noise_pred = self.ip_model.pipe.unet(latent_model_input, tt, encoder_hidden_states=img_prompt_embeddings).sample

            # perform guidance (high scale from paper!)
            noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)
        
        # w(t), sigma_t^2
        w = (1 - self.alphas[t])
        grad = grad_scale * w[:, None, None, None] * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        loss = SpecifyGradient.apply(latents, grad)

        return loss
