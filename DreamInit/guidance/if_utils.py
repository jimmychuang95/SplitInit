from transformers import logging
from diffusers import IFPipeline, DDPMScheduler, DiffusionPipeline

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

import os

from torch.cuda.amp import custom_bwd, custom_fwd
from guidance.perpneg_utils import weighted_perpendicular_aggregator

from guidance.attn_map_utils import *




def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

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


class IF(nn.Module):
    def __init__(self, device, vram_O, is_BSD, t_range=[0.02, 0.98]):
        print(f"\n IF Init Start - Allocated memory: {torch.cuda.memory_allocated() / 1024**2} MB")
        super().__init__()

        self.device = device

        print(f'[INFO] loading DeepFloyd IF-I-XL...')

        # model_key = "/mnt/models--DeepFloyd--IF-I-XL-v1.0/snapshots/c03d510e9b75bce9f9db5bb85148c1402ad7e694"
        model_key = "DeepFloyd/IF-I-XL-v1.0"

        is_torch2 = torch.__version__[0] == '2'

        # Create model
        pipe = IFPipeline.from_pretrained(model_key, variant="fp16", torch_dtype=torch.float16)
        # if not is_torch2:
        #     pipe.enable_xformers_memory_efficient_attention()

        if vram_O:
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

        # print(f"\n Finish load Pipe - Allocated memory: {torch.cuda.memory_allocated() / 1024**2} MB")
        pipe = init_pipeline(pipe)

        self.unet = pipe.unet
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.scheduler = pipe.scheduler

        self.pipe = pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        self.is_BSD = is_BSD

        print(f'[INFO] loaded DeepFloyd IF-I-XL!')

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        # prompt: [str]
        # TODO: should I add the preprocessing at https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/deepfloyd_if/pipeline_if.py#LL486C10-L486C28
        prompt = self.pipe._text_preprocessing(prompt, clean_caption=False)
        inputs = self.tokenizer(prompt, padding='max_length', max_length=77, truncation=True, add_special_tokens=True, return_tensors='pt')
        # print(f'tokenized input: {inputs.input_ids[0]}')
        # decoded_text = self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=False)
        # print(f'[INFO] Decoded text: {decoded_text}')
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]
        return embeddings
    
    def unload_encoder(self):
        self.text_encoder = None
        self.pipe.text_encoder = None
        # self.tokenizer = None
        # self.pipe.tokenizer = None
        torch.cuda.empty_cache()
        # print(f"\n Finish unload Encoder - Allocated memory: {torch.cuda.memory_allocated() / 1024**2} MB")


    def train_step(self, text_embeddings, pred_rgb, guidance_scale=100, grad_scale=1):

        # [0, 1] to [-1, 1] and make sure shape is [64, 64]
        images = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False) * 2 - 1

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, (images.shape[0],), dtype=torch.long, device=self.device)
        print(f't: {t}')
        # t = torch.randint(self.min_step, self.max_step + 1, (1,), dtype=torch.long, device=self.device).repeat(pred_rgb.shape[0])

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(images)
            images_noisy = self.scheduler.add_noise(images, noise, t)

            # pred noise
            model_input = torch.cat([images_noisy] * 2)
            model_input = self.scheduler.scale_model_input(model_input, t)
            tt = torch.cat([t] * 2)
            noise_pred = self.unet(model_input, tt, encoder_hidden_states=text_embeddings).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1], dim=1)
            noise_pred_text, predicted_variance = noise_pred_text.split(model_input.shape[1], dim=1)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # TODO: how to use the variance here?
            # noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)

        if self.is_BSD:
            classifier_guidance = (noise_pred_text - noise_pred_uncond) * 15
            denoise_guidance =  noise_pred_uncond

            def dot(tensor1, tensor2):
                return torch.sum(tensor1 * tensor2, dim=1)
            alpha = dot(denoise_guidance.flatten(1) - classifier_guidance.flatten(1), denoise_guidance.flatten(1)) / dot(denoise_guidance.flatten(1) - classifier_guidance.flatten(1), denoise_guidance.flatten(1) - classifier_guidance.flatten(1))
            alpha = torch.where(alpha < 0, 0, alpha)
            alpha = torch.where(alpha > 1, 1, alpha)
            alpha = alpha[:, None, None, None]
            grad_ = alpha * classifier_guidance + (1 - alpha) * denoise_guidance
            alpha_prod_t = self.scheduler.alphas_cumprod.to(self.device)[int(t)]
            w = alpha_prod_t ** 0.5
            grad =  grad_ * w
            
            grad = torch.nan_to_num(grad)
            loss = SpecifyGradient.apply(images, grad)

            return loss

        # w(t), sigma_t^2
        w = (1 - self.alphas[t])
        grad = grad_scale * w[:, None, None, None] * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        targets = (images - grad).detach()
        loss = 0.5 * F.mse_loss(images.float(), targets, reduction='sum') / images.shape[0]

        return loss

    def train_step_perpneg(self, text_embeddings, weights, pred_rgb, global_step, prompt, workspace, guidance_scale=100, grad_scale=1):

        B = pred_rgb.shape[0]
        K = (text_embeddings.shape[0] // B) - 1 # maximum number of prompts   

        # [0, 1] to [-1, 1] and make sure shape is [64, 64]
        images = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False) * 2 - 1

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, (images.shape[0],), dtype=torch.long, device=self.device)
        # t = torch.randint(self.min_step, self.max_step + 1, (1,), dtype=torch.long, device=self.device).repeat(pred_rgb.shape[0])

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(images)
            images_noisy = self.scheduler.add_noise(images, noise, t)

            # pred noise
            model_input = torch.cat([images_noisy] * (1 + K))
            model_input = self.scheduler.scale_model_input(model_input, t)
            tt = torch.cat([t] * (1 + K))
            unet_output = self.unet(model_input, tt, encoder_hidden_states=text_embeddings, global_step=global_step).sample
            noise_pred_uncond, noise_pred_text = unet_output[:B], unet_output[B:]

            # print(f'unet_output: {unet_output.shape}, noise_pred_uncond: {noise_pred_uncond.shape}, noise_pred_text: {noise_pred_text.shape}')

            noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1], dim=1)
            noise_pred_text, predicted_variance = noise_pred_text.split(model_input.shape[1], dim=1)
            # noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            delta_noise_preds = noise_pred_text - noise_pred_uncond.repeat(K, 1, 1, 1)
            noise_pred = noise_pred_uncond + guidance_scale * weighted_perpendicular_aggregator(delta_noise_preds, weights, B)

        if self.is_BSD:
            classifier_guidance = weighted_perpendicular_aggregator(delta_noise_preds, weights, B) * 15
            denoise_guidance =  noise_pred_uncond

            def dot(tensor1, tensor2):
                return torch.sum(tensor1 * tensor2, dim=1)
            alpha = dot(denoise_guidance.flatten(1) - classifier_guidance.flatten(1), denoise_guidance.flatten(1)) / dot(denoise_guidance.flatten(1) - classifier_guidance.flatten(1), denoise_guidance.flatten(1) - classifier_guidance.flatten(1))
            alpha = torch.where(alpha < 0, 0, alpha)
            alpha = torch.where(alpha > 1, 1, alpha)
            alpha = alpha[:, None, None, None]

            grad_ = alpha * classifier_guidance + (1 - alpha) * denoise_guidance
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
            grad =  grad_ * w
            grad = torch.nan_to_num(grad)
            targets = (images - grad).detach()
            loss = 0.5 * F.mse_loss(images.float(), targets, reduction='sum') / images.shape[0]
            #loss = SpecifyGradient.apply(images, grad)

        else: # sds loss
            # w(t), sigma_t^2
            w = (1 - self.alphas[t])
            grad = grad_scale * w[:, None, None, None] * (noise_pred - noise)
            grad = torch.nan_to_num(grad)

            targets = (images - grad).detach()
            loss = 0.5 * F.mse_loss(images.float(), targets, reduction='sum') / images.shape[0]

        if global_step % 100 == 0 or global_step == 1:
            save_dir = os.path.join(workspace, 'unet_process')
            os.makedirs(save_dir, exist_ok=True)

            unet_images = torch.cat([images, images_noisy, images - grad], dim=0)
            image_grid = vutils.make_grid(unet_images, nrow=4, normalize=False)
            vutils.save_image(image_grid, os.path.join(save_dir, f'denoise_{global_step}.jpg'))
            save_attention_maps(attn_maps, self.pipe, prompt, global_step, base_dir=save_dir, B = B, unconditional=True)

        return loss

    @torch.no_grad()
    def produce_imgs(self, text_embeddings, height=64, width=64, num_inference_steps=50, guidance_scale=7.5):

        images = torch.randn((1, 3, height, width), device=text_embeddings.device, dtype=text_embeddings.dtype)
        images = images * self.scheduler.init_noise_sigma

        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            model_input = torch.cat([images] * 2)
            model_input = self.scheduler.scale_model_input(model_input, t)

            # predict the noise residual
            noise_pred = self.unet(model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1], dim=1)
            noise_pred_text, predicted_variance = noise_pred_text.split(model_input.shape[1], dim=1)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)

            # compute the previous noisy sample x_t -> x_t-1
            images = self.scheduler.step(noise_pred, t, images).prev_sample

        images = (images + 1) / 2

        return images


    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        pos_embeds = self.get_text_embeds(prompts) # [1, 77, 768]
        neg_embeds = self.get_text_embeds(negative_prompts)
        text_embeds = torch.cat([neg_embeds, pos_embeds], dim=0) # [2, 77, 768]

        # Text embeds -> img
        imgs = self.produce_imgs(text_embeds, height=height, width=width, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale) # [1, 4, 64, 64]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs


if __name__ == '__main__':

    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', type=str)
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('--vram_O', action='store_true', help="optimization for low VRAM usage")
    parser.add_argument('-H', type=int, default=64)
    parser.add_argument('-W', type=int, default=64)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device('cuda')

    sd = IF(device, opt.vram_O)

    imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    plt.show()




