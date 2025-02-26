import torch
import argparse

from trainer import *
from generator.DreamInit import DreamInit
from generator.provider import GenerateCircleCameras, cameraList_from_RcamInfos
from diffusers import IFPipeline, StableDiffusionPipeline

from inference import generate_camera_path,parse_args




def save_xyz_color(opt, prompt,model_path,save_xyz_path,save_color_path, gaussians):
    # opt = parse_args()
    opt.xyzres = True
    opt.fp16 = True
    opt.image_h = opt.h
    opt.image_w = opt.w
    device = torch.device('cuda')
    opt.device = device

    generator = DreamInit(opt).to(device)
    model_ckpt = torch.load(model_path, map_location='cpu')
    generator.load_state_dict(model_ckpt['model'])
    if 'ema' in model_ckpt and opt.ema:
        ema = ExponentialMovingAverage(generator.parameters(), decay=0.99)
        ema.load_state_dict(model_ckpt['ema'])
        ema.copy_to()

    model_key = "DeepFloyd/IF-I-XL-v1.0"


    # pipe = IFPipeline.from_pretrained(
    #     model_key,
    #     variant="fp16",
    #     torch_dtype=torch.float16)
    # tokenizer = pipe.tokenizer
    # text_encoder = pipe.text_encoder.to(device)

    # cameras = generate_camera_path(120, opt)

    # generator.eval()

    # inputs = tokenizer(prompt, padding='max_length', max_length=77, truncation=True, add_special_tokens=True, return_tensors='pt')
    # embeddings = text_encoder(inputs.input_ids.to(device))[0]
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=opt.fp16):
            # gaussian_models, _, opacity_list = generator.gaussian_generate(embeddings)
            gaussian_models = gaussians

        selected_index = gaussian_models[0]._opacity.squeeze() > 0.01
        xyz = gaussian_models[0]._xyz[selected_index]
        features_dc = gaussian_models[0]._features_dc[selected_index]

        torch.save(xyz, os.path.join(save_xyz_path))
        torch.save(features_dc, os.path.join(save_color_path))

if __name__ == '__main__':
    prompt = "a humanoid banana sitting at a desk doing homework."
    model_path = "workspace2/apple_banana_IF_lr5e-5_cbs4_opacity-2__scales2_rotations1_grid48/checkpoints/BrightDreamer_ep0030.pth"
    save_xyz_path = "test_xyz.pt"
    save_color_path= "test_color.pt"
    save_xyz_color(prompt,model_path,save_xyz_path,save_color_path)