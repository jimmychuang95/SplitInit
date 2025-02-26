import torch
import argparse

from trainer import *
from generator.DreamInit import DreamInit
from generator.provider import GenerateCircleCameras, cameraList_from_RcamInfos
from diffusers import IFPipeline, StableDiffusionPipeline

# camera path generation
def generate_camera_path(num_view, opt):
    eval_cameras = GenerateCircleCameras(opt, num_view, render45=False)
    cameras = cameraList_from_RcamInfos(eval_cameras, 1.0, opt, SSAA=True)
    return cameras


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--prompt', type=str)
    parser.add_argument('--save_path', type=str, default='test')

    # network params
    parser.add_argument('--bound', type=float, default=1, help="assume the scene is bounded in box(-bound, bound)")
    parser.add_argument('--xyzres', action='store_true', help="xyzres")
    parser.add_argument('--free_distance', type=float, default=0.2)

    # render params
    parser.add_argument('--radius_range', type=float, nargs='*', default=[3.5, 3.5], help="training camera radius range")
    parser.add_argument('--theta_range', type=float, nargs='*', default=[45, 105], help="training camera range along the polar angles (i.e. up and down). See advanced.md for details.")
    parser.add_argument('--phi_range', type=float, nargs='*', default=[-180, 180], help="training camera range along the azimuth angles (i.e. left and right). See advanced.md for details.")
    parser.add_argument('--fovy_range', type=float, nargs='*', default=[20, 20], help="training camera fovy range")
    parser.add_argument('--default_radius', type=float, default=3.5, help="radius for the default view")
    parser.add_argument('--default_polar', type=float, default=90, help="polar for the default view")
    parser.add_argument('--default_azimuth', type=float, default=0, help="azimuth for the default view")
    parser.add_argument('--default_fovy', type=float, default=20, help="fovy for the default view")
    parser.add_argument('--angle_overhead', type=float, default=30, help="[0, angle_overhead] is the overhead region")
    parser.add_argument('--angle_front', type=float, default=60,
                        help="[0, angle_front] is the front region, [180, 180+angle_front] the back region, otherwise the side region.")



    parser.add_argument('--w', type=int, default=512, help="render width for NeRF in training")
    parser.add_argument('--h', type=int, default=512, help="render height for NeRF in training")
    parser.add_argument('--ema', action='store_true', help="load ema weights")

    ### others
    parser.add_argument('--fixed_rotations', type=float, default=1.0)
    parser.add_argument('--fixed_scales', type=float, default=-3.0)
    parser.add_argument('--grid_resolution', type=int, default=48)

    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_args()
    opt.xyzres = True
    opt.fp16 = True
    opt.image_h = opt.h
    opt.image_w = opt.w
    device = torch.device('cuda')
    opt.device = device

    generator = SemanticDreamer(opt).to(device)
    model_ckpt = torch.load(opt.model_path, map_location='cpu')
    generator.load_state_dict(model_ckpt['model'])
    if 'ema' in model_ckpt and opt.ema:
        ema = ExponentialMovingAverage(generator.parameters(), decay=0.99)
        ema.load_state_dict(model_ckpt['ema'])
        ema.copy_to()

    pipe = IFPipeline.from_pretrained(
        "/hpc2hdd/home/hli886/container_data2/project/3D-GGN/models--DeepFloyd--IF-I-XL-v1.0/snapshots/c03d510e9b75bce9f9db5bb85148c1402ad7e694",
        variant="fp16",
        torch_dtype=torch.float16)
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder.to(device)

    # pipe = StableDiffusionPipeline.from_pretrained("/mnt/project/generative_gaussian/models--stabilityai--stable-diffusion-2-1-base/snapshots/5ede9e4bf3e3fd1cb0ef2f7a3fff13ee514fdf06", variant="fp16", torch_dtype=torch.float16)
    # tokenizer = pipe.tokenizer
    # text_encoder = pipe.text_encoder.to(device)

    cameras = generate_camera_path(120, opt)

    generator.eval()

    inputs = tokenizer(opt.prompt, padding='max_length', max_length=77, truncation=True, add_special_tokens=True, return_tensors='pt')
    embeddings = text_encoder(inputs.input_ids.to(device))[0]
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=opt.fp16):
            gaussian_models = generator.gaussian_generate(embeddings)
        images = generator.render(gaussian_models, [cameras])['rgbs']
        images = images * 255
        images = images.permute(0, 2, 3, 1)
        
        save_path = opt.save_path+'/'+opt.prompt
        os.makedirs(save_path, exist_ok=True)
        imageio.mimwrite(os.path.join(save_path, opt.prompt + '.mp4'),
                         images.to(torch.uint8).cpu(), fps=25, quality=8,
                         macro_block_size=1)
        # output images to cooresponding folder
        os.makedirs(os.path.join(save_path, 'images'), exist_ok=True)
        for i in range(images.shape[0]):
            imageio.imsave(os.path.join(save_path, 'images', f'{i}.png'), images[i].to(torch.uint8).cpu().numpy())


        selected_index = gaussian_models[0]._opacity.squeeze() > 0.05
        xyz = gaussian_models[0]._xyz[selected_index]
        features_dc = gaussian_models[0]._features_dc[selected_index]

        torch.save(xyz, os.path.join('/mnt/xyz.pt'))
        torch.save(features_dc, os.path.join('/mnt/features_dc.pt'))
        print(xyz.shape)
        print(features_dc.shape)