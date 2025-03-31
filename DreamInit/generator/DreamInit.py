import torch
import torch.nn as nn
from .generator import Generator
from generator.gaussian_utils.gaussian_model import GaussianModel
from generator.gaussian_utils.gaussian_renderer import render
import numpy as np

from gsplat import rasterization
import math

C0 = 0.28209479177387814
def RGB2SH(rgb):
    return (rgb - 0.5) / C0

class PipelineParams:
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False

class DreamInit(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt=opt
        self.sh_degree = 0
        self.img_channel = 3 + 1 + 4 + 3 + 3 * (self.sh_degree + 1) ** 2
        self.grid_resolution = self.opt.grid_resolution
        self.bound = 1.0
        # self.free_distance = self.opt.free_distance
        self.generator = Generator(opt=opt, hidden_dim=self.opt.hidden_dim)  # opacity and color
        self.pp = PipelineParams()
        self.register_buffer('background', torch.ones(3))



        xyz = torch.stack(torch.meshgrid(torch.arange(-self.bound, self.bound, self.bound*2/self.grid_resolution),
                                                                         torch.arange(-self.bound, self.bound, self.bound*2/self.grid_resolution),
                                                                         torch.arange(-self.bound, self.bound, self.bound*2/self.grid_resolution)), dim=3).reshape(-1,3).contiguous()


        self.register_buffer('xyz', xyz)

        self.fixed_rotations = self.opt.fixed_rotations
        self.fixed_scales = self.opt.fixed_scales


    def gaussian_generate(self, text_embeddings):

        B = text_embeddings.shape[0]
        input = self.xyz.unsqueeze(0).repeat(B, 1, 1)


        gaussians_property, gtf_attns = self.generator(input, text_embeddings)

        gaussians_property = gaussians_property.to(torch.float32)
        gaussian_list = []  # (B,)
        opacity_list = []
        for i in range(B):

            gaussian = GaussianModel(self.sh_degree)
            gaussian._xyz = self.xyz
            gaussian._opacity = gaussians_property[i, :, 0:1]
            gaussian._rotation = torch.tensor([self.fixed_rotations,0,0,0]).repeat(gaussians_property[i].shape[0], 1).to("cuda")  # fixed
            gaussian._scaling = torch.tensor([self.fixed_scales]*3).repeat(gaussians_property[i].shape[0], 1).to("cuda")  # fixed
            gaussian._features_dc = (3.545*torch.sigmoid(gaussians_property[i, :, 1:4].reshape(-1, 1, 3)) - 1.7725)
            gaussian._features_rest = gaussians_property[i, :, 4:].reshape(-1, 15, 3)
            gaussian_list.append(gaussian)
            opacity_list.append(gaussian._opacity)

        if self.training:
            return gaussian_list, gtf_attns
        else:
            return gaussian_list, gtf_attns, opacity_list

    def render(self, gaussians, views):
        B = len(gaussians)
        C = len(views[0])
        rgbs = []
        for i in range(B):
            gaussian = gaussians[i]
            for j in range(C):
                view = views[i][j]
                render_pkg = render(view, gaussian, self.pp, self.background)
                rgb = render_pkg['render']
                rgbs.append(rgb)
        rgbs = torch.stack(rgbs, dim=0)
        return {'rgbs': rgbs}
    
    def calculate_K(self, width, height, FoVx, FoVy, device):
        fx = 0.5 * width / math.tan(0.5 * FoVx)
        fy = 0.5 * height / math.tan(0.5 * FoVy)
        cx = width / 2
        cy = height / 2

        K = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0,  1]
        ], device=device)

        return K
    
    def render_gsplat(self, gaussians, views, sh_degree=None, background=None, image_width=None, image_height=None):
        B = len(gaussians)
        C = len(views[0])
        
        rgbs = []
        for i in range(B):
            gaussian = gaussians[i]
            for j in range(C):
                view = views[i][j]

                if image_width is not None and image_height is not None:
                    K = self.calculate_K(image_width, image_height, view.FoVx, view.FoVy, device='cuda')
                    width = image_width
                    height = image_height
                else:
                    K = view.K
                    width = view.image_width
                    height = view.image_height

                output, _, _ = rasterization(
                    gaussian.get_xyz,
                    gaussian.get_rotation,
                    gaussian.get_scaling,
                    gaussian.get_opacity.squeeze(-1),
                    gaussian.get_features,
                    view.viewmat[None],
                    K[None],
                    width=width,
                    height=height,
                    sh_degree=sh_degree,
                    backgrounds=background,
                )
                rgb = output.permute(0, 3, 1, 2).squeeze(0)
                rgbs.append(rgb)
        rgbs = torch.stack(rgbs, dim=0)
        return {'rgbs': rgbs}


    def forward(self, text_zs, cameras):
        background = torch.ones((1, 3)).to("cuda")
        if self.training:
            gaussians, gtf_attns = self.gaussian_generate(text_zs)
            # outputs = self.render(gaussians, cameras)
            outputs = self.render_gsplat(gaussians, cameras, sh_degree=0, background=background)
            return outputs, gtf_attns, gaussians
        else:
            gaussians, gtf_attns, opacity_list = self.gaussian_generate(text_zs)
            # outputs = self.render(gaussians, cameras)
            outputs = self.render_gsplat(gaussians, cameras, sh_degree=0, background=background)
            return outputs, gtf_attns, opacity_list, gaussians

    def get_params(self, lr):

        params = [
            {'params': self.parameters(), 'lr': lr},
        ]
        return params