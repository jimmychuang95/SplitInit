import os
import glob
import tqdm
import math
import imageio
import psutil
from pathlib import Path
import random
import shutil
import tensorboardX

import numpy as np

import time

import torch
import torch.optim as optim
from torchvision.utils import save_image

from rich.console import Console
from torch_ema import ExponentialMovingAverage

from packaging import version as pver
from save_xyz_color import save_xyz_color

from generator.gaussian_utils.gaussian_model import GaussianModel

def adjust_text_embeddings(embeddings, azimuth, opt):
    with torch.no_grad():
        text_z_list = []
        weights_list = []
        K = 0
        for b in range(len(azimuth)):
            text_z_, weights_ = get_pos_neg_text_embeddings(embeddings, azimuth[b], opt)
            K = max(K, weights_.shape[0])
            text_z_list.append(text_z_)
            weights_list.append(weights_)
        # Interleave text_embeddings from different dirs to form a batch
        text_embeddings = []
        for i in range(K):
            for text_z in text_z_list:
                # if uneven length, pad with the first embedding
                text_embeddings.append(text_z[i] if i < len(text_z) else text_z[0])
        text_embeddings = torch.stack(text_embeddings, dim=0) # torch.Size([12, 77, 4096])

        # Interleave weights from different dirs to form a batch
        weights = []
        for i in range(K):
            for weights_ in weights_list:
                weights.append(weights_[i] if i < len(weights_) else torch.zeros_like(weights_[0]))
        weights = torch.stack(weights, dim=0) # [B * K]
        return text_embeddings, weights

def get_pos_neg_text_embeddings(embeddings, azimuth_val, opt):
    if azimuth_val >= -90 and azimuth_val < 90:
        if azimuth_val >= 0:
            r = 1 - azimuth_val / 90
        else:
            r = 1 + azimuth_val / 90
        start_z = embeddings['front']
        end_z = embeddings['side']
        # if random.random() < 0.3:
        #     r = r + random.gauss(0, 0.08)
        pos_z = r * start_z + (1 - r) * end_z
        text_z = torch.cat([pos_z, embeddings['front'], embeddings['side']], dim=0) # torch.Size([3, 77, 4096])
        if r > 0.8:
            front_neg_w = 0.0
        else:
            front_neg_w = math.exp(-r * opt.front_decay_factor) * opt.negative_w
        if r < 0.2:
            side_neg_w = 0.0
        else:
            side_neg_w = math.exp(-(1-r) * opt.side_decay_factor) * opt.negative_w

        weights = torch.tensor([1.0, front_neg_w, side_neg_w])
    else:
        if azimuth_val >= 0:
            r = 1 - (azimuth_val - 90) / 90
        else:
            r = 1 + (azimuth_val + 90) / 90
        start_z = embeddings['side']
        end_z = embeddings['back']
        # if random.random() < 0.3:
        #     r = r + random.gauss(0, 0.08)
        pos_z = r * start_z + (1 - r) * end_z
        text_z = torch.cat([pos_z, embeddings['side'], embeddings['front']], dim=0)
        front_neg_w = opt.negative_w
        if r > 0.8:
            side_neg_w = 0.0
        else:
            side_neg_w = math.exp(-r * opt.side_decay_factor) * opt.negative_w / 2

        weights = torch.tensor([1.0, side_neg_w, front_neg_w])
    return text_z, weights.to(text_z.device)

def get_CPU_mem():
    return psutil.Process(os.getpid()).memory_info().rss /1024**3


def get_GPU_mem():
    num = torch.cuda.device_count()
    mem, mems = 0, []
    for i in range(num):
        mem_free, mem_total = torch.cuda.mem_get_info(i)
        mems.append(int(((mem_total - mem_free)/1024**3)*1000)/1000)
        mem += mems[-1]
    return mem, mems

def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')

def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True


@torch.jit.script
def linear_to_srgb(x):
    return torch.where(x < 0.0031308, 12.92 * x, 1.055 * x ** 0.41666 - 0.055)


@torch.jit.script
def srgb_to_linear(x):
    return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


class Trainer(object):
    def __init__(self,
        argv, # command line args
        name, # name of this experiment
        opt, # extra conf
        model, # network
        guidance, # guidance network
        criterion=None, # loss function, if None, assume inline implementation in train_step
        optimizer=None, # optimizer
        ema_decay=None, # if use EMA, set the decay
        lr_scheduler=None, # scheduler
        metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
        local_rank=0, # which GPU am I
        world_size=1, # total num of GPUs
        device=None, # device to use, usually setting to None is OK. (auto choose device)
        mute=False, # whether to mute all print
        fp16=False, # amp optimize level
        workspace='workspace', # workspace to save logs & ckpts
        best_mode='min', # the smaller/larger result, the better
        use_loss_as_metric=True, # use loss as the first metric
        report_metric_at_train=False, # also report metrics at training
        use_checkpoint="latest", # which ckpt to use at init time
        use_tensorboardX=True, # whether to use tensorboard for logging
        scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
        ):

        self.argv = argv
        self.name = name
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()

        self.model = model

        # guide model
        self.guidance = guidance

        self.optimizer = optimizer(self.model)


        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1) # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
            print(ema_decay)
            # self.log(f"[INFO] Using EMA with decay: {ema_decay}.")
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.total_train_t = 0
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [], # metrics[0], or valid_loss
            "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)

        self.log(f'[INFO] Cmdline: {self.argv}')
        self.log(f'[INFO] opt: {self.opt}')
        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else: # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

        self.log(self.model)


    def __del__(self):
        if self.log_ptr:
            self.log_ptr.close()


    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute:
                #print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr:
                print(*args, file=self.log_ptr)
                self.log_ptr.flush() # write immediately to file

    ### ------------------------------

    def train_step(self, data, global_step, save_guidance_path: Path = None):
        """
            Args:
                save_guidance_path: an image that combines the NeRF render, the added latent noise,
                    the denoised result and optionally the fully-denoised image.
        """
        B = len(data)
        C = self.opt.c_batch_size
        text_zs = []
        cameras = []
        for i in range(B):
            text_zs.append(data[i]['text_embeddings']['default'])  # incorporate each word
            cameras.append(data[i]['cameras'])
        text_zs = torch.cat(text_zs, dim=0)  # [B, 77, 1024]

        outputs = self.model(text_zs, cameras)  # must pass through ddp.forward, connot directly use ddp.module
        pred_images = outputs['rgbs']  # [B*C, 3, H, W]
        as_latent = False

        loss = 0
        for i in range(B):
            azimuth = []
            for j in range(C):
                azimuth.append(data[i]['cameras'][j].delta_azimuth)

            text_embeddings = data[i]['text_embeddings']
            images = pred_images[i * C: (i + 1) * C]
            if 'SD' in self.guidance:
                text_z = [text_embeddings['uncond']] * C
                if self.opt.perpneg:

                    text_z_comp, weights = adjust_text_embeddings(text_embeddings, azimuth, self.opt)
                    text_z.append(text_z_comp)

                else:
                    for c in range(C):
                        if azimuth[c] >= -90 and azimuth[c] < 90:
                            if azimuth[c] >= 0:
                                r = 1 - azimuth[c] / 90
                            else:
                                r = 1 + azimuth[c] / 90
                            start_z = text_embeddings['front']
                            end_z = text_embeddings['side']
                        else:
                            if azimuth[c] >= 0:
                                r = 1 - (azimuth[c] - 90) / 90
                            else:
                                r = 1 + (azimuth[c] + 90) / 90
                            start_z = text_embeddings['side']
                            end_z = text_embeddings['back']
                        text_z.append(r * start_z + (1 - r) * end_z)

                text_z = torch.cat(text_z, dim=0)
                if self.opt.perpneg:
                    loss = loss + self.guidance['SD'].train_step_perpneg(text_z, weights, images, as_latent=as_latent,
                                                                         guidance_scale=self.opt.guidance_scale,
                                                                         grad_scale=self.opt.lambda_guidance,
                                                                         save_guidance_path=save_guidance_path)
                else:
                    loss = loss + self.guidance['SD'].train_step(text_z, images, as_latent=as_latent,
                                                                 guidance_scale=self.opt.guidance_scale,
                                                                 grad_scale=self.opt.lambda_guidance,
                                                                 save_guidance_path=save_guidance_path)


            if 'IF' in self.guidance:
                with torch.no_grad():
                    text_z = [text_embeddings['uncond']] * C
                    if self.opt.perpneg:
                        text_z_comp, weights = adjust_text_embeddings(text_embeddings, azimuth, self.opt)
                        text_z.append(text_z_comp)

                    else:
                        for c in range(C):
                            if azimuth[c] >= -90 and azimuth[c] < 90:
                                if azimuth[c] >= 0:
                                    r = 1 - azimuth[c] / 90
                                else:
                                    r = 1 + azimuth[c] / 90
                                start_z = text_embeddings['front']
                                end_z = text_embeddings['side']
                            else:
                                if azimuth[c] >= 0:
                                    r = 1 - (azimuth[c] - 90) / 90
                                else:
                                    r = 1 + (azimuth[c] + 90) / 90
                                start_z = text_embeddings['side']
                                end_z = text_embeddings['back']
                            text_z.append(r * start_z + (1 - r) * end_z)

                    text_z = torch.cat(text_z, dim=0) # torch.Size([16, 77, 4096])
                if self.opt.perpneg:
                    loss = loss + self.guidance['IF'].train_step_perpneg(text_z, weights, images, global_step, prompt=self.opt.prompt, workspace=self.workspace,
                                                                        guidance_scale=self.opt.guidance_scale,
                                                                        grad_scale=self.opt.lambda_guidance)
                else:
                    loss = loss + self.guidance['IF'].train_step(text_z, images,
                                                                guidance_scale=self.opt.guidance_scale,
                                                                grad_scale=self.opt.lambda_guidance)


            if 'IF2' in self.guidance:
                text_z = [text_embeddings['uncond']] * C
                if self.opt.perpneg:

                    text_z_comp, weights = adjust_text_embeddings(text_embeddings, azimuth, self.opt)
                    text_z.append(text_z_comp)

                else:
                    for c in range(C):
                        if azimuth[c] >= -90 and azimuth[c] < 90:
                            if azimuth[c] >= 0:
                                r = 1 - azimuth[c] / 90
                            else:
                                r = 1 + azimuth[c] / 90
                            start_z = text_embeddings['front']
                            end_z = text_embeddings['side']
                        else:
                            if azimuth[c] >= 0:
                                r = 1 - (azimuth[c] - 90) / 90
                            else:
                                r = 1 + (azimuth[c] + 90) / 90
                            start_z = text_embeddings['side']
                            end_z = text_embeddings['back']
                        text_z.append(r * start_z + (1 - r) * end_z)

                text_z = torch.cat(text_z, dim=0)
                if self.opt.perpneg:
                    loss = loss + self.guidance['IF2'].train_step_perpneg(text_z, weights, images,
                                                                          guidance_scale=self.opt.guidance_scale,
                                                                          grad_scale=self.opt.lambda_guidance)
                else:
                    loss = loss + self.guidance['IF2'].train_step(text_z, images,
                                                                  guidance_scale=self.opt.guidance_scale,
                                                                  grad_scale=self.opt.lambda_guidance)
        return images, None, loss

    def post_train_step(self):

        # unscale grad before modifying it!
        # ref: https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
        self.scaler.unscale_(self.optimizer)

        # clip grad
        if self.opt.grad_clip >= 0:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.opt.grad_clip)

    def eval_step(self, data):
        with torch.no_grad():
            B = len(data)
            C = self.opt.c_batch_size
            text_zs = []
            cameras = []
            for i in range(B):
                text_zs.append(data[i]['text_embeddings']['default'])  # incorporate each word
                cameras.append(data[i]['cameras'])
            text_zs = torch.cat(text_zs, dim=0)  # [B, 77, 1024]
            outputs, all_top_k_indices, opacity_list, output_gaussians = self.model(text_zs, cameras)

            prompt_objects = "potted plant skateboard"
            layer_class_outputs = []
            for b in range(B):
                opacities = opacity_list[b]
                selected_indices = [i for i, opacity in enumerate(opacities) if opacity > 0.01]
                if not selected_indices:
                    print(f"========= No valid indices =========")
                
                else:
                    print(f'============== total / valid gaussians: {len(opacities)} / {len(selected_indices)} ==============')

                    k = all_top_k_indices[0][b].shape[1]
                    if 'IF' in self.guidance:
                        clean_prompt = self.guidance['IF'].pipe._text_preprocessing(self.opt.prompt, clean_caption=False)
                        inputs = self.guidance['IF'].tokenizer(clean_prompt, padding='max_length', max_length=77, truncation=True, add_special_tokens=True, return_tensors='pt')
                    elif 'SD' in self.guidance:
                        inputs = self.guidance['SD'].tokenizer(self.opt.prompt, padding='max_length', max_length=self.guidance['SD'].tokenizer.model_max_length, return_tensors='pt')

                    appear_index_text = []
                    desired_token = [1, 8, 7, 6, 9]
                    class_color_mapping = {
                        1: (1.0, 0.0, 0.0),  # red - astronaut
                        8: (0.0, 1.0, 0.0),  # green - o
                        7: (0.0, 0.0, 1.0),  # blue - gar
                        6: (1.0, 1.0, 0.0),  # yellow - kan
                        9: (1.0, 0.0, 1.0),  # purple - o
                        #10: (0.0, 1.0, 1.0),  # cyan - airplane
                    }

                    
                    for i in range (len(all_top_k_indices)):
                        print(f"============== Layer {i} top {k} attention scores ==============")
                        max_indices = np.array(all_top_k_indices[i][b].cpu()) # [110592, 5]

                        filtered_indices = max_indices[selected_indices] # [valid gaussians num, 5]
                        
                        filtered_indices_tensor = torch.tensor(filtered_indices, dtype=torch.long, device=self.device).view(-1)
                        counts = torch.bincount(filtered_indices_tensor)
                        values = torch.arange(len(counts), device=self.device)
                        results = list(zip(values.tolist(), counts.tolist()))
                        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

                        for value, count in sorted_results:
                            if count > 0:
                                if 'IF' in self.guidance:
                                    decoded_text = self.guidance['IF'].tokenizer.decode(inputs.input_ids[0][value], skip_special_tokens=False)
                                elif 'SD' in self.guidance:
                                    decoded_text = self.guidance['SD'].tokenizer.decode(inputs.input_ids[0][value], skip_special_tokens=False)

                                if (value, decoded_text) not in appear_index_text and decoded_text not in ["<pad>", "</s>", ""]:
                                    appear_index_text.append((value, decoded_text))

                                # if decoded_text.strip() and decoded_text not in ["<pad>", "</s>"]:
                                print(f"index {value} -> token {inputs.input_ids[0][value]} -> {decoded_text} 出現 {count} 次")

                        gaussian_classes = np.full((max_indices.shape[0],), None, dtype=object)
                        for gaussian_idx in range(max_indices.shape[0]):
                            gaussian_top_k = max_indices[gaussian_idx]
                            gaussian_class = next((idx for idx in gaussian_top_k if idx in desired_token), None)
                            gaussian_classes[gaussian_idx] = gaussian_class

                        print(f'gaussian_classes shape: {gaussian_classes.shape}')

                        gaussian = GaussianModel(0)
                        gaussian._xyz = output_gaussians[b]._xyz
                        gaussian._opacity = output_gaussians[b]._opacity
                        gaussian._rotation = output_gaussians[b]._rotation
                        gaussian._scaling = output_gaussians[b]._scaling

                        colors = []
                        for gaussian_idx in range(len(output_gaussians[b]._xyz)):
                            gaussian_class = gaussian_classes[gaussian_idx]  # 假設你已有 gaussian_classes 的對應類別
                            color = class_color_mapping.get(gaussian_class, (0.5, 0.5, 0.5))  # 預設為灰色
                            colors.append(color)
                        gaussian._features_dc = torch.tensor(colors, dtype=torch.float32, device=self.device).unsqueeze(1)
                        print(appear_index_text)
                        class_outputs = self.model.render([gaussian], [cameras[b]])['rgbs']
                        layer_class_outputs.append(class_outputs)
                        

                # print(f'output_gaussians feaure dc: {output_gaussians[0]._features_dc.shape}')
                
                # new_oapcity = output_gaussians[0]._opacity
                # for i in range(len(output_gaussians[0]._opacity)):
                #     if output_gaussians[0]._features_dc[i][0][0] < 0.2:
                #         new_oapcity[i] = -1000
                
                # plane_gaussian = GaussianModel(0)
                # plane_gaussian._xyz = output_gaussians[0]._xyz
                # plane_gaussian._opacity = new_oapcity
                # plane_gaussian._rotation = output_gaussians[0]._rotation
                # plane_gaussian._scaling = output_gaussians[0]._scaling
                # plane_gaussian._features_dc = output_gaussians[0]._features_dc

                # plane_outputs = self.model.render([plane_gaussian], [cameras[b]])['rgbs']
                # layer_class_outputs.append(plane_outputs)

                # model_path = os.path.join(self.workspace, 'checkpoints', f"DreamInit_ep{self.epoch:04d}.pth")
                # save_xyz_color(self.opt, self.opt.prompt, model_path, self.workspace + '/xyz_split.pt', self.workspace + '/color_split.pt', [plane_gaussian])
            

            pred_images = outputs['rgbs']  # [B*C, 3, H, W]

            return pred_images, layer_class_outputs, output_gaussians

    def train(self, train_loader, valid_loader, test_loader, max_epochs):

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        start_t = time.time()

        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch

            if self.epoch == 1:
                self.epoch = 0
                if self.local_rank == 0:
                    self.evaluate_one_epoch(valid_loader)
                # self.save_checkpoint()
                self.epoch = epoch

            self.train_one_epoch(train_loader, max_epochs)

            if self.epoch % self.opt.eval_interval == 0:
                if self.local_rank == 0:
                    self.evaluate_one_epoch(valid_loader)
                if self.epoch % self.opt.save_interval == 0:
                    self.save_checkpoint()
                    # self.save_checkpoint()
                    # pass 

            if self.epoch % self.opt.test_interval == 0:
                self.test(test_loader)

        end_t = time.time()

        self.total_train_t = end_t - start_t + self.total_train_t

        self.log(f"[INFO] training takes {(self.total_train_t)/ 60:.4f} minutes.")

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader, name=None):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name)
        self.use_tensorboardX = use_tensorboardX

    def test(self, loader):
        torch.cuda.empty_cache()
        self.log(f"++> Test {self.workspace} at epoch {self.epoch} ...")

        save_path = self.workspace + '/results'
        os.makedirs(save_path, exist_ok=True)

        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                             bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0

            for data in loader:
                for i in range(len(data)):
                    self.local_step += 1

                    with torch.cuda.amp.autocast(enabled=self.fp16):
                        preds, layer_class_outputs, output_gaussians = self.eval_step(data)
                        preds = preds * 255
                        preds = preds.permute(0, 2, 3, 1)
                        imageio.mimwrite(os.path.join(save_path, f"ep{self.epoch:05d}_{data[i]['text']}.mp4"),
                                         preds.to(torch.uint8).cpu(), fps=25, quality=8,
                                         macro_block_size=1)
                        
                        # save layer gaussian text embeddings video
                        # for l in range(len(layer_class_outputs)):
                        #     class_outputs = layer_class_outputs[l] * 255
                        #     class_outputs = class_outputs.permute(0, 2, 3, 1)
                        #     imageio.mimwrite(os.path.join(save_path, f"ep{self.epoch:05d}_layer{l}.mp4"),
                        #                     class_outputs.to(torch.uint8).cpu(), fps=25, quality=8,
                        #                     macro_block_size=1)
                        
                        


            
        
        if self.epoch > 0:
            model_path = os.path.join(self.workspace, 'checkpoints', f"DreamInit_ep{self.epoch:04d}.pth")
            save_xyz_path = os.path.join(self.workspace, f'xyz_{self.epoch:04d}.pt')
            save_color_path = os.path.join(self.workspace, f'color_{self.epoch:04d}.pt')
            save_xyz_color(self.opt, self.opt.prompt,model_path,save_xyz_path,save_color_path, output_gaussians)

        if self.local_rank == 0:
            pbar.close()

        if self.ema is not None:
            self.ema.restore()



        self.log(f"Test Finished.")

    def train_one_epoch(self, loader, max_epochs):
        self.log(f"==> [{time.strftime('%Y-%m-%d_%H-%M-%S')}] Start Training {self.workspace} Epoch {self.epoch}/{max_epochs}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")
        torch.cuda.empty_cache()

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        if self.opt.save_guidance:
            save_guidance_folder = Path(self.workspace) / 'guidance'
            save_guidance_folder.mkdir(parents=True, exist_ok=True)

        for data in loader:

            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                if self.opt.save_guidance and (self.global_step % self.opt.save_guidance_interval == 0):
                    save_guidance_path = save_guidance_folder / f'step_{self.global_step:07d}.png'
                else:
                    save_guidance_path = None
                pred_rgbs, pred_depths, loss = self.train_step(data, global_step=self.global_step, save_guidance_path=save_guidance_path)

            self.scaler.scale(loss).backward()
            self.post_train_step()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # if self.scheduler_update_every_step:
            #     self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.local_rank == 0:

                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), rank={self.local_rank}, lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                pbar.update(loader.batch_size)


        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        cpu_mem, gpu_mem = get_CPU_mem(), get_GPU_mem()[0]
        self.log(f"==> [{time.strftime('%Y-%m-%d_%H-%M-%S')}] Finished Epoch {self.epoch}/{max_epochs}. CPU={cpu_mem:.1f}GB, GPU={gpu_mem:.1f}GB.")
        torch.cuda.empty_cache()

    def evaluate_one_epoch(self, loader, name=None):
        torch.cuda.empty_cache()
        self.log(f"++> Evaluate {self.workspace} at epoch {self.epoch} ...")

        if name is None:
            name = f'{self.name}_ep{self.epoch:05d}'

        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                             bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0

            save_img = []
            for data in loader:
                self.local_step += 1

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, _, _ = self.eval_step(data)
                    preds = preds.permute(0, 2, 3, 1)
                    save_img.append(preds)

            save_img = torch.cat(save_img, dim=0).permute(0, 3, 1, 2).contiguous().cpu()
            save_path = os.path.join(self.workspace, 'validation', f'{name}.png')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_image(save_img, save_path, nrows=save_img.shape[0] // 10)

        if self.local_rank == 0:
            pbar.close()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def save_checkpoint(self):

        if self.local_rank == 0:
            # name = f'{self.name}_ep{self.epoch:04d}'
            name = f'{self.name}_ep{self.epoch:04d}'

            state = {'epoch': self.epoch,
                     'global_step': self.global_step,
                     'optimizer': self.optimizer.state_dict(),
                     'lr_scheduler': self.lr_scheduler.state_dict(),
                     'scaler': self.scaler.state_dict()}

            if self.opt.num_gpus > 1:
                state['model'] = self.model.module.state_dict()
            else:
                state['model'] = self.model.state_dict()

            if self.ema is not None:
                state['ema'] = self.ema.state_dict()

            file_path = f"{name}.pth"
            torch.save(state, os.path.join(self.ckpt_path, file_path))

    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return
        if self.opt.num_gpus > 1:
            missing_keys, unexpected_keys = self.model.module.load_state_dict(checkpoint_dict['model'], strict=False)
        else:
            missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        else:
            self.log(f"[INFO] no missing keys.")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")
        else:
            self.log(f"[INFO] no unexpected keys.")

        if self.ema is not None and 'ema' in checkpoint_dict:
            try:
                self.ema.load_state_dict(checkpoint_dict['ema'])
                self.log("[INFO] loaded EMA.")
            except:
                self.log("[WARN] failed to loaded EMA.")

        if model_only:
            return

        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")

        if self.optimizer and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
                self.optimizer.param_groups[0]['lr'] = self.opt.lr
                self.log(len(self.optimizer.param_groups))
            except:
                self.log("[WARN] Failed to load optimizer.")

        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")

        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")
