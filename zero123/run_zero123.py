'''
conda activate zero123
cd stable-diffusion
python gradio_new.py 0
'''

import diffusers  # 0.12.1
import math
import fire
import gradio as gr
import lovely_numpy
import lovely_tensors
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import rich
import sys
import time
import argparse
import torch
from contextlib import nullcontext
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from einops import rearrange
from functools import partial
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import create_carvekit_interface, load_and_preprocess, instantiate_from_config
from lovely_numpy import lo
from omegaconf import OmegaConf
from PIL import Image
from rich import print
from transformers import AutoFeatureExtractor #, CLIPImageProcessor
from torch import autocast
from torchvision import transforms
import torch.nn.functional as F

_SHOW_DESC = True
_SHOW_INTERMEDIATE = False
# _SHOW_INTERMEDIATE = True

# _TITLE = 'Zero-Shot Control of Camera Viewpoints within a Single Image'
_TITLE = 'Zero-1-to-3: Zero-shot One Image to 3D Object'

# This demo allows you to generate novel viewpoints of an object depicted in an input image using a fine-tuned version of Stable Diffusion.
_DESCRIPTION = '''
This demo allows you to control camera rotation and thereby generate novel viewpoints of an object within a single image.
It is based on Stable Diffusion. Check out our [project webpage](https://zero123.cs.columbia.edu/) and [paper](https://arxiv.org/) if you want to learn more about the method!
Note that this model is not intended for images of humans or faces, and is unlikely to work well for them.
'''

_ARTICLE = 'See uses.md'


def load_model_from_config(config, ckpt, device, verbose=False):
    print(f'Loading model from {ckpt}')
    pl_sd = torch.load(ckpt, map_location='cpu')
    if 'global_step' in pl_sd:
        print(f'Global Step: {pl_sd["global_step"]}')
    sd = pl_sd['state_dict']
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print('missing keys:')
        print(m)
    if len(u) > 0 and verbose:
        print('unexpected keys:')
        print(u)

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def sample_model(input_im, model, sampler, precision, h, w, ddim_steps, n_samples, scale,
                 ddim_eta, x, y, z):
    precision_scope = autocast if precision == 'autocast' else nullcontext
    with precision_scope('cuda'):
        with model.ema_scope():
            c = model.get_learned_conditioning(input_im).tile(n_samples, 1, 1)
            T = torch.tensor([math.radians(x), math.sin(
                math.radians(y)), math.cos(math.radians(y)), z])
            T = T[None, None, :].repeat(n_samples, 1, 1).to(c.device)
            c = torch.cat([c, T], dim=-1)
            c = model.cc_projection(c)
            cond = {}
            cond['c_crossattn'] = [c]
            cond['c_concat'] = [model.encode_first_stage((input_im.to(c.device))).mode().detach().repeat(n_samples, 1, 1, 1)]
            if scale != 1.0:
                uc = {}
                uc['c_concat'] = [torch.zeros(n_samples, 4, 32, 32).to(c.device)]
                uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
            else:
                uc = None

            shape = [4, 32, 32]
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=cond,
                                             batch_size=n_samples,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta,
                                             x_T=None)
            # samples_ddim = F.interpolate(samples_ddim, (64, 64), mode='bilinear')
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = F.interpolate(x_samples_ddim, (h, w), mode='bicubic', antialias=True)
            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()

def preprocess_image(models, input_im, preprocess, h, w):
    '''
    :param input_im (PIL Image).
    :return input_im (H, W, 3) array in [0, 1].
    '''

    print('old input_im:', input_im.size)
    start_time = time.time()

    if preprocess:
        input_im = load_and_preprocess(models['carvekit'], input_im, h, w)
        input_im = (input_im / 255.0).astype(np.float32)
        # (H, W, 3) array in [0, 1].
    else:
        input_im = input_im.resize([h, w], Image.Resampling.LANCZOS)
        input_im = np.asarray(input_im, dtype=np.float32) / 255.0
        # (H, W, 4) array in [0, 1].

        # old method: thresholding background, very important
        # input_im[input_im[:, :, -1] <= 0.9] = [1., 1., 1., 1.]

        # new method: apply correct method of compositing to avoid sudden transitions / thresholding
        # (smoothly transition foreground to white background based on alpha values)
        alpha = input_im[:, :, 3:4]
        white_im = np.ones_like(input_im)
        input_im = alpha * input_im + (1.0 - alpha) * white_im

        input_im = input_im[:, :, 0:3]
        # (H, W, 3) array in [0, 1].

    print(f'Infer foreground mask (preprocess_image) took {time.time() - start_time:.3f}s.')
    print('new input_im:', lo(input_im))

    return input_im


def main_run(models, device,
             x=0.0, y=0.0, z=0.0,
             raw_im=None, preprocess=True,
             scale=3.0, n_samples=4, ddim_steps=50, ddim_eta=1.0,
             precision='fp32', h=256, w=256, sampler='ddim'):
    '''
    :param raw_im (PIL Image).
    '''
    
    raw_im.thumbnail([1536, 1536], Image.Resampling.LANCZOS)
    safety_checker_input = models['clip_fe'](raw_im, return_tensors='pt').to(device)
    (image, has_nsfw_concept) = models['nsfw'](
       images=np.ones((1, 3)), clip_input=safety_checker_input.pixel_values)
    print('has_nsfw_concept:', has_nsfw_concept)
    has_nsfw_concept = False # save time
    if np.any(has_nsfw_concept):
        print('NSFW content detected.')
        return None
    else:
        print('Safety check passed.')

    input_im = preprocess_image(models, raw_im, preprocess, h, w)

    show_in_im1 = (input_im * 255.0).astype(np.uint8)
    show_in_im2 = Image.fromarray(show_in_im1)

    input_im = transforms.ToTensor()(input_im).unsqueeze(0).to(device)
    input_im = input_im * 2 - 1
    input_im = transforms.functional.resize(input_im, [h, w])

    if sampler == 'ddim':
        sampler = DDIMSampler(models['turncam'])
    elif sampler == 'plms':
        sampler = PLMSSampler(models['turncam'])
        ddim_eta = 0
    else:
        print("sampler", sampler, "not supported")
        return None

    # used_x = -x  # NOTE: Polar makes more sense in Basile's opinion this way!
    used_x = x  # NOTE: Set this way for consistency.
    x_samples_ddim = sample_model(input_im, models['turncam'], sampler, precision, h, w,
                                  ddim_steps, n_samples, scale, ddim_eta, used_x, y, z)

    output_ims = []
    for x_sample in x_samples_ddim:
        x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        output_ims.append(Image.fromarray(x_sample.astype(np.uint8)))
    
    return (show_in_im2, output_ims)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some arguments for zero123.')
    parser.add_argument('--input-image', type=str,
                    help='input image path', required=True)
    parser.add_argument('--no-preprocess', action='store_true')
    parser.add_argument('--x', type=float, default=-40.0, 
                    help='polar angle')
    parser.add_argument('--y', type=float, default=-65.0, 
                    help='azimuth angle')
    parser.add_argument('--z', type=float, default=0.0, 
                    help='radius')
    parser.add_argument('--h', type=int, default=512, 
                    help='input height')
    parser.add_argument('--w', type=int, default=512, 
                    help='input width')
    parser.add_argument('--scale', type=float, default=3.0, 
                    help='scale')
    parser.add_argument('--ddim_steps', type=int, default=50, 
                    help='ddim_steps')
    parser.add_argument('--ddim_eta', type=float, default=1.0, 
                    help='ddim_eta')
    parser.add_argument('--precision', type=str, default='fp32', 
                    help='precision')
    parser.add_argument('--n-samples', type=int, default=4, 
                    help='number of generated samples')
    parser.add_argument('--device-idx', type=int, default=0, 
                    help='cuda device index')
    parser.add_argument('--ckpt', type=str, default='105000.ckpt', 
                    help='saved checkpoints')
    parser.add_argument('--config', type=str, default='configs/sd-objaverse-finetune-c_concat-256.yaml', 
                    help='model config')
    parser.add_argument('--sampler', type=str, default='ddim', 
                    help='sampler type, ddim or plms')

    args = parser.parse_args()

    device = f'cuda:{args.device_idx}'
    config = OmegaConf.load(args.config)

    # Instantiate all models beforehand for efficiency.
    models = dict()
    print('Instantiating LatentDiffusion...')
    models['turncam'] = load_model_from_config(config, args.ckpt, device=device)
    print('Instantiating Carvekit HiInterface...')
    models['carvekit'] = create_carvekit_interface()
    print('Instantiating StableDiffusionSafetyChecker...')
    models['nsfw'] = StableDiffusionSafetyChecker.from_pretrained(
       'CompVis/stable-diffusion-safety-checker').to(device)
    print('Instantiating AutoFeatureExtractor...')
    models['clip_fe'] = AutoFeatureExtractor.from_pretrained(
       'CompVis/stable-diffusion-safety-checker')

    result = main_run(
        models, device,
        args.x, args.y, args.z,
        raw_im = Image.open(args.input_image),
        preprocess = not args.no_preprocess,
        scale = args.scale,
        ddim_steps = args.ddim_steps,
        ddim_eta = args.ddim_eta,
        precision = args.precision,
        n_samples = args.n_samples,
        h = args.h,
        w = args.w,
        sampler = args.sampler,
    )

    if result:
        show_in_im, output_ims = result 
        show_in_im = show_in_im.save('/root/zero123/zero123/input_im_proc.png')
        for i, im in enumerate(output_ims):
            output_im = im.save(f'/root/zero123/zero123/output_im_{i}.png')
