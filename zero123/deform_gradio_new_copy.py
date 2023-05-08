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
import torch
import json, os
from contextlib import nullcontext
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from einops import rearrange
from functools import partial
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import create_carvekit_interface, load_and_preprocess, instantiate_from_config
from lovely_numpy import lo
from omegaconf import OmegaConf
from PIL import Image
from rich import print
from transformers import AutoFeatureExtractor #, CLIPImageProcessor
from torch import autocast
from torchvision import transforms
import pdb
# from r3m import load_r3m

_GPU_INDEX = 5
TEMPLATE_TO_CONDITION_LABEL = {"lifting": 0, "moving": 1, "poking": 2, "pulling": 3, "pushing": 4, "turning": 5}

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
                 ddim_eta, handpose):
    precision_scope = autocast if precision == 'autocast' else nullcontext
    with precision_scope('cuda'):
        with model.ema_scope():
            input_im_encoded = model.get_learned_conditioning(input_im).tile(n_samples, 1, 1)
            cond_handpose = torch.tensor([0,0,0,0,0,0]).to(input_im_encoded.device)
            cond_handpose[TEMPLATE_TO_CONDITION_LABEL[handpose]]=1
            c = torch.cat([input_im_encoded, cond_handpose.tile(n_samples, 1, 1)], dim=-1)
            c = model.cc_projection(c)
            cond = {}
            cond['c_crossattn'] = [c]
            c_concat = model.encode_first_stage((input_im.to(c.device))).mode().detach()
            cond['c_concat'] = [c_concat.repeat(n_samples, 1, 1, 1)]
            if scale != 1.0:
                uc = {}
                uc['c_concat'] = [torch.zeros(n_samples, 4, h // 8, w // 8).to(c.device)]
                uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
            else:
                uc = None

            shape = [4, h // 8, w // 8]
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=cond,
                                             batch_size=n_samples,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta,
                                             x_T=None)
            print(samples_ddim.shape)
            # samples_ddim = torch.nn.functional.interpolate(samples_ddim, 64, mode='nearest', antialias=False)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()


def preprocess_image(models, input_im, preprocess):
    '''
    :param input_im (PIL Image).
    :return input_im (H, W, 3) array in [0, 1].
    '''
    print('old input_im:', input_im.size)
    start_time = time.time()

    if preprocess:
        input_im = load_and_preprocess(models['carvekit'], input_im)
        input_im = (input_im / 255.0).astype(np.float32)
        # (H, W, 3) array in [0, 1].
    else:
        input_im = input_im.resize([256, 256], Image.Resampling.LANCZOS)
        input_im = np.asarray(input_im, dtype=np.float32) / 255.0
        # (H, W, 4) array in [0, 1].

        # old method: thresholding background, very important
        # input_im[input_im[:, :, -1] <= 0.9] = [1., 1., 1., 1.]

        # new method: apply correct method of compositing to avoid sudden transitions / thresholding
        # (smoothly transition foreground to white background based on alpha values)
        # alpha = input_im[:, :, 3:4]
        # white_im = np.ones_like(input_im)
        # input_im = alpha * input_im + (1.0 - alpha) * white_im

        # input_im = input_im[:, :, 0:3]
        # (H, W, 3) array in [0, 1].

    print(f'Infer foreground mask (preprocess_image) took {time.time() - start_time:.3f}s.')
    print('new input_im:', lo(input_im))

    return input_im
#96527
def run_demo():
    filename = "/proj/vondrick3/datasets/Something-Somethingv2/data/rawframes/202366/img_00009.jpg"
    filenameTarget = "/proj/vondrick3/datasets/Something-Somethingv2/data/rawframes/202366/img_00012.jpg"
    device_idx=_GPU_INDEX
    ckpt='/proj/vondrick3/sruthi/zero123/zero123/logs/good_2023-04-27T11-21-36_sd-somethingsomething-finetune/checkpoints/last.ckpt'
    config='configs/sd-somethingsomething-finetune.yaml'
    # print('sys.argv:', _GPU_INDEX)
    # if len(sys.argv) > 1:
    #     print('old device_idx:', device_idx)
    #     device_idx = int(sys.argv[1])
    #     print('new device_idx:', device_idx)
    save_path = "/".join(ckpt.split("/")[:-2])+"/"+ckpt.split("/")[-1][:-4]+"/"+"_".join(filename.split("/")[-2:])
    if not os.path.exists("/".join(save_path.split("/")[:-1])):
        os.mkdir("/".join(save_path.split("/")[:-1]))
    device = f'cuda:{device_idx}'
    config = OmegaConf.load(config)

    # Instantiate all models beforehand for efficiency.
    models = dict()
    print('Instantiating LatentDiffusion...')
    models['turncam'] = load_model_from_config(config, ckpt, device=device)
    print('Instantiating Carvekit HiInterface...')
    models['carvekit'] = create_carvekit_interface()
    input_im = preprocess_image(models, Image.open(filename), False)
    show_in_imI = Image.fromarray((input_im * 255.0).astype(np.uint8))
    show_in_imI.save(f"{save_path}_input.png")
    target_im = preprocess_image(models, Image.open(filenameTarget), False)
    show_in_imT = Image.fromarray((target_im * 255.0).astype(np.uint8))
    show_in_imT.save(f"{save_path}_target.png")

    input_im = transforms.ToTensor()(input_im).unsqueeze(0).to(device)
    input_im = input_im * 2 - 1
    input_im = transforms.functional.resize(input_im, [256,256])

    sampler = DDIMSampler(models['turncam'])
    x_samples_ddim = sample_model(input_im, models['turncam'], sampler, 'fp32', 256, 256,
                                    50, 4, 3.0, 1.0, 'pushing')

    output_ims = []
    for x_sample in x_samples_ddim:
        x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        output_ims.append(Image.fromarray(x_sample.astype(np.uint8)))
    
    print(output_ims)
    for i in range(len(output_ims)):
        output_ims[i].save(f"{save_path}_temp_sample{i}.png")

if __name__ == '__main__':
    # all_folders=os.listdir('/proj/vondrick3/datasets/Something-Somethingv2/data/rawframes')
    # labels = json.load(open('/proj/vondrick3/datasets/Something-Somethingv2/labels/train.json'))
    # file = open('trainlist.txt','w')
    # for x in labels:
    #     if x['id'] in all_folders and x['template'] in SET_OF_SSV2_LABELS :
    #         if len(os.listdir("/proj/vondrick3/datasets/Something-Somethingv2/data/rawframes/"+x['id'])[10:-10])>5:
    #             print('writing', x['id'])
    #             file.write(x['id']+"\n")
    # file.close()
    # labels = json.load(open('/proj/vondrick3/datasets/Something-Somethingv2/labels/validation.json'))
    # file = open('vallist.txt','w')
    # for x in labels:
    #     if x['id'] in all_folders and x['template'] in SET_OF_SSV2_LABELS :
    #         if len(os.listdir("/proj/vondrick3/datasets/Something-Somethingv2/data/rawframes/"+x['id'])[10:-10])>5:
    #             file.write(x['id']+"\n")
    # file.close()


    fire.Fire(run_demo)
