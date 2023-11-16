import torch
from torchvision.transforms.functional import to_pil_image
import numpy as np

from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler, DSDPipeline
import glob
import os
from PIL import Image
import PIL
from tqdm import tqdm
import argparse
import json
import random


class DSD_demo_Model():
    def __init__(self,
        model_path = None,
        model_id = "stabilityai/stable-diffusion-2-1-base",
        **kwargs,
    ):
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        model = DSDPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float32)
        if model_path is not None:
            model.unet.load_attn_procs(model_path)
        self.model = model


    def image_preprocess(self, image):
        w, h = image.size
        w, h = map(lambda x: x - x % 32, (w, h)) 
        image = image.resize((w, h), resample=PIL.Image.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        image = image.squeeze(0)
        return 2.0 * image - 1.0


    def __call__(self, text, img, **kwargs):
        processed_img = self.image_preprocess(img).cuda()
        score = 0.1*self.model(prompt=list(text), image=processed_img, guidance_scale=0, sampling_steps=50, layer_mask = None, sampled_timestep = None, level = None)
        return score

    


def DSD_Model(accelerator, args = None):
    if args.version == 'sdxl':
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        model = DSDPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    elif args.version == '2.1':
        if args.model:
            model_id = args.model
        else:
            model_id = "stabilityai/stable-diffusion-2-1-base"
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        model = DSDPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
    elif args.version == '2.1-v':
        model_id = "stabilityai/stable-diffusion-2-1"
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        model = DSDPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
    elif args.version == '1.5':
        model_id = "stabilityai/stable-diffusion-v1-5"
        model = DSDPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    model = model.to(accelerator.device)
    if args.output_dir != '':
        model.unet.load_attn_procs(args.output_dir)
    return model