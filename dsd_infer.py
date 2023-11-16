import torch
from torchvision.transforms.functional import to_pil_image
import numpy as np

from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler
import glob
import os
from PIL import Image
from tqdm import tqdm
import argparse
import json
import random

from model import DSD_Model
from custom_datasets import get_dataset
from torch.utils.data import DataLoader
from eval_score import evaluate_scores
import csv
from accelerate import Accelerator
import cProfile
import re
import copy
import datetime
import logging


def next_available_dirpath(target_dir):
    base_name = target_dir
    index = 1
    while os.path.exists(target_dir):
        target_dir = f"{base_name}_{index}"
        index += 1
    index -= 2
    target_dir = f"{base_name}_{index}"
    return target_dir


def sanitize_text(text):
    text = re.sub(r'[^\w\s]', '', text)  
    text = text.replace(' ', '_')  
    return text[:50]  
    

def average_score(tensor_lists):
    total_mean_sum = 0
    total_tensors = sum(len(inner_list) for inner_list in tensor_lists)
    
    for inner_list in tensor_lists:
        for tensor_container in inner_list:
            tensor = tensor_container[0]  
            total_mean_sum += torch.mean(tensor).item()

    return total_mean_sum / total_tensors



def rename_image_based_on_score(score, img_name, img_idx, text_folder_path):
    img_name = img_name[0].split('/')[-1]
    
    if img_idx == 0:
        img_path = os.path.join(text_folder_path, 'positive', os.path.basename(img_name))
        folder = 'positive'
    else:
        img_path = os.path.join(text_folder_path, 'negative', os.path.basename(img_name))
        folder = 'negative'
    
    if os.path.exists(img_path):
        new_img_name = f"{os.path.splitext(img_name)[0]}_score_{score}{os.path.splitext(img_name)[1]}"
        new_img_path = os.path.join(text_folder_path, folder, new_img_name)
        os.rename(img_path, new_img_path)



def compute_score(i, args, batch, model, save_path='./saved_images'):
    imgs, texts = batch[0], batch[1]
    imgs, imgs_resize = imgs[0], imgs[1]
    imgs_resize = [img.cuda() for img in imgs_resize]

    scores = []
    for txt_idx, text in enumerate(texts):
        sanitized_text = sanitize_text(text[0])  
        text_folder_path = os.path.join(save_path, sanitized_text)
        # text_folder_path = next_available_dirpath(text_folder_base) 
        for img_idx, (img_name, resized_img) in enumerate(zip(imgs, imgs_resize)):
            if len(resized_img.shape) == 3:
                resized_img = resized_img.unsqueeze(0)
            
            print(f'Batch size {args.batchsize}, Batch {i}, Text {txt_idx}, Image {img_idx}')
            score = model(prompt=list(text), image=resized_img, guidance_scale=args.guidance_scale, sampling_steps=args.sampling_time_steps, use_bias = args.bias, layer_mask = None, sampled_timestep = None, level = None)

            scores.append(score)
            # rename_image_based_on_score(score, img_name[0], img_idx, text_folder_path)
            
            

    scores = torch.stack(scores).permute(1, 0) if args.batchsize > 1 else torch.stack(scores).unsqueeze(0)

    return scores




def main(args):

    accelerator = Accelerator()
    
    dataset = get_dataset(args.val_data, 'data', transform=None)

    dataloader = DataLoader(dataset, batch_size=args.batchsize, shuffle=False, num_workers=0)

    model = DSD_Model(accelerator, args)

    model, dataloader = accelerator.prepare(model, dataloader)


    r1s = []
    r5s = []
    metrics = []
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        scores = compute_score(i, args, batch, model)
        scores = scores.contiguous()
        accelerator.wait_for_everyone()
        scores = accelerator.gather(scores)
        batch[-1] = accelerator.gather(batch[-1])
        if accelerator.is_main_process:  
            if args.val_data == 'Refcocog':
                r1,r5 = evaluate_scores(args, scores, batch)
                r1s += r1
                r5s += r5
                r1 = sum(r1s) / len(r1s)
                r5 = sum(r5s) / len(r5s)
                logging.info(f'Batch {i + 1}, R@1: {r1}, R@5: {r5}, Sequence length: {len(r1s)}')
            elif args.val_data == 'winoground':
                text_scores, img_scores, group_scores = evaluate_scores(args, scores, batch)
                metrics += list(zip(text_scores, img_scores, group_scores))
                text_score = sum([m[0] for m in metrics]) / len(metrics)
                img_score = sum([m[1] for m in metrics]) / len(metrics)
                group_score = sum([m[2] for m in metrics]) / len(metrics)
                logging.info(f'Batch {i + 1}, Sequence length: {len(metrics)}, Text score: {text_score}, Image score: {img_score}, Group score: {group_score}')
            elif args.val_data == 'vqa' or args.val_data == 'vqa_binary' or args.val_data == 'vqa_other':
                r1,r5 = evaluate_scores(args, scores, batch)
                r1s += r1
                r5s += r5
                r1 = sum(r1s) / len(r1s)
                r5 = sum(r5s) / len(r5s)
                logging.info(f'Batch {i + 1}, R@1: {r1}, R@5: {r5}, Sequence length: {len(r1s)}')
            else:
                acc = evaluate_scores(args, scores, batch)
                metrics += acc
                acc = sum(metrics) / len(metrics)
                logging.info(f'Batch {i + 1}, Accuracy: {acc}, Sequence length: {len(metrics)}')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_data', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--model', type=str, default = None)
    parser.add_argument('--bias', action='store_false', help='Set bias to False')
    parser.add_argument('--sampling_time_steps', type=int, default=30)
    parser.add_argument('--version', type=str, default='2.1')
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--guidance_scale', type=float, default=0.0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('./logs', exist_ok=True)
    log_filename = f'./logs/inference_{args.val_data}_{now}.log'

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()])
    logging.info(args)
    main(args)