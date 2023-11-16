import datasets
import json
import argparse
import logging
import math
import os
import shutil
import time
import random
from pathlib import Path
from typing import Optional

import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import re


def sanitize_text(text):
    text = re.sub(r'[^\w\s]', '', text)  
    text = text.replace(' ', '_')  
    return text[:50]  


def compute_score(i, args, batch, model, save_path='./saved_images'):

    imgs, texts = batch[0], batch[1]
    imgs, imgs_resize = imgs[0], imgs[1]
    imgs_resize = [img.cuda() for img in imgs_resize]

    batchsize = imgs_resize[0].shape[0]
    scores = []
    for txt_idx, text in enumerate(texts):
        sanitized_text = sanitize_text(text[0])  
        text_folder_path = os.path.join(save_path, sanitized_text)
        for img_idx, (img_name, resized_img) in enumerate(zip(imgs, imgs_resize)):
            if len(resized_img.shape) == 3:
                resized_img = resized_img.unsqueeze(0)
            
            print(f'Batch {i}, Text {txt_idx}, Image {img_idx}')
            score = model(prompt=list(text), image=resized_img, guidance_scale=args.guidance_scale, sampling_steps=args.sampling_time_steps, layer_mask = None, use_bias = args.bias, sampled_timestep = None, level = None)
            scores.append(score)
            
            # rename_image_based_on_score(score, img_name[0], img_idx, text_folder_path)

    scores = torch.stack(scores).permute(1, 0) if batchsize > 1 else torch.stack(scores).unsqueeze(0)
    return scores


def compute_loss(args, scores, img_idx, margin=0.2):
    img_idx = 0
    batch_size = scores.size(0)
    img_idx_tensor = torch.full((batch_size,), fill_value=img_idx, dtype=torch.long, device=scores.device)
    
    positive_scores = scores.gather(1, img_idx_tensor.unsqueeze(-1)).squeeze(-1)
    
    modified_scores = scores.scatter(1, img_idx_tensor.unsqueeze(-1), float('-inf'))
    negative_scores, _ = modified_scores.max(dim=1)
    
    loss = torch.clamp(negative_scores - positive_scores + margin, min=0).mean()

    return loss


def compute_loss_CE(args, scores, img_idx):
    if type(scores) != list:
        img_idx = img_idx.cpu().numpy()
        scores = scores.cpu().numpy()
    scores = np.stack(scores, axis=0)
    retrieval_accuracy = []
    max_more_than_once = 0
    for i in range(scores.shape[0]):
        number_of_argmax_appearances = np.sum(scores[i] == np.max(scores[i]))
        if number_of_argmax_appearances > 1:
            max_more_than_once += 1
        if img_idx[i] == np.argmax(scores[i]):
            retrieval_accuracy.append(1)
        else:
            retrieval_accuracy.append(0)
    scores_tensor = torch.tensor(scores)
    probabilities = F.softmax(scores_tensor, dim=1)
    labels = torch.tensor(img_idx)
    ce_loss = F.cross_entropy(scores_tensor, labels)
    return ce_loss.item(), retrieval_accuracy