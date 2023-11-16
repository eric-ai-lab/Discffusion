import numpy as np
import json
from math import floor
from typing import Union, Callable
import torch
import torch.nn.functional as F


def evaluate_comvg(args, scores, img_idx):
    if type(scores) != list:
        img_idx = img_idx.cpu().numpy()
        scores = scores.detach().cpu().numpy()
    scores = np.stack(scores, axis=0)
    retrieval_accuracy = []
    for i in range(scores.shape[0]):
        if img_idx[i] == np.argmax(scores[i]):
            retrieval_accuracy.append(1)
        else:
            retrieval_accuracy.append(0)
    return retrieval_accuracy


def evaluate_vqa(args, scores, img_idx):
    if type(scores) != list:
        img_idx = img_idx.cpu().numpy()
        scores = scores.detach().cpu().numpy()
    scores = np.stack(scores, axis=0)
    retrieval_accuracy = []
    for i in range(scores.shape[0]):
        if img_idx[i] == np.argmax(scores[i]):
            retrieval_accuracy.append(1)
        else:
            retrieval_accuracy.append(0)

    r5 = []
    for i in range(scores.shape[0]):
        if img_idx[i] in np.argsort(scores[i],axis=0)[-5:]:
            r5.append(1)
        else:
            r5.append(0)
    return retrieval_accuracy, r5


def evaluate_retrieval(args, scores, img_idx):
    if type(scores) != list:
        img_idx = img_idx.cpu().numpy()
        scores = scores.detach().cpu().numpy()
    scores = np.stack(scores, axis=0)
    retrieval_accuracy = []
    for i in range(scores.shape[0]):
        if img_idx[i] == np.argmax(scores[i]):
            retrieval_accuracy.append(1)
        else:
            retrieval_accuracy.append(0)
            
    r5 = []
    for i in range(scores.shape[0]):
        if img_idx[i] in np.argsort(scores[i],axis=0)[-5:]:
            r5.append(1)
        else:
            r5.append(0)
    return retrieval_accuracy, r5



def evaluate_scores(args, scores, batch):
    if args.val_data == 'ComVG_sub' or args.val_data == 'ComVG_obj' or args.val_data == 'ComVG_verb' or args.val_data== 'ComVG':
        img_idx = batch[-1]
        score = evaluate_comvg(args, scores, img_idx)
    elif args.val_data == 'vqa_binary' or args.val_data == 'vqa_other' or args.val_data == 'vqa':
        img_idx = batch[-1]
        score = evaluate_vqa(args, scores, img_idx)
    elif args.val_data == 'Refcocog':
        img_idx = batch[-1]
        score = evaluate_retrieval(args, scores, img_idx) 
    elif args.val_data == 'winoground':
        score = evaluate_winoground(scores)
    else:
        raise NotImplementedError
    return score
