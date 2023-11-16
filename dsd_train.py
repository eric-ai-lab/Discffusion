# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fine-tuning script for Stable Diffusion for text2image with support for LoRA."""

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
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import AutoTokenizer, CLIPTextModel

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, EulerDiscreteScheduler, DiffusionPipeline, UNet2DConditionModel, DSDPipeline
from model import DSD_Model
from diffusers.loaders import AttnProcsLayers
from diffusers.models.cross_attention import LoRACrossAttnProcessor
from diffusers.optimization import get_scheduler
# from diffusers.training_utils import compute_snr
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from custom_datasets import get_dataset
from eval_score import evaluate_scores
from utils import *
import datetime
import logging


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.15.0.dev0")

import torch.utils.checkpoint as checkpoint

logger = get_logger(__name__, log_level="INFO")



def save_model_card(repo_id: str, images=None, base_model=str, dataset_name=str, repo_folder=None):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- lora
inference: true
---
    """
    model_card = f"""
# LoRA text2image fine-tuning - {repo_id}
These are LoRA adaption weights for {base_model}. The weights were fine-tuned on the {dataset_name} dataset. You can find some example images in the following. \n
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--use_validation_split",
        action="store_true",
        help=(
            "Whether or not to use a validation split from the training dataset. Will use the validation percentage"
            " set in `args.validation_split_percentage`."
        ),
    )
    parser.add_argument(
        "--validation_split_percentage",
        type=float,
        default=0.1,
        help=(
            "The percentage of the train dataset to use as validation dataset if `args.use_validation_split` is set."
        ),
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="vanilla_finetuning",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument('--train_data', type=str, default='ComVG')
    parser.add_argument('--val_data', type=str, default='ComVG_sub')
    parser.add_argument(
        "--train_batch_size", type=int, default=2, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--val_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=1000,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help=(
            "Experiment with different tokenizer"
        ),
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument('--bias', action='store_false', help='Set bias to False')
    parser.add_argument('--validation', action='store_true', help='Set validation during training to True')
    parser.add_argument('--val_num', type=int, default=100, help='Number of validation used')
    parser.add_argument('--sampling_time_steps', type=int, default=30)
    parser.add_argument('--margin', type=float, default=0.1)
    parser.add_argument('--guidance_scale', type=float, default=0.0)



    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank


    # # Sanity checks
    # if args.dataset_name is None and args.train_data_dir is None:
    #     raise ValueError("Need either a dataset name or a training folder.")

    return args


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        return timesteps, num_inference_steps - t_start


def average_score(tensor_lists):
    means = [torch.mean(tensor[0]) for inner_list in tensor_lists for tensors in inner_list for tensor in tensors]
    return sum(means) / len(means)



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





def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)
    

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb



    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('./logs', exist_ok=True)
    log_filename = f'./logs/train_{args.train_data}_{args.val_data}_{now}.log'
    logging.basicConfig(level=logging.INFO,
                    format=f'%(asctime)s [Process {accelerator.process_index}] %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()])
    
    
    # logging.basicConfig(
    #     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    #     datefmt="%m/%d/%Y %H:%M:%S",
    #     level=logging.INFO,
    # )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id
            

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )



    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    logger.info(f"Using {weight_dtype} for weights.")
    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)


   
    # Set correct lora layers
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        lora_attn_procs[name] = LoRACrossAttnProcessor(
            hidden_size=hidden_size, 
            cross_attention_dim=cross_attention_dim,
            rank=args.rank,
        )

    unet.set_attn_processor(lora_attn_procs)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    lora_layers = AttnProcsLayers(unet.attn_processors)


    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    
    optimizer = optimizer_cls(
        lora_layers.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )


    '''
    # download the dataset.
    if args.dataset_name is not None:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            data_dir=args.train_data_dir,
        )
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )
    column_names = dataset["train"].column_names
    dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
    if args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )
    '''   

    
    train_dataset = get_dataset(args.train_data, f'data', transform=None, mode='train')

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    test_dataset = get_dataset(args.val_data, 'data', transform=None, mode='eval')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.dataloader_num_workers)

    if args.use_validation_split:
        val_dataset = get_dataset(args.train_data, f'data', transform=None, mode='validation')
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=args.val_batch_size,
            num_workers=args.dataloader_num_workers,
        )
    else:
        val_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=False,
            batch_size=args.val_batch_size,
            num_workers=args.dataloader_num_workers,
        )



    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    lora_layers, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        lora_layers, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("DSD-fine-tune", config=vars(args))


    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps


    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0


    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0

        uncond_tokens = [""]

        for step, batch in enumerate(train_dataloader):

            with accelerator.accumulate(unet):
                # # Convert images to latent space
                # latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                # latents = latents * vae.config.scaling_factor
                images, texts, _ = batch
                _, imgs_resize = images[0], images[1]

                
                if args.tokenizer is not None:
                    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
                    # tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
                else:
                    tokenizer = tokenizer

                scores = []
                for text_idx, text in enumerate(texts):
                    text = tokenizer(text[0], padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt").to(accelerator.device)
                    for img_idx, img in enumerate(imgs_resize):
                        if len(img.shape) == 3:
                            img = img.unsqueeze(0)
                
                        print(f'Image {img_idx}, Text {text_idx}')
                        latents = vae.encode(img.to(dtype=weight_dtype)).latent_dist.sample()
                        latents = latents * vae.config.scaling_factor

                        noise = torch.randn_like(latents)
                        if args.noise_offset:
                            noise += args.noise_offset * torch.randn(
                                (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                            )

                        bsz = latents.shape[0]
                        timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                        timesteps = timesteps.long()

                        latent_model_input = torch.cat([latents] * 2) 
                        noisy_latents = noise_scheduler.add_noise(latent_model_input, noise, timesteps)

                        # # Get the text embedding for conditioning
                        # encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                        encoder_hidden_states = text_encoder(**text)[0]
                        max_length = encoder_hidden_states.shape[1]
                        uncond_input = tokenizer(
                            uncond_tokens,
                            padding="max_length",
                            max_length=max_length,
                            truncation=True,
                            return_tensors="pt",
                        )
                        negative_prompt_embeds = text_encoder(
                            uncond_input.input_ids.to(accelerator.device),
                            attention_mask=None,)[0]
                        prompt_embeds = torch.cat([negative_prompt_embeds, encoder_hidden_states])
                        noise_uncond, noise_cond, dbscore = checkpoint.checkpoint(unet, noisy_latents, timesteps, noise, encoder_hidden_states=prompt_embeds, use_bias = args.bias, layer_mask = None, use_reentrant=False)

                        scores.append(dbscore.mean())
                scores = torch.stack(scores).permute(1, 0) if args.train_batch_size > 1 else torch.stack(scores).unsqueeze(0)


                scores = scores.contiguous()
                accelerator.wait_for_everyone()
                scores = accelerator.gather(scores)
                batch[-1] = accelerator.gather(batch[-1])


                img_idx = batch[-1]
                loss = compute_loss(args, scores, img_idx, margin = args.margin) 
                

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = lora_layers.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()     


            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                
                if global_step % args.checkpointing_steps == 0:
                    logger.info("***** Saving state *****")
                    if accelerator.is_main_process:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        
                        save_path = os.path.join(args.output_dir, f"{args.train_data}-checkpoint-{global_step}")
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        unet.save_attn_procs(save_path)
                        logger.info(f"Saved model to {save_path}")


                        if args.validation and global_step % (args.checkpointing_steps * args.validation_epochs) == 0:
                            pipeline_img2img = DSDPipeline.from_pretrained(
                                args.pretrained_model_name_or_path,
                                scheduler = EulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler"),
                                unet=accelerator.unwrap_model(unet),
                                torch_dtype=weight_dtype
                            )
                            pipeline_img2img = pipeline_img2img.to(accelerator.device)
                            pipeline_img2img.set_progress_bar_config(disable=True)

                            pipeline_img2img, val_dataloader = accelerator.prepare(pipeline_img2img, val_dataloader)

                            num_val = 0
                            r1s = []
                            r5s = []
                            accs = []
                            for i, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
                                if num_val >= args.val_num:
                                    break
                                if i % 50 != 0:
                                    continue
                                scores = compute_score(i, args, batch, pipeline_img2img, save_path='./saved_images')

                                scores = scores.contiguous()
                                accelerator.wait_for_everyone()
                                scores = accelerator.gather(scores)
                                batch[-1] = accelerator.gather(batch[-1])

                                if accelerator.is_main_process:
                                    if args.val_data == 'Refcocog' or args.val_data == 'vqa_other' or args.val_data == 'vqa_binary':
                                        r1,r5  = evaluate_scores(args, scores, batch)
                                        r1s += r1
                                        r5s += r5
                                        r1 = sum(r1s) / len(r1s)
                                        r5 = sum(r5s) / len(r5s)
                                        logging.info(f'Validation Batch {i + 1}, R@1: {r1}, R@5: {r5}, Sequence length: {len(r1s)}')
                                    else:
                                        acc = evaluate_scores(args, scores, batch)
                                        accs += acc
                                        acc = sum(accs) / len(accs)
                                        logging.info(f'Validation Batch {i + 1}, Accuracy: {acc}, Sequence length: {len(accs)}')

                                num_val += 1


            

            logs = {"train_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

       
    logger.info("***** Eval main process *****")
    pipeline_img2img = DSDPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        scheduler = EulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler"),
        unet=accelerator.unwrap_model(unet),
        # torch_dtype=torch.float16
        torch_dtype=weight_dtype
    )
    pipeline_img2img = pipeline_img2img.to(accelerator.device)
    pipeline_img2img.set_progress_bar_config(disable=True)

    pipeline_img2img, test_dataloader = accelerator.prepare(pipeline_img2img, test_dataloader)

    r1s = []
    r5s = []
    accs = []
    for i, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):

        scores = compute_score(i, args, batch, pipeline_img2img, save_path='./saved_images')

        scores = scores.contiguous()
        accelerator.wait_for_everyone()
        scores = accelerator.gather(scores)
        batch[-1] = accelerator.gather(batch[-1])

        if accelerator.is_main_process:
            if args.val_data == 'Refcocog' or args.val_data == 'vqa_other' or args.val_data == 'vqa_binary':
                r1,r5  = evaluate_scores(args, scores, batch)
                r1s += r1
                r5s += r5
                r1 = sum(r1s) / len(r1s)
                r5 = sum(r5s) / len(r5s)
                logging.info(f'Batch {i + 1}, R@1: {r1}, R@5: {r5}, Sequence length: {len(accs)}')
            else:
                acc = evaluate_scores(args, scores, batch)
                accs += acc
                acc = sum(accs) / len(accs)
                print(f'Accuracy: {acc}')
                logging.info(f'Batch {i + 1}, Accuracy: {acc}, Sequence length: {len(accs)}')

    del pipeline_img2img
    torch.cuda.empty_cache()


    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, f"final-checkpoint-{global_step}")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        unet = unet.to(torch.float32)
        unet.save_attn_procs(save_path)
        logger.info(f"Saved the final state to {save_path}")


        # if args.push_to_hub:
        #     save_model_card(
        #         repo_id,
        #         images=images,
        #         base_model=args.pretrained_model_name_or_path,
        #         dataset_name=args.dataset_name,
        #         repo_folder=args.output_dir,
        #     )
        #     upload_folder(
        #         repo_id=repo_id,
        #         folder_path=args.output_dir,
        #         commit_message="End of training",
        #         ignore_patterns=["step_*", "epoch_*"],
        #     )
    


        # Final inference
        validation_prompts = ["A dog is smiling"]
        # Load previous pipeline
        pipeline = DiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path, revision=args.revision, torch_dtype=weight_dtype
        )
        pipeline = pipeline.to(accelerator.device)

        # load attention processors
        pipeline.unet.load_attn_procs(save_path)

        # run inference
        generator = torch.Generator(device=accelerator.device)
        if args.seed is not None:
            generator = generator.manual_seed(args.seed)
        images = []
        for validation_prompt in validation_prompts:
            for _ in range(10):
                image = pipeline(validation_prompt, num_inference_steps=30, generator=generator).images[0]
                images.append((image, validation_prompt))


    if accelerator.is_main_process:
        for tracker in accelerator.trackers:
            if len(images) != 0:
                if tracker.name == "tensorboard":
                    np_images = np.stack([np.asarray(img) for img in images])
                    tracker.writer.add_images("test", np_images, epoch, dataformats="NHWC")
                if tracker.name == "wandb":
                    tracker.log(
                        {
                           "test": [
                            wandb.Image(image, caption=f"{i}: {prompt}")
                            for i, (image, prompt) in enumerate(images)
                        ]
                        }
                    )

    accelerator.end_training()
    logger.info("***** End training *****")


if __name__ == "__main__":
    main()