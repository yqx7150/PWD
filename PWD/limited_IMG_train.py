"""
Train a diffusion model on images.
"""
import argparse

import torch

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import  load_CT_IMG_data_640x640
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    CT_IMG_create_model_and_diffusion
)
import torch as th
from guided_diffusion.train_util import TrainLoop
import os



def main():
    os.environ['TORCH_USE_CUDA_DSA'] = '1'
    os.environ['RDMAV_FORK_SAFE'] = '1'

    args = create_argparser().parse_args()
    device = dist_util.dev(args.gpu_id)
    th.cuda.set_device(args.gpu_id)
    dist_util.setup_dist()
    logger.configure(args.save_path)

    logger.log("Creating CT_IMG model and diffusion...")
    model, diffusion = CT_IMG_create_model_and_diffusion(
        **args_to_dict(args, CT_IMG_model_and_diffusion_defaults().keys())
    )

    if th.cuda.is_available():
        print("CUDA")
    else:
        print("CPU")
    # dev() 函数输入设备ID
    model.to(device)
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("Creating CT_IMG_data loader...")

    data = load_CT_IMG_data_640x640(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        mode='train',
        angle=args.angle
    )

    logger.log("training...")

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        data_mode=args.data_mode,

        batch_size=args.batch_size,
        microbatch=args.microbatch,

        lr=args.lr,
        ema_rate=args.ema_rate,
        device_id=device,
        log_interval=args.log_interval,
        save_interval=args.save_interval,

        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,

        save_path=args.save_path,
    ).run_loop()
def diffusion_defaults():
    """
    Defaults for image and classifier training.
    """
    return dict(
        learn_sigma=False,
        diffusion_steps=2000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )

def CT_IMG_model_and_diffusion_defaults():
    """
    Defaults for CT_image training.
    """
    res = dict(
        image_size=640,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="16,8",
        channel_mult="",
        dropout=0.0,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
        learn_sigma=True,
    )
    res.update(diffusion_defaults())
    return res
def create_argparser():
    defaults = dict(
        # MODEL_FLAGS=
        gpu_id=1,
        data_mode='img',
        # TRAIN_FLAGS
        lr=1e-4,
        batch_size=1,
        data_dir="/home/ly/Python/guided-diffusion-main/Dataset/Yofo_Dataset/limited_dataset_01/640x640/train_data/img",
        angle=90,
        schedule_sampler="uniform",
        weight_decay=0.0,
        lr_anneal_steps=0,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values

        log_interval=5000,  # show Log
        save_interval=50000,  # save Model number

        # resume_checkpoint="/home/ly/Python/Guided_difussion/result/result_Model/120_limited_Model/120_limited_640_cond=0.2/mode_img_150000.pt",
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,

        save_path="/home/ly/Python/Guided_difussion/result/model/"
    )
    defaults.update(CT_IMG_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
