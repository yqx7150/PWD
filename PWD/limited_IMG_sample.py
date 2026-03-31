"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import time

from functools import partial

import numpy as np
import torch as th


from CT_rec_lib.cuda_tools import fp_2d
from CT_rec_lib.recon_tool import bp_2d
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_CT_IMG_data_640x640
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict, CT_IMG_create_model_and_diffusion, diffusion_defaults,
)




from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import mean_squared_error as MSE
def sino_limited(label, rec_result, limited_num, image_size):
    # 统一范围
    label_sino = fp_2d(label)
    bz_label = bp_2d(label_sino, 1600, image_size)
    bz_rec_sino = fp_2d(rec_result)
    # sino保真
    limited_sino = label_sino[:limited_num, :]

    bz_rec_sino[:limited_num, :] = limited_sino

    # 保真结果重建
    rec_result_bp = bp_2d(bz_rec_sino, 1600, image_size)

    return bz_label, rec_result_bp

def main():
    os.environ['TORCH_USE_CUDA_DSA'] = '1'
    os.environ['RDMAV_FORK_SAFE'] = '1'


    args = create_argparser().parse_args()

    device = dist_util.dev(args.gpu_id)
    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = CT_IMG_create_model_and_diffusion(
        **args_to_dict(args, CT_IMG_model_and_diffusion_defaults().keys())
    )

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location='cpu')
    )
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("Sampling start...")
    all_GT_images, all_Result, all_limited_images = [], [], []

    # 设置数据读取步骤
    data = load_CT_IMG_data_640x640(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        mode='test',
        angle=args.angle,
    )

    run_sampler = partial(diffusion.CT_IMG_sample_loop_test)
    psnr_all, mse_all, ssim_all = [], [], []
    for i, data_batch in enumerate(data):
        # 构建数据
        [img_GT, limited_img, img_path] = data_batch
        print(img_path)
        img_name = img_path[0].split('/img/')[1].split('.raw')[0]


        print(img_name)
        # if img_name in ["4208"]:
        # if True:
        ###DDIM
        if img_name in ["5908"]:
            # 把每一张欠采数据,丢进去采样
            limited_img = limited_img.to(device)
            img_GT = img_GT.to(device)
            limited_img = [limited_img, img_GT]
            th.cuda.empty_cache()
            # 开始计时
            start_time = time.time()
            result_img = run_sampler(
                model=model,
                limited_img=limited_img,
                shape=limited_img[0].shape,
                slover_data=args.slover_data,
                # 加一个保真项 和 角度数量
                limited_sino=None,
                limited_num=None,
            )
            result_img.clamp(limited_img[1].min(), limited_img[1].max())
            # 结束计时
            end_time = time.time()
            # 计算运行时间
            elapsed_time = end_time - start_time

            # 打印运行时间，精确到小数点后两位
            print(f"Sample 运行时间：{elapsed_time:.2f}秒")
            # result_img = th.mean(result_img, 0, keepdim=True)
            result_img = result_img.permute(0, 2, 3, 1)

            '''保存重建结果'''
            save_path_rec = args.save_dir + img_name + '_rec.raw'
            save_path_gt = args.save_dir + img_name + '_gt.raw'
            save_path_limited = args.save_dir + img_name + '_limited.raw'

            result_img = np.squeeze(result_img.cpu().numpy())
            label = np.squeeze(img_GT.cpu().numpy())
            limited_img_ = np.squeeze(limited_img[0].cpu().numpy())
            # 保存重建结果
            result_img.astype(np.float32).tofile(save_path_rec)
            limited_img_.astype(np.float32).tofile(save_path_limited)
            label.astype(np.float32).tofile(save_path_gt)


            # 计算每一张图片的 PSNR 和 SSIM 以及 MSE
            psnr = PSNR(label, result_img, data_range=label.max() - label.min())
            mse = MSE(label, result_img)
            ssim = SSIM(label, result_img, data_range=label.max() - label.min())
            print(f" PSNR : {psnr}, MSE : {mse}, SSIM : {ssim}")

            psnr_all.append(psnr)
            ssim_all.append(ssim)
            mse_all.append(mse)
        # break
    logger.log(f"PSNR_MEAN_DATA: {np.array(psnr_all).mean()}, PSNR_MAX_DATA: {np.array(psnr_all).max()}, PSNR_MIN_DATA: {np.array(psnr_all).min()}")
    logger.log(f"SSIM_MEAN_DATA: {np.array(ssim_all).mean()}, SSIM_MAX_DATA: {np.array(ssim_all).max()}, SSIM_MIN_DATA: {np.array(ssim_all).min()}")
    logger.log(f"MSE_MEAN_DATA: {np.array(mse_all).mean()}, MSE_MAX_DATA: {np.array(mse_all).max()}, MSE_MIN_DATA: {np.array(mse_all).min()}")
    logger.log("Sampling complete")


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
        clip_denoised=True,
        use_ddim=False,
        gpu_id=3,
        data_dir="/home/ly/Python/guided-diffusion-main/Dataset/Yofo_Dataset/limited_datasets_640x640/test_data/img",
        # data_dir="/home/ly/Python/guided-diffusion-main/Dataset/Yofo_Dataset/limited_dataset_01/640x640/train_data/img",
        # data_dir="/home/ly/Python/Guided_difussion/test/img",
        batch_size=1,
        image_size=640,
        angle=90,
        model_path="/home/ly/Python/Guided_difussion/result/640x640_Result/result_Model/90_limited_Model/90_limited_640_cond=0.5/ema_img_0.9999_1000000.pt",
        save_dir='/home/ly/Python/Guided_difussion/result/',
        slover_data='no',  # CG: img ; APGM: sino; no
    )
    defaults.update(CT_IMG_model_and_diffusion_defaults())

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser



if __name__ == "__main__":
    main()
