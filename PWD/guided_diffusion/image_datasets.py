import math
import random
import glob
import os
from PIL import Image
import blobfile as bf
from matplotlib import pyplot as plt
from mpi4py import MPI
import numpy as np
from scipy.interpolate import interp1d
from torch.utils.data import DataLoader, Dataset
from CT_rec_lib.cuda_tools import fp_2d
from CT_rec_lib.limited_CT_tool import FanBeam
from CT_rec_lib.recon_tool import bp_2d
from CT_rec_lib import *
import torch as th

def _list_all_files(data_dir):
    all_items = os.listdir(data_dir)
    result = []
    for item in all_items:
        result.append(os.path.join(data_dir, item))
    return result

def _list_ct_files(data_dir):
    # 读取raw文件路径
    img_data_dir = data_dir
    img_raw_files = glob.glob(os.path.join(img_data_dir, '*.raw'))
    img_results = []
    for file in img_raw_files:
        img_results.append(file)
    return img_results


def _list_npy_files(data_dir):
    # 读取raw文件路径
    img_data_dir = data_dir
    img_raw_files = glob.glob(os.path.join(img_data_dir, '*.npy'))
    img_results = []
    for file in img_raw_files:
        img_results.append(file)
    return img_results


def normalize_image(img):
    # 检查是否为 NumPy 数组
    if not isinstance(img, np.ndarray):
        raise ValueError("输入图像必须是 NumPy 数组")
    # 检查是否为二维
    if img.ndim != 2:
        raise ValueError("输入图像必须是二维数组")
    # 计算最小值和最大值
    min_val = np.min(img)
    max_val = np.max(img)

    # 防止最大值等于最小值（避免除零）
    if max_val == min_val:
        return np.zeros_like(img)
    # 归一化公式
    # normalized_img = (img - min_val) / (max_val - min_val)
    normalized_img = 2 * (img - min_val) / (max_val - min_val) - 1
    return normalized_img
    # return normalized_img



class CT_img_Dataset_640x640(Dataset):
    def __init__(self, image_paths, image_size, angle):
        self.image_paths = image_paths
        self.image_size = image_size
        self.angle = angle

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        limited_path = img_path.replace('img', 'limited_' + str(self.angle)).replace('.raw', '_limited_' + str(
            self.angle) + '.raw')

        img_data = np.fromfile(img_path, dtype=np.float32).reshape(self.image_size, self.image_size)
        limited_img_data = np.fromfile(limited_path, dtype=np.float32).reshape(self.image_size, self.image_size)
        # img_data = normalize_image(img_data)
        # limited_img_data = normalize_image(limited_img_data)
        # print(limited_img_data.max(),limited_img_data.min())
        # print(img_data.max(),img_data.min())
        img_data = img_data[None, :, :]
        limited_img_data = limited_img_data[None, :, :]

        return img_data, limited_img_data, img_path


def load_CT_IMG_data_640x640(
        *,
        data_dir,
        batch_size,
        image_size,
        mode,
        angle
):
    if not data_dir:
        raise ValueError("data directory exit Error !!!!!!!!" + data_dir)
    img_files = _list_ct_files(data_dir)
    ct_dataset = CT_img_Dataset_640x640(image_paths=img_files,
                                        image_size=image_size,
                                        angle=angle
                                        )

    loader = DataLoader(
        ct_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    if mode == 'train':
        while True:
            yield from loader
    elif mode == 'test':
        yield from loader

class CT_Mar_Dataset_640x640(Dataset):
    def __init__(self, image_paths, image_size, pre):
        self.image_paths = image_paths
        self.image_size = image_size
        self.pre = pre
        self.files = []
        masks = os.listdir(os.path.join(image_paths, 'sino_mask'))
        for file in masks:
            self.files.append(os.path.join(image_paths, 'sino_mask', file))
        # for dir_path in os.listdir(image_paths):
        #     dir_path = os.path.join(image_paths, dir_path)
        #     masks = os.listdir(os.path.join(dir_path, 'mask'))
        #     for file in masks:
        #         self.files.append(os.path.join(dir_path, 'mask', file))
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # print(len(self.files))
        mask_path = self.files[idx]
        sino_path = mask_path.replace('sino_mask', 'sino')
        print(mask_path)
        print(sino_path)
        sino_data = np.fromfile(sino_path, dtype=np.float32).reshape(640, 640)
        mask = np.fromfile(mask_path, dtype=np.float32).reshape(640, 640)

        sino_data[sino_data < 0] = 0
        sino_data[np.isnan(sino_data)] = 0
        sino_data[np.isinf(sino_data)] = 0
        sino_data = sino_data / 5.0 - 0.5

        mask = mask > 0
        mask = ~mask

        mask_data = sino_data * mask
        sino_data = sino_data[None, :, :]
        mask_data = mask_data[None, :, :]
        if self.pre:
            print(sino_path.split("sino/")[1])
            img_num = int(sino_path.split("sino/")[1].split(".raw")[0])
            pre_file = os.path.join(sino_path.split("sino/")[0], "sino/", f"{max(0, img_num - 1):05d}" + ".raw" )
            pre_data = np.fromfile(pre_file, dtype=np.float32).reshape(640, 640)

            pre_data[pre_data < 0] = 0
            pre_data[np.isnan(pre_data)] = 0
            pre_data[np.isinf(pre_data)] = 0
            pre_data = pre_data / 5.0 - 0.5
            pre_data = pre_data[None, :, :]
            return sino_data, mask_data, pre_data, sino_path
        else:
            return sino_data, mask_data, sino_path

def load_data_MAR_640x640(
        data_dir,
        batch_size,
        image_size,
        pre,
        mode,
):
    if not data_dir:
        raise ValueError("data directory exit Error !!!!!!!!" + data_dir)
    # img_files = _list_all_files(data_dir)
    ct_dataset = CT_Mar_Dataset_640x640(image_paths=data_dir, image_size=image_size,pre=pre)
    loader = DataLoader(
        ct_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    if mode == 'train':
        while True:
            yield from loader
    elif mode == 'test':
        yield from loader

if __name__ == '__main__':

    batch_size = 4
    image_size = 640
    data = load_data_MAR_640x640(
        data_dir='/home/ly/Python/Guided_difussion/MAR_result/1.2.156.89797079.473640376.1108204523.2270370995/MAR_data/resize_Data',
        batch_size=batch_size,
        image_size=image_size,
        pre=True,
        mode='test',
    )
    for i, data_batch in enumerate(data):
        print("xxx")

        # 获取 numpy 数据
        sino_1 = np.squeeze(data_batch[0].cpu().numpy())  # shape: (B, H, W)
        sino_2 = np.squeeze(data_batch[1].cpu().numpy())  # shape: (B, H, W)
        sino_3 = np.squeeze(data_batch[2].cpu().numpy())  # shape: (B, H, W)
        Path = data_batch[2] # shape: (B, H, W)
        #
        # 可视化 batch 中的前几张
        for j in range(min(4, sino_1.shape[0])):
            plt.figure(figsize=(8, 4))

            plt.subplot(1, 3, 1)
            plt.imshow(sino_1[j], cmap='gray')
            plt.title(f'Sino 1 - Sample {j}')
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(sino_2[j], cmap='gray')
            plt.title(f'Sino 2 - Sample {j}')
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(sino_3[j], cmap='gray')
            plt.title(f'Sino 2 - Sample {j}')
            plt.axis('off')

            plt.tight_layout()
            plt.show()

        break
