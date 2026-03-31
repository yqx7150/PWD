import sys
import os
import numpy as np
from matplotlib import pyplot as plt
from numpy import matlib
# from .loadData import loadData
from scipy.interpolate import griddata
from scipy.signal import medfilt2d

import scipy.io as sio

class FanBeam():
    def __init__(self, img_size):
        self.limited_projGeom90 = astra.create_proj_geom('fanflat', 2.0, 768,
                                                          np.linspace(0, 2*np.pi/4, 180, endpoint=False), 6800,
                                                          5400)#1.8, 1000
        self.limited_projGeom60 = astra.create_proj_geom('fanflat', 2.0, 768,
                                                         np.linspace(0, 2*np.pi/6, 120, endpoint=False), 6800,
                                                        5400)

        self.limited_projGeom30 = astra.create_proj_geom('fanflat', 2.0, 768,
                                                         np.linspace(0, 2*np.pi / 12, 60, endpoint=False), 6800,
                                                         5400)
        self.limited_projGeom120 = astra.create_proj_geom('fanflat', 2.0, 768,
                                                         np.linspace(0, 2*np.pi / 3, 240, endpoint=False), 6800,
                                                         5400)
        self.volGeom = astra.create_vol_geom(img_size, img_size, (-img_size / 2) * 400 / img_size, (img_size / 2) * 400 / img_size,
                                             (-img_size / 2) * 400 / img_size, (img_size / 2) * 400 / img_size)

    def FP(self, img, ang_num):
        if ang_num == 90:
            projGeom = self.limited_projGeom90
        elif ang_num == 60:
            projGeom = self.limited_projGeom60
        elif ang_num == 30:
            projGeom = self.limited_projGeom30
        elif ang_num == 120:
            projGeom = self.limited_projGeom120
        volGeom = self.volGeom
        rec_id = astra.data2d.create('-vol', volGeom, img)
        proj_id = astra.data2d.create('-sino', projGeom)
        cfg = astra.astra_dict('FP_CUDA')
        cfg['VolumeDataId'] = rec_id
        cfg['ProjectionDataId'] = proj_id
        #   cfg['option'] = {'VoxelSuperSampling': 2}
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        rec = astra.data2d.get(rec_id).T
        pro = astra.data2d.get(proj_id)
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(rec_id)
        astra.data2d.delete(proj_id)

        return pro

    def FBP(self, proj, ang_num):
        if ang_num == 90:
            projGeom = self.limited_projGeom90
        elif ang_num == 60:
            projGeom = self.limited_projGeom60
        elif ang_num == 30:
            projGeom = self.limited_projGeom30
        elif ang_num == 120:
            projGeom = self.limited_projGeom120

        volGeom = self.volGeom
        rec_id = astra.data2d.create('-vol', volGeom)
        proj_id = astra.data2d.create('-sino', projGeom, proj)
        cfg = astra.astra_dict('FBP_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = proj_id
        #   cfg['option'] = {'VoxelSuperSampling': 2}
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        rec = astra.data2d.get(rec_id)
        pro = astra.data2d.get(proj_id)
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(rec_id)
        astra.data2d.delete(proj_id)

        return rec

    def SIRT(self, VOL, proj, ang_num,iter_num):
        if ang_num == 90:
            projGeom = self.limited_projGeom90
        elif ang_num == 60:
            projGeom = self.limited_projGeom60
        elif ang_num == 30:
            projGeom = self.limited_projGeom30
        elif ang_num == 120:
            projGeom = self.limited_projGeom120

        volGeom = self.volGeom
        if VOL is None:
            rec_id = astra.data2d.create('-vol', volGeom)
        else:
            rec_id = astra.data2d.create('-vol', volGeom, VOL)
        proj_id = astra.data2d.create('-sino', projGeom, proj)
        cfg = astra.astra_dict('SIRT_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = proj_id
        #   cfg['option'] = {'VoxelSuperSampling': 2}
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, iter_num)
        rec = astra.data2d.get(rec_id)
        pro = astra.data2d.get(proj_id)
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(rec_id)
        astra.data2d.delete(proj_id)

        return rec


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
    normalized_img = (img - min_val) / (max_val - min_val)
    return normalized_img


# 示例：对 limited_img 进行归一化

if __name__ == '__main__':
    from PIL import Image
    import numpy as np

    # 读取 png 图像为 NumPy 数组
    image_path = '/home/ly/Python/guided-diffusion-main/AAPM_Dataset/1.png'  # 替换为实际文件路径
    image = Image.open(image_path)
    img = np.array(image)

    # 显示图像数组的形状
    print("image_array",img.shape)




    file_path = "/home/ly/Python/guided-diffusion-main/AAPM_Dataset/train_img/1257.npy"  # 替换为实际文件路径
    # img = np.load(file_path)
    min_val = np.min(img)
    max_val = np.max(img)
    print(f"数组的值范围: 最小值={min_val}, 最大值={max_val}")
    fanBeam = FanBeam(img_size=512)
    print(img.shape)  # (512, 512)
    limited_sinogram = fanBeam.FP(img, ang_num=60).astype(np.float32)
    limited_sinogram = limited_sinogram.astype(np.float32)
    print(limited_sinogram.shape)  # (120, 768)
    #
    limited_img = fanBeam.FBP(limited_sinogram, ang_num=60).astype(np.float32)
    # 假设 limited_img 已经计算完成
    min_val = np.min(limited_img)
    max_val = np.max(limited_img)
    print(f"数组的值范围: 最小值={min_val}, 最大值={max_val}")
    limited_img = normalize_image(limited_img)
    print(limited_img.shape)  # (512, 512)

    # 绘图展示
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.colorbar(label="Pixel Value")
    plt.axis('off')

    # 限制角度的正弦图
    plt.subplot(1, 3, 2)
    plt.imshow(limited_sinogram, cmap='gray', aspect='auto')
    plt.title("Limited Sinogram (60 angles)")
    plt.colorbar(label="Sinogram Value")
    plt.axis('off')

    # 重建图像
    plt.subplot(1, 3, 3)
    plt.imshow(limited_img, cmap='gray')
    plt.title("Reconstructed Image")
    plt.colorbar(label="Pixel Value")
    plt.axis('off')

    # 显示图像
    plt.tight_layout()
    plt.show()