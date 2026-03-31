#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/11 8:47
# @Author  : wanglinghang
# @File    : cuda_tools.py
import ctypes
import sys

import numpy as np
import xml.dom.minidom as xml
import os





import matplotlib.pyplot as plt
def fp_2d(vol):
    dll_file = r'/home/ly/Python/guided-diffusion-main/CT_rec_lib/libutils.so'
    noView = 1600
    det_num = 1600
    detector_Spacing = 1
    img_spacing = 1
    SAD = 1400 * 4
    SID = 1700 * 4

    lib = ctypes.cdll.LoadLibrary(dll_file)

    vol = np.ascontiguousarray(vol, dtype=np.float32)
    vol_ptr = ctypes.cast(vol.ctypes.data, ctypes.POINTER(ctypes.c_float))

    vol_shape = np.flip(np.array(vol.shape, dtype=np.uint32))
    vol_shape = np.ascontiguousarray(vol_shape, dtype=np.uint32)
    vol_shape_ptr = ctypes.cast(vol_shape.ctypes.data, ctypes.POINTER(ctypes.c_uint))

    proj = np.zeros((noView, det_num), dtype=np.float32)
    proj = np.ascontiguousarray(proj, dtype=np.float32)
    proj_ptr = ctypes.cast(proj.ctypes.data, ctypes.POINTER(ctypes.c_float))

    proj_shape = np.array([det_num, noView], dtype=np.uint32)
    proj_shape = np.ascontiguousarray(proj_shape, dtype=np.uint32)
    proj_shape_ptr = ctypes.cast(proj_shape.ctypes.data, ctypes.POINTER(ctypes.c_uint))

    lib.fp_2d(vol_ptr, vol_shape_ptr,
              proj_ptr, proj_shape_ptr,
              ctypes.c_float(SAD), ctypes.c_float(SID),
              ctypes.c_float(img_spacing), ctypes.c_float(detector_Spacing)
              )
    return proj
import os
import glob



if __name__ == '__main__':
    # img = np.fromfile('Normal_0.raw', dtype=np.float32).reshape((640, 640))
    # print(img)
    # proj = fp_2d(img)
    # proj.astype(np.float32).tofile('0_sino1600x1600.raw')
    # print(proj)
    # 设置文件夹路径
    folder_folder = '/home/ly/Python/limated_CT_Dataset/data/data_img_640_640/'
    save_folder = '/home/ly/Python/limated_CT_Dataset/data/data_sino_1600_1600/'
    # 遍历文件夹下的所有文件
    raw_file_names = []
    for filename in os.listdir(folder_folder):
        # 检查文件名是否以.raw结尾
        if filename.endswith('.raw'):
            # 将文件名添加到列表中
            raw_file_names.append(filename)
    print(raw_file_names)

    for file in raw_file_names:
        file_path = folder_folder + file
        save_name = str(file[:-4]) + '_sino.raw'
        img = np.fromfile(file_path, dtype=np.float32).reshape((640, 640))
        proj = fp_2d(img)
        save_path = save_folder + save_name
        print(save_path)
        proj.astype(np.float32).tofile(save_path)

