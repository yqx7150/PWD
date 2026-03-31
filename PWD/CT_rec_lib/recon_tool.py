#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: WLH
@file: recon_tool.py
@time: 24-9-20 下午1:50
"""
import ctypes
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import xml.dom.minidom as xml
import math
import os




def bp_2d(proj,limted,image_size):
    dll_file = r'/home/ly/Python/guided-diffusion-main/CT_rec_lib/librecon_tool.so'
    proj = proj[:limted, :]
    padded_sino = np.zeros((1600, 1600))
    # 将原始数组复制到新数组的左上角
    padded_sino[:limted, :1600] = proj
    vol_shape = image_size
    noView = padded_sino.shape[0]
    coef = 3.14 * 10 / noView
    detector_Spacing = 1
    img_spacing = 1
    SAD = 1400 * 4
    SID = 1700 * 4

    lib = ctypes.cdll.LoadLibrary(dll_file)
    # 768*1600
    proj_shape = np.array([padded_sino.shape[1], padded_sino.shape[0]], dtype=np.uint32)
    proj_shape_ptr = ctypes.cast(proj_shape.ctypes.data, ctypes.POINTER(ctypes.c_uint))

    projs = np.ascontiguousarray(padded_sino, dtype=np.float32)
    projs_ptr = ctypes.cast(projs.ctypes.data, ctypes.POINTER(ctypes.c_float))

    proj_filter = np.zeros((padded_sino.shape[1], padded_sino.shape[0]), dtype=np.float32)
    proj_filter = np.ascontiguousarray(proj_filter, dtype=np.float32)
    proj_filter_ptr = ctypes.cast(proj_filter.ctypes.data, ctypes.POINTER(ctypes.c_float))

    lib.rampFilter2d(proj_filter_ptr,
                     projs_ptr,
                     proj_shape_ptr,
                     ctypes.c_float(detector_Spacing))

    vol = np.zeros((vol_shape, vol_shape), dtype=np.float32)
    vol = np.ascontiguousarray(vol, dtype=np.float32)
    vol_ptr = ctypes.cast(vol.ctypes.data, ctypes.POINTER(ctypes.c_float))

    vol_shape = np.flip(np.array(vol.shape, dtype=np.uint32))
    vol_shape = np.ascontiguousarray(vol_shape, dtype=np.uint32)
    vol_shape_ptr = ctypes.cast(vol_shape.ctypes.data, ctypes.POINTER(ctypes.c_uint))

    lib.BP_Test(vol_ptr, vol_shape_ptr,
                proj_filter_ptr, proj_shape_ptr,
                ctypes.c_float(SAD), ctypes.c_float(SID),
                ctypes.c_float(img_spacing), ctypes.c_float(detector_Spacing),
                ctypes.c_float(coef)
                )
    return vol


if __name__ == '__main__':
    proj = np.fromfile('0_sino.raw', dtype=np.float32).reshape((1600, 1600))
    print(proj.shape)
    limited = 300

    pVol = bp_2d(proj, limited)
    print(pVol.shape)
    pVol.astype(np.float32).tofile('bp_'+str(limited)+'.raw')



