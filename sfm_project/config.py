#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置文件
"""

import os
import cv2
import numpy as np

# ============ 相机内参 ============
# 官方全分辨率（Middlebury Temple Ring，3072×2048）标定内参，仅作「兜底参考」。
#
# 运行时内参由 core.calibration 自动确定（AUTO_CALIBRATE=True 时）：
#   EXIF 焦距 → 默认 f = 1.0×宽（方形像素、主点居中）。
# 覆盖所有数据集（含 images2 及任意新图组），不再依赖官方标定值或手工缩放。
# 若要强制用下面的官方值（例如对比实验），把 Config.AUTO_CALIBRATE 置 False 即可。
_OFFICIAL_K = np.array([
    [2759.48, 0, 1520.69],
    [0, 2764.16, 1006.81],
    [0, 0, 1]
])
_OFFICIAL_SIZE = (3072, 2048)  # (宽, 高)


def scale_intrinsics(K, src_size, dst_size):
    """按图像尺寸缩放相机内参矩阵（备用，见上面 TODO 说明）。

    缩放规则：fx、cx 按宽度比例缩放；fy、cy 按高度比例缩放。
    """
    sx = dst_size[0] / src_size[0]
    sy = dst_size[1] / src_size[1]
    K_scaled = K.astype(np.float64).copy()
    K_scaled[0, 0] *= sx  # fx
    K_scaled[1, 1] *= sy  # fy
    K_scaled[0, 2] *= sx  # cx
    K_scaled[1, 2] *= sy  # cy
    return K_scaled


class Config:
    """全局配置"""

    # 相机内参：见文件头说明。运行时会被 core.calibration 自动标定覆盖。
    K = _OFFICIAL_K.copy()

    # 是否在流水线开始时自动确定内参 K（EXIF → 默认）。False 则用上面的 K。
    AUTO_CALIBRATE = True

    # 特征提取配置
    FEATURE = {
        'nfeatures': 0,  # 0表示不限制
        'nOctaveLayers': 3,
        'contrastThreshold': 0.04,
        'edgeThreshold': 10
    }

    # 特征匹配配置
    MATCHING = {
        'ratio': 0.5,  # Lowe's ratio test
        'norm_type': cv2.NORM_L2
    }

    # 重建配置
    RECONSTRUCTION = {
        'min_matches': 8,  # 最小匹配点数
        'reproj_threshold': 0.5  # 重投影误差阈值（像素）
    }

    # BA 配置
    BA = {
        'max_iterations': 100,
        'use_cuda': True,  # True 时优先走 CUDA 并行过滤，失败自动回退 CPU
        'reproj_threshold': 0.5  # 重投影误差阈值
    }

    # 路径（以本文件所在目录为基准，避免依赖运行时的当前工作目录）
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.normpath(os.path.join(_BASE_DIR, "..", "images2"))
    OUTPUT_PATH = os.path.join(_BASE_DIR, "output")
