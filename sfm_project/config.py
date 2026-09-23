#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置文件
"""

import os
import cv2
import numpy as np

# ============ 相机内参 ============
# 官方全分辨率（Middlebury Temple Ring，3072×2048）标定内参。
#
# 【已知问题】images2 的图片实际是 640×480，严格来说内参需按尺寸等比缩小。
# 但实测（2026-09-09）缩小后 fx≈575、fy≈648 不再相等（fx/fy 相差约 12%），
# 增量重建出现严重深度歧义：点云沿 z（深度）方向被拉长 10 倍以上，并出现
# 大量相机后方（负深度）点。原因是当前流水线只做「相邻帧顺序匹配 + 增量 PnP
# + 仅优化 3D 点的 BA」，对环形小基线数据集的深度约束本就弱，正确的小焦距
# 内参会把这种病态暴露出来。
# 在补齐「自标定/EXIF 内参 + 全局特征匹配 + 完整（含相机位姿）BA」之前，
# 保留官方全分辨率内参反而能得到几何更稳定的稀疏点云。
# TODO(A): 确定 640×480 真实内参，并增强重建鲁棒性后再启用缩放。
_OFFICIAL_K = np.array([
    [2759.48, 0, 1520.69],
    [0, 2764.16, 1006.81],
    [0, 0, 1]
])
_OFFICIAL_SIZE = (3072, 2048)  # (宽, 高)
_DATASET_SIZE = (640, 480)     # (宽, 高)


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

    # 相机内参：见文件头说明，当前保留官方全分辨率值
    K = _OFFICIAL_K.copy()

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
