#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
相机内参自动确定模块（不依赖棋盘格/标定板）

按优先级确定内参 K（方形像素、主点在图像中心）：
  1. EXIF 焦距（若有）→ 换算成像素；
  2. 都没有 → 默认 f = 1.0 × 图像宽（约 53° 水平视场角，是手机/相机的常见中庸值）。

说明：曾尝试用基础矩阵 F 做「本质矩阵自标定」（σ1=σ2 约束、或 cheirality+重投影
搜索），但在本课程环拍数据集上两种方法都会落到搜索边界的伪极小值（重投影误差随
f 单调增大、σ1=σ2 在 f=0.5W 处取到伪极小），估计结果不稳定。故弃用自标定，改走
「EXIF → 默认值」这条稳健、可预测的路线，保证不会因为一个错误的 f 破坏重建。
"""

import cv2
import numpy as np


def build_intrinsics(f, W, H):
    """按「方形像素 + 主点在图像中心」构造 3×3 内参矩阵。"""
    return np.array([
        [f, 0, W / 2.0],
        [0, f, H / 2.0],
        [0, 0, 1.0],
    ], dtype=np.float64)


def _exif_float(v):
    """把 EXIF 值（可能是 IFDRational / 元组 / 数字）安全转成 float。"""
    if v is None:
        return None
    if isinstance(v, (tuple, list)) and len(v) == 2:
        return float(v[0]) / float(v[1]) if v[1] else None
    if hasattr(v, 'numerator') and hasattr(v, 'denominator'):
        den = float(v.denominator)
        return float(v.numerator) / den if den else None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def read_exif_focal(image_path, W):
    """从 EXIF 读焦距并换算成像素值，失败返回 None。

    优先用「物理焦距(mm) × FocalPlaneXResolution(px/mm)」；没有则用
    「35mm 等效焦距 / 36 × 宽」近似。
    """
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
        img = Image.open(image_path)
        exif = img._getexif()
        if not exif:
            return None
        tag_map = {TAGS.get(k, k): v for k, v in exif.items()}

        fl_mm = _exif_float(tag_map.get('FocalLength'))
        res = _exif_float(tag_map.get('FocalPlaneXResolution'))
        if fl_mm and res:
            return fl_mm * res

        fl35 = _exif_float(tag_map.get('FocalLengthIn35mmFilm'))
        if fl35:
            return fl35 / 36.0 * W
        return None
    except Exception:
        return None


def auto_intrinsics(image_path, W, H, default_f_ratio=1.0):
    """自动确定内参 K。优先级：EXIF → 默认(fx=fy=1.0W)。

    Args:
        image_path: 第一张图的路径（读 EXIF 用）。
        W, H: 图像宽高。
        default_f_ratio: 无 EXIF 时用的默认焦距相对图像宽的比例（默认 1.0）。
    Returns:
        (K, source) —— K 为 3×3 内参，source 为描述来源的字符串。
    """
    # 焦距的合理范围（相对图像宽）：太离谱的值（EXIF 解析错）一律不采信
    def _plausible(fv):
        return fv is not None and 0.3 * W <= fv <= 3.0 * W

    f = read_exif_focal(image_path, W)
    source = "EXIF"
    if not _plausible(f):
        f = default_f_ratio * W
        source = f"默认({default_f_ratio}W)"
    return build_intrinsics(f, W, H), source
