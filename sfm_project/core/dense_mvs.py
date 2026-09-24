#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
密集重建模块（Plane-Sweep Stereo，自研，不依赖 COLMAP）

把「稀疏 SfM 已给出的相机位姿 (R,t) + 内参 K + 原始图像」升级成**逐像素稠密点云**。

原理（fronto-parallel 平面扫描）：
  1. 选一张参考图，在它前方扫一排深度平面 d_1..d_N；
  2. 深度为 d 的平面在两个相机之间诱导一个单应变换 H(d) = K (d·R_rel + t_rel·nᵀ) K⁻¹，
     把其它视角的图像 warp 到参考视角；
  3. 对每个像素，在所有视角里比颜色一致性（NCC），一致性最好的深度就是该像素深度；
  4. 深度图反投影回 3D，得到稠密点云（可直接喂给 core.meshing.points_to_mesh）。

坐标约定（与 core.reconstruction 一致）：
  - rotations[i]：世界→相机 3×3 旋转矩阵
  - motions[i]  ：世界→相机 3×1 平移向量
  - 投影：x = K (R X + t)，相机中心 C = -Rᵀ t；第 0 帧是世界原点。

注意：本模块使用与 SfM **同一套 K 和位姿**，保证稠密结果与稀疏点云内部自洽。
内参 K 的标定问题（config.py 头注释所述）是另一个独立事项，会一起继承/一起修复。
"""

import os
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import median_filter


# ============================ 纯几何工具 ============================

def _pose_to_relative(R_ref, t_ref, R_src, t_src):
    """世界→相机位姿 转成 参考→源 的相对位姿。"""
    R_rel = R_src @ R_ref.T
    t_rel = t_src - R_rel @ t_ref
    return R_rel, t_rel


def plane_sweep_homography(K, R_rel, t_rel, depth):
    """fronto-parallel 平面（参考相机 z=depth）诱导的单应矩阵。

    H = K (depth·R_rel + t_rel·nᵀ) K⁻¹，n = (0,0,1)ᵀ
    把参考图齐次像素映射到源图齐次像素：p_src ∝ H @ p_ref。
    """
    n = np.array([0.0, 0.0, 1.0])
    H = K @ (depth * R_rel + t_rel.reshape(3, 1) @ n.reshape(1, 3)) @ np.linalg.inv(K)
    return H


def depth_range_from_points(points, R_ref, t_ref, lo=2.0, hi=98.0, margin=0.15):
    """从稀疏点云推算参考相机前方的深度扫描范围 [near, far]。"""
    Xc = (R_ref @ points.T).T + t_ref.reshape(1, 3)  # (N,3) 参考相机系
    z = Xc[:, 2]
    z = z[z > 0]
    if len(z) < 10:
        raise ValueError("参考相机前方的点太少，无法推断深度范围")
    near = float(np.percentile(z, lo))
    far = float(np.percentile(z, hi))
    span = far - near
    near = max(near - span * margin, 1e-3)
    far = far + span * margin
    return near, far


def _homography_self_test():
    """用「纯平移相机」的解析解校验单应公式，防止符号/转置出错。"""
    K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])
    R_ref = np.eye(3)
    t_ref = np.zeros((3, 1))
    R_src = np.eye(3)
    t_src = np.array([[0.5], [0.0], [0.0]])  # 源相机沿 x 平移 0.5

    d = 3.0
    R_rel, t_rel = _pose_to_relative(R_ref, t_ref, R_src, t_src)
    H = plane_sweep_homography(K, R_rel, t_rel, d)

    # 参考像素 (u,v) 在深度 d 的 3D 点
    u, v = 400.0, 300.0
    X_ref = d * (np.linalg.inv(K) @ np.array([u, v, 1.0]))
    # 源像素 = K (X_ref + t)
    p_src = K @ (X_ref + t_src.flatten())
    u_src, v_src = p_src[0] / p_src[2], p_src[1] / p_src[2]

    # 单应版本
    p_h = H @ np.array([u, v, 1.0])
    u_h, v_h = p_h[0] / p_h[2], p_h[1] / p_h[2]

    err = abs(u_src - u_h) + abs(v_src - v_h)
    assert err < 1e-6, f"homography self-test failed, err={err}"
    print(f"[self-test] 单应公式校验通过 (err={err:.2e})")


# ============================ 稠密重建 ============================

@dataclass
class DenseResult:
    """稠密重建的统一结果结构（对外 API 返回该对象）。"""
    points: np.ndarray          # (M,3) 稠密点（多视图融合后）
    colors: np.ndarray          # (M,3) RGB 0-255
    depth_map: np.ndarray       # (H,W) 主视图逐像素深度（仅用于出图）
    cost_map: np.ndarray        # (H,W) 主视图最小 NCC 成本
    near: float                 # 主视图深度范围
    far: float
    ref_idx: int                # 主视图索引（出图用）
    ref_indices: list           # 实际参与扫描的所有参考视图索引
    scale: float                # 工作分辨率相对原图的缩放

    @property
    def num_points(self):
        return int(len(self.points))

    @property
    def num_views(self):
        return int(len(self.ref_indices))

    def to_dict(self):
        return {
            'num_points': self.num_points,
            'num_views': self.num_views,
            'ref_indices': [int(i) for i in self.ref_indices],
            'near': float(self.near),
            'far': float(self.far),
            'scale': float(self.scale),
        }


def estimate_depth_map(ref_img, src_imgs, K, R_ref, t_ref, Rs, ts, depths,
                       patch=5, device=None):
    """plane-sweep：对参考图逐像素求深度（NCC 一致性）。

    Args:
        ref_img: (H,W,3) 参考图（BGR 或 RGB 均可，内部统一成 0-1 float）
        src_imgs: [(H,W,3), ...] 源图列表
        K: 3×3 内参（与图像分辨率匹配，已经过缩放）
        R_ref, t_ref: 参考相机位姿
        Rs, ts: 各源相机位姿列表
        depths: (N,) 深度平面列表
        patch: 一致性窗口边长
    Returns:
        depth_map (H,W) 亚像素细化后的深度, cost_map (H,W) argmin 处 NCC 成本,
        peak_map (H,W) 极小值锐利度（深度是否唯一可靠）
    """
    import torch
    import torch.nn.functional as F

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    H_img, W_img = ref_img.shape[:2]
    ref = torch.from_numpy(ref_img.astype(np.float32) / 255.0).permute(2, 0, 1).to(device)
    srcs = [torch.from_numpy(s.astype(np.float32) / 255.0).permute(2, 0, 1).to(device)
            for s in src_imgs]

    # 参考图齐次像素坐标 (3,H,W)
    u = torch.arange(W_img, device=device, dtype=torch.float32)
    v = torch.arange(H_img, device=device, dtype=torch.float32)
    vv, uu = torch.meshgrid(v, u, indexing='ij')
    p_ref = torch.stack([uu, vv, torch.ones_like(uu)], dim=0)

    Kt = torch.from_numpy(K.astype(np.float32)).to(device)
    Kinv = torch.from_numpy(np.linalg.inv(K).astype(np.float32)).to(device)
    R_ref_t = torch.from_numpy(np.asarray(R_ref, np.float32)).to(device)
    t_ref_t = torch.from_numpy(np.asarray(t_ref, np.float32)).reshape(3).to(device)
    n = torch.tensor([0.0, 0.0, 1.0], device=device)

    # 预计算每个源的相对位姿
    src_data = []
    for s, R_s, t_s in zip(srcs, Rs, ts):
        R_st = torch.from_numpy(np.asarray(R_s, np.float32)).to(device)
        t_st = torch.from_numpy(np.asarray(t_s, np.float32)).reshape(3).to(device)
        R_rel = R_st @ R_ref_t.T
        t_rel = t_st - R_rel @ t_ref_t
        src_data.append((s, R_rel, t_rel))

    num_d = len(depths)
    pad = patch // 2
    cost_vol = torch.zeros((num_d, H_img, W_img), device=device)

    # NCC（归一化互相关）一致性：patch 内先零均值 + 归一化再算相关系数，
    # 对亮度/对比度差异不敏感，cost 极小值比 NCC 更「尖」→ 深度更准、平面更平。
    # 参考图的 patch 统计只算一次。
    ref_mean = F.avg_pool2d(ref, patch, stride=1, padding=pad)          # (3,H,W)
    ref_c = ref - ref_mean
    ref_std = torch.sqrt(F.avg_pool2d(ref_c.pow(2), patch, stride=1, padding=pad) + 1e-6)

    for di, d in enumerate(depths):
        d_t = torch.tensor(float(d), device=device)
        cost = torch.zeros((H_img, W_img), device=device)
        for s, R_rel, t_rel in src_data:
            H = Kt @ (d_t * R_rel + torch.outer(t_rel, n)) @ Kinv      # (3,3)
            p_src = torch.einsum('ij,jhw->ihw', H, p_ref)              # (3,H,W)
            xs = p_src[0] / p_src[2]
            ys = p_src[1] / p_src[2]
            xs_n = 2.0 * xs / (W_img - 1) - 1.0
            ys_n = 2.0 * ys / (H_img - 1) - 1.0
            grid = torch.stack([xs_n, ys_n], dim=-1).unsqueeze(0)      # (1,H,W,2)
            warped = F.grid_sample(s.unsqueeze(0), grid, mode='bilinear',
                                   padding_mode='zeros', align_corners=True).squeeze(0)  # (3,H,W)
            wm = F.avg_pool2d(warped, patch, stride=1, padding=pad)
            wc = warped - wm
            wstd = torch.sqrt(F.avg_pool2d(wc.pow(2), patch, stride=1, padding=pad) + 1e-6)
            cross = F.avg_pool2d(ref_c * wc, patch, stride=1, padding=pad)  # (3,H,W)
            ncc = (cross / (ref_std * wstd)).mean(dim=0)               # (H,W) ∈ [-1,1]
            cost += (1.0 - ncc)                                         # 0=一致, 2=相反
        cost_vol[di] = cost

    # ---- 亚像素深度细化：在 cost 曲线上做抛物线拟合（逆深度域插值）----
    cost_vol = cost_vol.cpu().numpy()  # (num_d, H, W)
    idx = np.argmin(cost_vol, axis=0)  # (H, W) 整数 argmin
    c0 = np.take_along_axis(cost_vol, idx[None], axis=0)[0]
    cm = np.take_along_axis(cost_vol, np.clip(idx - 1, 0, num_d - 1)[None], axis=0)[0]
    cp = np.take_along_axis(cost_vol, np.clip(idx + 1, 0, num_d - 1)[None], axis=0)[0]
    denom = cm - 2.0 * c0 + cp
    denom = np.where(np.abs(denom) < 1e-10, 1.0, denom)
    delta = np.clip(0.5 * (cm - cp) / denom, -1.0, 1.0)
    refined_i = idx.astype(np.float32) + delta

    # 深度是按「逆深度」均匀采样的，所以在逆深度域做线性插值
    inv_depths = 1.0 / np.asarray(depths, np.float64)
    i0 = np.clip(np.floor(refined_i).astype(int), 0, num_d - 2)
    frac = refined_i - i0
    inv_refined = inv_depths[i0] * (1.0 - frac) + inv_depths[i0 + 1] * frac
    depth_map = (1.0 / inv_refined).astype(np.float32)

    # 极小值「锐利度」：argmin 与左右相邻深度的成本差。差大=深度唯一可靠；
    # 差小=各深度平面成本都差不多（低纹理/遮挡/斜射），深度不可靠 → 后续剔除。
    peak_map = np.minimum(cm - c0, cp - c0).astype(np.float32)

    return depth_map, c0, peak_map


def backproject_depth(ref_img, depth_map, valid, K, R_ref, t_ref):
    """深度图反投影成 3D 点云 + 颜色（RGB 0-255）。"""
    H, W = depth_map.shape
    Kinv = np.linalg.inv(K)
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    ones = np.ones_like(u)
    pts_ref = np.stack([u.ravel(), v.ravel(), ones.ravel()], axis=0)   # (3, H*W)
    X_ref = (Kinv @ pts_ref) * depth_map.ravel()[None, :]              # (3, H*W)
    X_world = (R_ref.T @ (X_ref - t_ref.reshape(3, 1))).T              # (H*W, 3)
    colors = ref_img[..., ::-1].reshape(-1, 3)                          # BGR→RGB
    mask = valid.ravel()
    return X_world[mask], colors[mask]


def _check_consistency(depth_r, K, R_r, t_r, R_s, t_s, depth_s, threshold=0.02):
    """左右一致性校验（LRC）：视图 r 每个像素的深度，反投影到世界再投影到视图 s，
    与视图 s 的深度图比对。相对深度误差超过 threshold 的判为不一致（飞点/误匹配）。

    Returns:
        (H,W) bool 掩码，True=通过校验
    """
    H, W = depth_r.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    Kinv = np.linalg.inv(K)
    pts = np.stack([u.ravel(), v.ravel(), np.ones(u.size)], axis=0)      # (3, N)
    X = Kinv @ pts * depth_r.ravel()[None, :]                             # (3, N) r 相机系
    Xw = R_r.T @ (X - t_r.reshape(3, 1))                                  # (3, N) 世界系
    Xs = R_s @ Xw + t_s.reshape(3, 1)                                     # (3, N) s 相机系
    zs = Xs[2]
    q = K @ Xs
    us = q[0] / q[2]
    vs = q[1] / q[2]
    inside = (us >= 0) & (us <= W - 1) & (vs >= 0) & (vs <= H - 1) & (zs > 0)
    us_i = np.clip(np.round(us).astype(int), 0, W - 1)
    vs_i = np.clip(np.round(vs).astype(int), 0, H - 1)
    d_s = depth_s[vs_i, us_i]
    rel_err = np.abs(d_s - zs) / np.maximum(zs, 1e-6)
    ok = inside & (d_s > 0) & (rel_err < threshold)
    return ok.reshape(H, W)


def _viewing_angle_mask(depth_map, K, max_angle_deg=60.0):
    """剔除「斜射」像素：表面法向与视线夹角过大的像素深度不可靠。

    MVS 里每个视角只在「正对表面」的方向上深度准，斜射方向（物体侧面）最糊；
    6 视角融合时这些斜射残渣就是「侧面重影」的来源。这里用深度图的差分法向
    与视线方向算夹角，夹角超过 max_angle_deg 的像素判为斜射 → 剔除。

    Returns:
        (H,W) bool 掩码，True=正对（可靠）
    """
    H, W = depth_map.shape
    Kinv = np.linalg.inv(K)
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    pts = np.stack([u.ravel(), v.ravel(), np.ones(u.size)], axis=0)   # (3,N)
    X = (Kinv @ pts) * depth_map.ravel()[None, :]                      # (3,N) 相机系
    X = X.reshape(3, H, W)

    # 表面法向 ≈ cross(dX/du, dX/dv)，中心差分（边界置 0，稍后一并被边缘掩码排除）
    du = np.zeros_like(X)
    dv = np.zeros_like(X)
    du[:, :, 1:-1] = (X[:, :, 2:] - X[:, :, :-2]) * 0.5
    dv[:, 1:-1, :] = (X[:, 2:, :] - X[:, :-2, :]) * 0.5
    N = np.cross(du, dv, axis=0)                                       # (3,H,W)

    # 视线方向 = 相机指向点（相机系原点出发，即 X）；|N·视线| 越大越正对
    cosang = np.abs(np.sum(N * X, axis=0)) / (
        np.linalg.norm(N, axis=0) * np.linalg.norm(X, axis=0) + 1e-9)
    return cosang > np.cos(np.deg2rad(max_angle_deg))


def dense_reconstruct(images, rotations, motions, K, sparse_points=None,
                      ref_indices=None, num_ref_views=5,
                      num_depths=64, max_dim=720,
                      patch=5, cost_quantile=0.5, texture_quantile=0.5,
                      peak_quantile=0.3, viewing_angle=60.0, median_ksize=5,
                      lrc=True, lrc_threshold=0.02, border=0.05,
                      sor=True, sor_std_ratio=1.5,
                      depth_range=None, device=None):
    """稠密重建编排入口（多视图 plane-sweep 融合）。

    单张参考图只能得到「它这一面」的深度图（一张 2.5D 曲面），直接喂 Poisson 会被
    强制闭合成一坨。所以这里对多张参考图各扫一遍、把点云融合成 360° 覆盖后再出网格。

    Args:
        images: 图像列表（BGR，与 rotations/motions 严格同序）
        rotations: 世界→相机 R 列表
        motions: 世界→相机 t 列表
        K: 相机内参（与位姿同源，保持一致）
        sparse_points: (N,3) 稀疏点，仅用于推算每张参考图的深度范围（传 depth_range 可省略）
        ref_indices: 参考图索引列表；None 时按 num_ref_views 在序列上均匀取
        num_ref_views: 自动选参考图的数量
        num_depths: 每视图的深度平面数量
        max_dim: 工作分辨率最长边（先降采样提速，内参同步缩放）
        patch: 一致性窗口边长
        cost_quantile: NCC 成本阈值分位（越小保留越少但越干净）
        texture_quantile: 纹理阈值分位——Sobel 梯度低于该分位的像素（天空/背景/地板）无深度约束，直接剔除
        peak_quantile: 深度极小值「锐利度」阈值分位——锐利度低（深度平坦/斜射/遮挡）的像素剔除
        viewing_angle: 正对角度阈值（度）——法向与视线夹角超过该值的斜射像素剔除（90 表示关闭）
        median_ksize: 深度图中值滤波核大小（去边缘飞点，0 关闭）
        lrc: 是否做左右一致性校验（每个视图用下一个视图的深度图交叉验证，去飞点/误匹配）
        lrc_threshold: LRC 相对深度误差阈值
        border: 边缘剔除比例
        sor: 融合后是否做统计离群点剔除（去表面外飞点/斜射残渣）
        sor_std_ratio: SOR 的 std_ratio（越小剔除越狠）
        depth_range: (near, far) 手动指定（作用于所有视图）
    Returns:
        DenseResult
    """
    n = len(images)
    if n < 2:
        raise ValueError("密集重建至少需要 2 张图像")
    if len(rotations) != n or len(motions) != n:
        raise ValueError(f"images/rotations/motions 数量不一致: {n}/{len(rotations)}/{len(motions)}")

    Rs = [np.asarray(r, np.float64) for r in rotations]
    ts = [np.asarray(t, np.float64).reshape(3, 1) for t in motions]

    # 降采样 + 缩放内参（位姿是世界系，不随分辨率变）
    H0, W0 = images[0].shape[:2]
    scale = min(1.0, max_dim / max(H0, W0))
    W, H = int(round(W0 * scale)), int(round(H0 * scale))
    K_s = K.astype(np.float64).copy()
    K_s[0, 0] *= scale
    K_s[1, 1] *= scale
    K_s[0, 2] *= scale
    K_s[1, 2] *= scale

    imgs = [cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA) for im in images]

    # 选参考视图：默认在序列上均匀取，覆盖 360°
    if ref_indices is None:
        ref_indices = np.unique(np.linspace(0, n - 1, num_ref_views).astype(int)).tolist()
    ref_indices = [int(i) for i in ref_indices]
    primary_ref = ref_indices[len(ref_indices) // 2]   # 中间那张当主视图（出图用）

    sparse = np.asarray(sparse_points, np.float64) if sparse_points is not None else None

    # ---- 第一遍：对每个参考视图做平面扫描，得到深度图 + 有效掩码 ----
    views = []  # 每项 dict: depth / cost / valid / near / far
    for ref_idx in ref_indices:
        if depth_range is None:
            if sparse is None:
                raise ValueError("需要 depth_range 或 sparse_points 来定深度范围")
            near, far = depth_range_from_points(sparse, Rs[ref_idx], ts[ref_idx])
        else:
            near, far = depth_range
        inv = np.linspace(1.0 / far, 1.0 / near, num_depths)
        depths = 1.0 / inv  # 从远到近

        src_idx = [i for i in range(n) if i != ref_idx]
        depth_map, cost_map, peak_map = estimate_depth_map(
            imgs[ref_idx], [imgs[i] for i in src_idx], K_s,
            Rs[ref_idx], ts[ref_idx],
            [Rs[i] for i in src_idx], [ts[i] for i in src_idx],
            depths, patch=patch, device=device)

        # 深度图中值滤波：去掉深度不连续处的"飞点"（边缘误匹配的孤立像素）
        if median_ksize > 0:
            depth_map = median_filter(depth_map, size=median_ksize)

        # 有效性掩码 = 有纹理 AND 低成本 AND 非边缘
        gray = cv2.cvtColor(imgs[ref_idx], cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.GaussianBlur(cv2.magnitude(gx, gy), (5, 5), 0)
        valid = grad > np.quantile(grad, texture_quantile)
        valid &= cost_map < np.quantile(cost_map, cost_quantile)
        if peak_quantile is not None:
            valid &= peak_map > np.quantile(peak_map, peak_quantile)
        if viewing_angle < 90.0:
            valid &= _viewing_angle_mask(depth_map, K_s, max_angle_deg=viewing_angle)
        b = int(border * min(H, W))
        if b > 0:
            valid[:b, :] = False
            valid[-b:, :] = False
            valid[:, :b] = False
            valid[:, -b:] = False

        views.append(dict(depth=depth_map, cost=cost_map, valid=valid, near=near, far=far))

    # ---- 第二遍：左右一致性校验（每个视图用下一个视图的深度图交叉验证）----
    if lrc:
        nv = len(views)
        for k in range(nv):
            kn = (k + 1) % nv  # 下一个参考视图（环形闭合）
            ok = _check_consistency(
                views[k]['depth'], K_s,
                Rs[ref_indices[k]], ts[ref_indices[k]],
                Rs[ref_indices[kn]], ts[ref_indices[kn]],
                views[kn]['depth'], threshold=lrc_threshold)
            views[k]['valid'] = views[k]['valid'] & ok

    # ---- 第三遍：反投影 + 融合 ----
    pts_list, col_list = [], []
    primary_depth = primary_cost = None
    primary_near = primary_far = None
    for k, ref_idx in enumerate(ref_indices):
        v = views[k]
        pts, cols = backproject_depth(
            imgs[ref_idx], v['depth'], v['valid'], K_s, Rs[ref_idx], ts[ref_idx])
        pts_list.append(pts)
        col_list.append(cols)
        if ref_idx == primary_ref:
            primary_depth, primary_cost = v['depth'], v['cost']
            primary_near, primary_far = v['near'], v['far']

    points = np.vstack(pts_list) if pts_list else np.empty((0, 3))
    colors = np.vstack(col_list) if col_list else np.empty((0, 3))

    # 统计离群点剔除（SOR）：融合后表面外会残留「碎碎」的飞点/斜射残渣，
    # 用 k 近邻平均距离的统计阈值把它们去掉，只保留密实的真实表面。
    if sor and len(points) > 3:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if len(colors) == len(points):
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=sor_std_ratio)
        points = np.asarray(pcd.points)
        if pcd.has_colors():
            colors = (np.asarray(pcd.colors) * 255.0).astype(np.uint8)

    return DenseResult(points=points, colors=colors, depth_map=primary_depth,
                       cost_map=primary_cost, near=primary_near, far=primary_far,
                       ref_idx=primary_ref, ref_indices=ref_indices, scale=scale)


# ============================ 演示入口 ============================

def _save_depth_map_viz(result, out_dir):
    """把深度图存成 PNG（turbo 伪彩色）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dm = result.depth_map.astype(np.float64)
    dm = np.where(dm > 0, dm, np.nan)
    plt.figure(figsize=(8, 6))
    plt.imshow(dm, cmap='turbo')
    plt.colorbar(label='depth')
    plt.title(f"depth map (ref={result.ref_idx}, near={result.near:.2f}, far={result.far:.2f})")
    plt.axis('off')
    path = os.path.join(out_dir, "depth_map.png")
    plt.savefig(path, dpi=110, bbox_inches='tight')
    plt.close()
    print(f"saved depth map: {path}")


if __name__ == "__main__":
    import sys
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # sfm_project
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from config import Config
    from main import reconstruct
    from core.meshing import points_to_mesh

    _homography_self_test()

    config = Config()
    print("图片目录:", config.DATA_PATH)

    # 1. 稀疏 SfM：拿位姿 + 稀疏点
    result = reconstruct(config.DATA_PATH, config=config, save=False, visualize=False)
    print(f"\n稀疏 SfM: {result.num_points} 点 / {result.num_images} 帧")

    # 2. 按同样顺序加载图像（与 _load_images 的排序规则一致）
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    names = sorted([f for f in os.listdir(config.DATA_PATH)
                    if f.lower().endswith(exts)])
    images = [cv2.imread(os.path.join(config.DATA_PATH, n)) for n in names]
    print("加载图像:", len(images), "张, 尺寸:", images[0].shape[:2][::-1])

    # 跳过坏帧后「图像全集」与「已注册位姿」不再一一对应：
    # 用 ReconstructionResult.registered_indices 把图像对齐到位姿。
    idx = result.registered_indices
    if idx is None or len(idx) != len(result.rotations):
        idx = list(range(len(result.rotations)))
    images = [images[i] for i in idx]
    print(f"参与稠密重建: {len(images)} 帧（图像索引 {idx}）")

    # 3. 稠密重建（多视图融合）
    dense = dense_reconstruct(images, result.rotations, result.motions, config.K,
                              sparse_points=result.points,
                              num_ref_views=6, num_depths=256, max_dim=720,
                              viewing_angle=70.0, peak_quantile=0.2, median_ksize=7)
    print(f"\n稠密重建: {dense.num_points} 点 / {dense.num_views} 个参考视图  "
          f"(near={dense.near:.2f}, far={dense.far:.2f})")

    # 4. 保存稠密点云 + 深度图
    out_dir = os.path.join(config.OUTPUT_PATH, "dense")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "dense_points.npy"), dense.points)
    np.save(os.path.join(out_dir, "dense_colors.npy"), dense.colors)
    _save_depth_map_viz(dense, out_dir)

    # 稠密点云 PLY（二进制，MeshLab 直接看——点云本身应该能认出物体形状）
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(dense.points)
    pcd.colors = o3d.utility.Vector3dVector(dense.colors / 255.0)
    o3d.io.write_point_cloud(os.path.join(out_dir, "dense_cloud.ply"), pcd)
    print(f"dense point cloud PLY: {len(dense.points)} 点 → dense_cloud.ply")

    # 5. 网格化（复用 core.meshing）
    mesh = points_to_mesh(dense.points, dense.colors,
                          out_dir=os.path.join(out_dir, "mesh"))

    print("\n" + "=" * 50)
    print(f"完成！稠密网格: {len(mesh.vertices)} 顶点 / {len(mesh.triangles)} 面")
    print(f"（多视图融合 {dense.num_views} 个参考视图；稠密点 {dense.num_points} vs 稀疏 {result.num_points}）")
    print(f"GLB: {os.path.join(out_dir, 'mesh', 'mesh.glb')}")
    print(f"稠密点云 PLY: {os.path.join(out_dir, 'dense_cloud.ply')}")
    print("=" * 50)
