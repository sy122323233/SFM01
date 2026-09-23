#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CUDA 加速版 BA 过滤模块

说明：
本模块不是 scipy least_squares 的完整 BA，
而是对当前项目中稳定有效的“重投影误差过滤式 BA”进行 CUDA 并行加速。

核心思路：
1. CPU 整理 2D-3D 观测关系；
2. GPU 并行计算每个观测的重投影误差；
3. GPU 使用 atomic add 累加每个 3D 点的误差和观测次数；
4. GPU 根据平均误差生成 keep_mask；
5. CPU 根据 keep_mask 过滤点云并同步更新 correspondences。
"""

import math
import numpy as np
from numba import cuda


@cuda.jit
def _compute_reprojection_errors_kernel(
    points_3d,
    rotations,
    motions,
    K,
    camera_indices,
    point_indices,
    points_2d,
    point_errors,
    point_counts
):
    idx = cuda.grid(1)

    if idx >= points_2d.shape[0]:
        return

    cam_id = camera_indices[idx]
    point_id = point_indices[idx]

    if point_id < 0:
        return

    X = points_3d[point_id, 0]
    Y = points_3d[point_id, 1]
    Z = points_3d[point_id, 2]

    r00 = rotations[cam_id, 0, 0]
    r01 = rotations[cam_id, 0, 1]
    r02 = rotations[cam_id, 0, 2]

    r10 = rotations[cam_id, 1, 0]
    r11 = rotations[cam_id, 1, 1]
    r12 = rotations[cam_id, 1, 2]

    r20 = rotations[cam_id, 2, 0]
    r21 = rotations[cam_id, 2, 1]
    r22 = rotations[cam_id, 2, 2]

    tx = motions[cam_id, 0]
    ty = motions[cam_id, 1]
    tz = motions[cam_id, 2]

    Xc = r00 * X + r01 * Y + r02 * Z + tx
    Yc = r10 * X + r11 * Y + r12 * Z + ty
    Zc = r20 * X + r21 * Y + r22 * Z + tz

    if math.fabs(Zc) < 1e-12:
        return

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    u = fx * (Xc / Zc) + cx
    v = fy * (Yc / Zc) + cy

    u_obs = points_2d[idx, 0]
    v_obs = points_2d[idx, 1]

    du = u - u_obs
    dv = v - v_obs

    error = math.sqrt(du * du + dv * dv)

    cuda.atomic.add(point_errors, point_id, error)
    cuda.atomic.add(point_counts, point_id, 1.0)


@cuda.jit
def _build_keep_mask_kernel(
    point_errors,
    point_counts,
    threshold,
    keep_mask
):
    idx = cuda.grid(1)

    if idx >= point_errors.shape[0]:
        return

    if point_counts[idx] <= 0:
        keep_mask[idx] = 0
        return

    avg_error = point_errors[idx] / point_counts[idx]

    if avg_error <= threshold:
        keep_mask[idx] = 1
    else:
        keep_mask[idx] = 0


def optimize_ba_cuda(
    points_3d,
    rotations,
    motions,
    key_points_list,
    correspondences,
    K,
    threshold=0.5
):
    print(" 已进入 CUDA BA：重投影误差并行计算 + 异常点过滤")
    print(" CUDA available:", cuda.is_available())

    if cuda.is_available():
        print(" 当前 GPU:", cuda.get_current_device().name)

    if points_3d is None or len(points_3d) == 0:
        print(" 点云为空，跳过 CUDA BA")
        return points_3d

    if not cuda.is_available():
        raise RuntimeError("当前环境 cuda.is_available() = False，无法使用 CUDA")

    n_points = len(points_3d)
    n_cameras = len(rotations)

    print(f" CUDA BA 输入点数: {n_points}")
    print(f" CUDA BA 相机数量: {n_cameras}")

    camera_indices, point_indices, points_2d = _build_observation_arrays(
        points_3d,
        key_points_list,
        correspondences
    )

    n_observations = len(points_2d)
    print(f" CUDA BA 观测数量: {n_observations}")

    if n_observations == 0:
        print(" 没有有效观测，跳过 CUDA BA")
        return points_3d

    points_3d_np = np.ascontiguousarray(points_3d, dtype=np.float32)

    rotations_np = np.ascontiguousarray(
        np.array(rotations, dtype=np.float32)
    )

    motions_np = np.ascontiguousarray(
        np.array([np.asarray(t).reshape(3) for t in motions], dtype=np.float32)
    )

    K_np = np.ascontiguousarray(K, dtype=np.float32)

    camera_indices_np = np.ascontiguousarray(camera_indices, dtype=np.int32)
    point_indices_np = np.ascontiguousarray(point_indices, dtype=np.int32)
    points_2d_np = np.ascontiguousarray(points_2d, dtype=np.float32)

    point_errors_np = np.zeros(n_points, dtype=np.float32)
    point_counts_np = np.zeros(n_points, dtype=np.float32)
    keep_mask_np = np.zeros(n_points, dtype=np.int32)

    d_points_3d = cuda.to_device(points_3d_np)
    d_rotations = cuda.to_device(rotations_np)
    d_motions = cuda.to_device(motions_np)
    d_K = cuda.to_device(K_np)

    d_camera_indices = cuda.to_device(camera_indices_np)
    d_point_indices = cuda.to_device(point_indices_np)
    d_points_2d = cuda.to_device(points_2d_np)

    d_point_errors = cuda.to_device(point_errors_np)
    d_point_counts = cuda.to_device(point_counts_np)
    d_keep_mask = cuda.to_device(keep_mask_np)

    threads_per_block = 256
    blocks_obs = (n_observations + threads_per_block - 1) // threads_per_block

    _compute_reprojection_errors_kernel[blocks_obs, threads_per_block](
        d_points_3d,
        d_rotations,
        d_motions,
        d_K,
        d_camera_indices,
        d_point_indices,
        d_points_2d,
        d_point_errors,
        d_point_counts
    )

    cuda.synchronize()

    blocks_points = (n_points + threads_per_block - 1) // threads_per_block

    _build_keep_mask_kernel[blocks_points, threads_per_block](
        d_point_errors,
        d_point_counts,
        np.float32(threshold),
        d_keep_mask
    )

    cuda.synchronize()

    keep_mask = d_keep_mask.copy_to_host().astype(bool)

    filtered_points = points_3d[keep_mask]
    removed_count = n_points - len(filtered_points)

    print(f" CUDA BA 过滤前点数: {n_points}")
    print(f" CUDA BA 过滤后点数: {len(filtered_points)}")
    print(f" CUDA BA 移除点数: {removed_count} ({removed_count / max(n_points, 1) * 100:.1f}%)")
    print(f" CUDA BA 阈值: {threshold:.2f} pixels")

    _update_correspondences_after_filter(
        correspondences,
        keep_mask
    )

    print(" CUDA BA 完成")

    return filtered_points


def _build_observation_arrays(points_3d, key_points_list, correspondences):
    camera_indices = []
    point_indices = []
    points_2d = []

    n_points = len(points_3d)

    for cam_id in range(len(correspondences)):
        if cam_id >= len(key_points_list):
            continue

        corr = correspondences[cam_id]
        key_points = key_points_list[cam_id]

        max_len = min(len(corr), len(key_points))

        for j in range(max_len):
            point_id = int(corr[j])

            if point_id < 0 or point_id >= n_points:
                continue

            x, y = key_points[j].pt

            camera_indices.append(cam_id)
            point_indices.append(point_id)
            points_2d.append([x, y])

    return (
        np.array(camera_indices, dtype=np.int32),
        np.array(point_indices, dtype=np.int32),
        np.array(points_2d, dtype=np.float32)
    )


def _update_correspondences_after_filter(correspondences, keep_mask):
    n_points = len(keep_mask)

    old_to_new = -np.ones(n_points, dtype=np.int32)

    new_idx = 0
    for old_idx in range(n_points):
        if keep_mask[old_idx]:
            old_to_new[old_idx] = new_idx
            new_idx += 1

    for cam_id in range(len(correspondences)):
        corr = correspondences[cam_id]

        for j in range(len(corr)):
            old_id = int(corr[j])

            if old_id < 0 or old_id >= n_points:
                corr[j] = -1
            else:
                corr[j] = old_to_new[old_id]