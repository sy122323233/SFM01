#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
稳定版 Bundle Adjustment

特点：
1. 先进行重投影误差过滤，清理明显异常点；
2. 固定相机位姿 R、t，不优化相机；
3. 仅优化 3D 点坐标，降低点云被拉烂的风险；
4. 使用 soft_l1 鲁棒损失，减少异常观测影响；
5. 优化完成后再次过滤异常点并同步 correspondence。
"""

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


class BundleAdjustment:
    def __init__(self, K):
        self.K = K.astype(np.float64)

        # 第一轮过滤阈值：沿用你之前效果好的 0.5
        self.pre_filter_threshold = 0.5

        # 点优化后的过滤阈值
        self.post_filter_threshold = 1.0

        # 最大迭代次数，先别太大
        self.max_iterations = 30

        # 最多优化多少个 3D 点
        # 如果你想全部优化，可以改成 None
        self.max_ba_points = 8000

        self.verbose = True

    def optimize(self, points_3d, rotations, motions, key_points_list, correspondences):
        print(" 开始稳定版 BA：先过滤，再固定相机优化 3D 点...")

        if points_3d is None or len(points_3d) == 0:
            print(" 点云为空，跳过 BA")
            return points_3d

        # 1. 先用原来的方式过滤异常点
        filtered_points = self._filter_points(
            points_3d,
            rotations,
            motions,
            key_points_list,
            correspondences,
            threshold=self.pre_filter_threshold,
            title="BA 前预过滤"
        )

        if len(filtered_points) == 0:
            print(" 预过滤后点云为空，返回原始点云")
            return points_3d

        # 2. 收集过滤后的观测
        camera_indices, point_indices, points_2d = self._build_observations(
            filtered_points,
            key_points_list,
            correspondences
        )

        if len(points_2d) == 0:
            print(" 没有有效观测，跳过点优化")
            return filtered_points

        print(f" 预过滤后点数: {len(filtered_points)}")
        print(f" 有效观测数量: {len(points_2d)}")

        # 3. 选择参与 BA 的点
        active_point_ids = self._select_ba_points(
            len(filtered_points),
            point_indices,
            self.max_ba_points
        )

        active_mask = np.zeros(len(filtered_points), dtype=bool)
        active_mask[active_point_ids] = True

        obs_mask = active_mask[point_indices]

        camera_indices_ba = camera_indices[obs_mask]
        point_indices_old = point_indices[obs_mask]
        points_2d_ba = points_2d[obs_mask]

        old_to_ba = -np.ones(len(filtered_points), dtype=np.int32)
        for new_id, old_id in enumerate(active_point_ids):
            old_to_ba[old_id] = new_id

        point_indices_ba = old_to_ba[point_indices_old]

        points_ba = filtered_points[active_point_ids].astype(np.float64)

        print(f" 参与点优化的 3D 点数: {len(points_ba)}")
        print(f" 参与点优化的观测数: {len(points_2d_ba)}")

        if len(points_ba) == 0 or len(points_2d_ba) == 0:
            print(" 参与 BA 的数据为空，跳过点优化")
            return filtered_points

        # 4. 固定相机，只优化点坐标
        x0 = points_ba.ravel()

        residual_before = self._point_only_residuals(
            x0,
            rotations,
            motions,
            camera_indices_ba,
            point_indices_ba,
            points_2d_ba
        )

        print(f" 点优化前平均重投影误差: {self._mean_error(residual_before):.4f} pixels")

        sparsity = self._point_only_sparsity(
            len(points_ba),
            point_indices_ba
        )

        result = least_squares(
            self._point_only_residuals,
            x0,
            jac_sparsity=sparsity,
            verbose=2 if self.verbose else 0,
            x_scale='jac',
            ftol=1e-5,
            xtol=1e-5,
            gtol=1e-5,
            method='trf',
            loss='soft_l1',
            f_scale=1.0,
            max_nfev=self.max_iterations,
            args=(
                rotations,
                motions,
                camera_indices_ba,
                point_indices_ba,
                points_2d_ba
            )
        )

        print(" 固定相机点优化结束")
        print(f" 是否收敛: {result.success}")
        print(f" 终止原因: {result.message}")

        residual_after = self._point_only_residuals(
            result.x,
            rotations,
            motions,
            camera_indices_ba,
            point_indices_ba,
            points_2d_ba
        )

        print(f" 点优化后平均重投影误差: {self._mean_error(residual_after):.4f} pixels")

        # 5. 更新参与优化的点
        optimized_points = filtered_points.copy()
        optimized_points[active_point_ids] = result.x.reshape((-1, 3))

        # 6. 后过滤
        final_points = self._filter_points(
            optimized_points,
            rotations,
            motions,
            key_points_list,
            correspondences,
            threshold=self.post_filter_threshold,
            title="BA 后过滤"
        )

        print(" 稳定版 BA 完成")
        return final_points

    def _build_observations(self, points_3d, key_points_list, correspondences):
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
            np.array(points_2d, dtype=np.float64)
        )

    def _select_ba_points(self, n_points, point_indices, max_points):
        counts = np.bincount(point_indices, minlength=n_points)
        observed_ids = np.where(counts > 0)[0]

        if max_points is None or len(observed_ids) <= max_points:
            return observed_ids

        sorted_ids = observed_ids[np.argsort(counts[observed_ids])[::-1]]
        selected = sorted_ids[:max_points]

        print(f" 点数较多，仅选择观测次数最多的 {max_points} 个点参与优化")

        return np.sort(selected)

    def _point_only_residuals(
        self,
        params,
        rotations,
        motions,
        camera_indices,
        point_indices,
        points_2d
    ):
        points_3d = params.reshape((-1, 3))

        residuals = np.zeros((len(points_2d), 2), dtype=np.float64)

        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]

        for i in range(len(points_2d)):
            cam_id = camera_indices[i]
            point_id = point_indices[i]

            R = rotations[cam_id]
            t = np.asarray(motions[cam_id]).reshape(3)

            X = points_3d[point_id]
            X_cam = R.dot(X) + t

            z = X_cam[2]

            if abs(z) < 1e-12:
                z = 1e-12

            u = fx * (X_cam[0] / z) + cx
            v = fy * (X_cam[1] / z) + cy

            residuals[i, 0] = u - points_2d[i, 0]
            residuals[i, 1] = v - points_2d[i, 1]

        return residuals.ravel()

    def _point_only_sparsity(self, n_points, point_indices):
        n_obs = len(point_indices)

        m = n_obs * 2
        n = n_points * 3

        A = lil_matrix((m, n), dtype=int)
        i = np.arange(n_obs)

        for s in range(3):
            A[2 * i, point_indices * 3 + s] = 1
            A[2 * i + 1, point_indices * 3 + s] = 1

        return A

    def _mean_error(self, residuals):
        residuals = residuals.reshape((-1, 2))
        errors = np.linalg.norm(residuals, axis=1)

        if len(errors) == 0:
            return float('inf')

        return np.mean(errors)

    def _filter_points(
        self,
        points_3d,
        rotations,
        motions,
        key_points_list,
        correspondences,
        threshold,
        title
    ):
        print(f" {title}：开始计算重投影误差并过滤异常点...")

        n_points = len(points_3d)

        point_errors = np.zeros(n_points, dtype=np.float64)
        point_count = np.zeros(n_points, dtype=np.float64)

        for cam_id in range(len(rotations)):
            if cam_id >= len(key_points_list):
                continue

            R = rotations[cam_id]
            t = np.asarray(motions[cam_id]).reshape(3, 1)

            rvec, _ = cv2.Rodrigues(R)

            if cam_id >= len(correspondences):
                continue

            corr = correspondences[cam_id]
            key_points = key_points_list[cam_id]
            max_len = min(len(corr), len(key_points))

            for j in range(max_len):
                point_id = int(corr[j])

                if point_id < 0 or point_id >= n_points:
                    continue

                point3d = points_3d[point_id].reshape(1, 1, 3)
                point2d_obs = np.array(key_points[j].pt, dtype=np.float64)

                point2d_proj, _ = cv2.projectPoints(
                    point3d,
                    rvec,
                    t,
                    self.K,
                    np.array([])
                )

                point2d_proj = point2d_proj.reshape(2)

                error = np.linalg.norm(point2d_obs - point2d_proj)

                point_errors[point_id] += error
                point_count[point_id] += 1

        keep_mask = np.zeros(n_points, dtype=bool)

        for i in range(n_points):
            if point_count[i] > 0:
                avg_error = point_errors[i] / point_count[i]
                if avg_error <= threshold:
                    keep_mask[i] = True

        filtered_points = points_3d[keep_mask]

        removed_count = n_points - len(filtered_points)

        print(f" {title}：过滤前点数: {n_points}")
        print(f" {title}：过滤后点数: {len(filtered_points)}")
        print(f" {title}：移除点数: {removed_count} ({removed_count / max(n_points, 1) * 100:.1f}%)")
        print(f" {title}：阈值: {threshold:.2f} pixels")

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

        return filtered_points
    def optimize_cuda(self, points_3d, rotations, motions, key_points_list, correspondences):
        """
        CUDA 版本 BA：后续将在这里接入 CUDA 加速
        当前阶段先保留 CPU fallback，保证程序不会崩
        """
        print(" CUDA 版本 BA 接口已调用")

        try:
            from core.bundle_adjustment_cuda import optimize_ba_cuda

            filtered_points = optimize_ba_cuda(
                points_3d,
                rotations,
                motions,
                key_points_list,
                correspondences,
                self.K
            )

            print(" CUDA BA 完成")
            return filtered_points

        except Exception as e:
            print(f" CUDA BA 运行失败，自动回退到 CPU 版本: {e}")
            return self.optimize(
                points_3d,
                rotations,
                motions,
                key_points_list,
                correspondences
            )