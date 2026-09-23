#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
三维重建模块
"""

import cv2
import numpy as np


class Triangulator:
    """三角测量器"""

    def __init__(self, K):
        self.K = K

    def triangulate(self, R1, t1, R2, t2, pts1, pts2):
        """三角测量重建3D点"""
        P1 = self._build_projection_matrix(R1, t1)
        P2 = self._build_projection_matrix(R2, t2)

        points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)

        points_3d = []
        for i in range(points_4d.shape[1]):
            point = points_4d[:, i]
            point /= point[3]
            points_3d.append(point[:3])

        return np.array(points_3d)

    def _build_projection_matrix(self, R, t):
        """构建投影矩阵 P = K[R|t]"""
        P = np.zeros((3, 4))
        P[:, :3] = R
        P[:, 3] = t.flatten()
        return np.dot(self.K, P)


class IncrementalReconstructor:
    """增量式重建器"""

    def __init__(self, K):
        self.K = K
        self.triangulator = Triangulator(K)
        self.rotations = []
        self.motions = []
        self.structure = None
        self.correspondence = []  # 改为列表，存储每帧的对应关系

    def _get_matched_points(self, key_points1, key_points2, matches):
        """获取匹配点坐标"""
        matche_points_1 = np.asarray([key_points1[m.queryIdx].pt for m in matches])
        matche_points_2 = np.asarray([key_points2[m.trainIdx].pt for m in matches])
        return matche_points_1, matche_points_2

    def _get_matched_colors(self, colors1, colors2, matches):
        """获取匹配点颜色"""
        matches_color_1 = np.asarray([colors1[m.queryIdx] for m in matches])
        matches_color_2 = np.asarray([colors2[m.trainIdx] for m in matches])
        return matches_color_1, matches_color_2

    def _estimate_pose(self, pts1, pts2):
        """从本质矩阵恢复R、t"""
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K)
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K, mask=mask)
        return R, t, mask.flatten().astype(bool)

    def initialize(self, key_points1, key_points2, matches):
        """使用前两帧初始化重建"""
        print("  初始化重建...")

        # 获取匹配点
        pts1, pts2 = self._get_matched_points(key_points1, key_points2, matches)
        print(f"    初始匹配点数: {len(pts1)}")

        # 估计本质矩阵和位姿
        R, t, mask = self._estimate_pose(pts1, pts2)
        print(f"    有效匹配点数: {np.sum(mask)}")

        # 只保留有效匹配点
        pts1_valid = pts1[mask]
        pts2_valid = pts2[mask]

        # 三角测量
        R1 = np.eye(3)
        t1 = np.zeros((3, 1))
        structure = self.triangulator.triangulate(R1, t1, R, t, pts1_valid, pts2_valid)
        print(f"    三角测量点数: {len(structure)}")

        # 初始化数据结构
        self.rotations = [R1, R]
        self.motions = [t1, t]
        self.structure = structure

        # 建立特征点与3D点的对应关系（每帧一个数组）
        self.correspondence = []

        # 第一帧的对应关系
        corr1 = np.ones(len(key_points1)) * -1
        # 第二帧的对应关系
        corr2 = np.ones(len(key_points2)) * -1

        # 建立对应关系
        point_idx = 0
        for i, match in enumerate(matches):
            if mask[i]:
                corr1[match.queryIdx] = point_idx
                corr2[match.trainIdx] = point_idx
                point_idx += 1

        self.correspondence.append(corr1)
        self.correspondence.append(corr2)

        return structure

    def add_frame(self, key_points_prev, key_points_curr,
                  descriptors_prev, descriptors_curr, matches):
        """添加新帧"""

        # 获取已有3D点与当前帧的匹配
        object_pts, image_pts = self._get_3d_2d_correspondence(
            key_points_curr, matches
        )

        if len(image_pts) < 8:
            print(f"    匹配点不足: {len(image_pts)} < 8")
            return False

        # PnP求解当前帧位姿
        success, R, t = self._solve_pnp(object_pts, image_pts)
        if not success:
            print("    PnP求解失败")
            return False

        self.rotations.append(R)
        self.motions.append(t)

        # 重建新的3D点
        new_pts_3d = self._reconstruct_new_points(
            key_points_prev, key_points_curr, matches, R, t
        )

        # 融合点云
        self._fuse_points(new_pts_3d, matches, key_points_curr)

        return True

    def _get_3d_2d_correspondence(self, key_points_curr, matches):
        """
        获取3D点与当前帧2D点的对应关系
        """
        object_points = []
        image_points = []

        # 获取前一帧的对应关系
        if len(self.correspondence) == 0:
            return np.array([]), np.array([])

        prev_corr = self.correspondence[-1]  # 前一帧的对应关系

        for match in matches:
            query_idx = match.queryIdx  # 前一帧的特征点索引
            train_idx = match.trainIdx  # 当前帧的特征点索引

            # 检查前一帧的这个特征点是否有对应的3D点
            if query_idx < len(prev_corr):
                struct_idx = int(prev_corr[query_idx])
                if struct_idx >= 0 and struct_idx < len(self.structure):
                    object_points.append(self.structure[struct_idx])
                    image_points.append(key_points_curr[train_idx].pt)

        return np.array(object_points), np.array(image_points)

    def _solve_pnp(self, object_pts, image_pts):
        """求解PnP问题，获取相机位姿"""
        # 使用RANSAC PnP
        _, r_vec, t, inliers = cv2.solvePnPRansac(
            object_pts.astype(np.float32),
            np.array(image_pts).astype(np.float32),
            self.K, None,
            iterationsCount=100,
            reprojectionError=8.0,
            confidence=0.99
        )

        if r_vec is None or len(inliers) < 8:
            return False, None, None

        # 转换为旋转矩阵
        R, _ = cv2.Rodrigues(r_vec)

        return True, R, t

    def _reconstruct_new_points(self, key_points_prev, key_points_curr,
                                matches, R_curr, t_curr):
        """重建新的3D点"""
        # 获取前一帧的位姿
        R_prev = self.rotations[-2]  # 倒数第二帧
        t_prev = self.motions[-2]

        # 获取匹配点
        pts_prev, pts_curr = self._get_matched_points(
            key_points_prev, key_points_curr, matches
        )

        # 三角测量新点
        new_points = self.triangulator.triangulate(
            R_prev, t_prev, R_curr, t_curr, pts_prev, pts_curr
        )

        return new_points

    def _fuse_points(self, new_points, matches, key_points_curr):
        """融合新的点云"""
        # 获取前一帧的对应关系
        prev_corr = self.correspondence[-1]

        # 创建当前帧的对应关系（初始化为-1）
        curr_corr = np.ones(len(key_points_curr)) * -1

        # 融合新点
        point_idx = 0
        for i, match in enumerate(matches):
            query_idx = match.queryIdx  # 前一帧索引
            train_idx = match.trainIdx  # 当前帧索引

            # 检查前一帧是否有对应的3D点
            if query_idx < len(prev_corr):
                struct_idx = int(prev_corr[query_idx])
                if struct_idx >= 0:
                    # 已有3D点，直接关联
                    curr_corr[train_idx] = struct_idx
                else:
                    # 新点，添加到点云
                    if i < len(new_points) and point_idx < len(new_points):
                        self.structure = np.append(self.structure, [new_points[i]], axis=0)
                        curr_corr[train_idx] = len(self.structure) - 1
                        point_idx += 1

        # 添加当前帧的对应关系
        self.correspondence.append(curr_corr)

        return self.structure

    def get_correspondence(self):
        """获取对应关系"""
        return self.correspondence