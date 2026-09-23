import cv2
import numpy as np


class FeatureMatcher:
    """特征匹配器"""

    def __init__(self, norm_type=cv2.NORM_L2, cross_check=False):
        self.bf = cv2.BFMatcher(norm_type, cross_check)

    def match_sequential(self, descriptors_list, ratio=0.5, visualize=False):
        """
        顺序匹配相邻图像
        Args:
            descriptors_list: 描述子列表
            ratio: Lowe's ratio test阈值
            visualize: 是否可视化匹配结果
        Returns:
            matches_list: 匹配结果列表
        """
        matches_list = []

        for i in range(len(descriptors_list) - 1):
            matches = self._match_pair(descriptors_list[i], descriptors_list[i + 1], ratio)
            matches_list.append(matches)
            print(f"Matched image {i + 1} and {i + 2}: {len(matches)} matches")

        return np.array(matches_list, dtype=object)

    def _match_pair(self, desc1, desc2, ratio):
        """匹配一对图像"""
        knn_matches = self.bf.knnMatch(desc1, desc2, k=2)

        good_matches = []
        for m, n in knn_matches:
            if m.distance < ratio * n.distance:
                good_matches.append(m)

        return np.array(good_matches)