import cv2
import numpy as np


class FeatureExtractor:
    """特征提取器"""

    def __init__(self, nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10):
        """
        初始化SIFT特征提取器
        Args:
            nfeatures: 保留的最佳特征数量
            nOctaveLayers: 每个octave的层数
            contrastThreshold: 对比度阈值
            edgeThreshold: 边缘阈值
        """
        self.sift = cv2.SIFT_create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold)

    def extract_from_images(self, image_names):
        """
        批量提取图像特征
        Returns:
            key_points_list: 关键点列表
            descriptors_list: 描述子列表
            colors_list: 颜色信息列表
            images_list: 图像列表
        """
        key_points_list = []
        descriptors_list = []
        colors_list = []
        images_list = []

        for image_name in image_names:
            image = cv2.imread(image_name)
            if image is None:
                print(f"Error: Cannot read {image_name}")
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            key_points, descriptor = self.sift.detectAndCompute(gray, None)

            if len(key_points) < 10:
                print(f"Warning: {image_name} has only {len(key_points)} feature points")
                continue

            # 提取特征点的颜色
            colors = self._extract_colors(image, key_points)

            key_points_list.append(key_points)
            descriptors_list.append(descriptor)
            colors_list.append(colors)
            images_list.append(image)

        return np.array(key_points_list, dtype=object), \
            np.array(descriptors_list, dtype=object), \
            np.array(colors_list, dtype=object), \
            images_list

    @staticmethod
    def _extract_colors(image, key_points):
        """提取特征点颜色（统一转成 RGB，便于后续保存与 Web 展示）"""
        colors = np.zeros((len(key_points), 3))
        for i, key_point in enumerate(key_points):
            x, y = int(key_point.pt[0]), int(key_point.pt[1])
            colors[i] = image[y, x][::-1]  # OpenCV 读入是 BGR，反转为 RGB
        return colors