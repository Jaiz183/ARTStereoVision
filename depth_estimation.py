import numpy as np
import cv2
import matplotlib.pyplot as plt
from common_functions import *
import math


def compute_disparity(matches: list[cv2.DMatch]) -> list[float]:
    return [match.distance for match in matches]


def compute_depth(
    disparities: list[float], focal_length: float, baseline: float
) -> list[float]:
    return [(focal_length * baseline) / disparity for disparity in disparities]


def compute_point_cloud(keypoints, depth_map: list[float]) -> list[np.array]:
    # Implement this!
    # Get x, y positions of each pixel on disparity map?
    return [
        np.array(keypoint[0], keypoint[1], depth)
        for keypoint in keypoints
        for depth in depth_map
    ]
