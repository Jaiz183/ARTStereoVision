import cv2
import numpy as np


def find_essential_matrix(corners1, corners2, camera_matrix):
    return cv2.findEssentialMat(corners1, corners2, camera_matrix)


def find_rectification_matrix(corners1, corners2, camera_matrix):
    # Find essential matrix
    R, T = find_essential_matrix(corners1, corners2, camera_matrix)
