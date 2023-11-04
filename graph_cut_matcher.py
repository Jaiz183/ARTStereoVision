import cv2
import numpy as np
from common_functions import *


# Graph cut was removed from opencv :(...or was it?
def compute_disparity_map(left_image, right_image) -> tuple:
    # Get dimensions for disparity map with left image.
    height, width, channels = left_image.shape

    # Initialise empty disparity maps (assuming RGB channels and signed integer outputs).
    left_disparities = np.zeros((height, width, channels), dtype="int16")
    right_disparities = np.zeros((height, width, channels), dtype="int16")

    state = cv2.CreateStereoGCState(16, 2)
    cv2.FindStereoCorrespondenceGC(
        left_image, right_image, left_disparities, right_disparities, state, 0
    )

    return left_disparities, right_disparities


if __name__ == "__main__":
    left_image, right_image = cv2.imread("cuboid.jpeg"), cv2.imread(
        "cuboid_transformed.png"
    )

    left_disparities, right_disparities = compute_disparity_map(left_image, right_image)
    # disparity_visual = np.zeros((left_image.height, left_image.width, 3), dtype="uint8")

    # Cast signed 16 bit integers in disparity map to unsigned 8 bit integers.
    disparity_map = left_disparities.astype(dtype="uint8")
    show_image("Disparity map", disparity_map)
