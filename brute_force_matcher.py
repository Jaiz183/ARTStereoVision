import numpy as np
import cv2
import matplotlib.pyplot as plt

image1 = cv2.imread("cuboid.jpeg", cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(
    "cuboid_transformed.jpeg", cv2.IMREAD_GRAYSCALE
)  # TODO add this image and test!

# Initialize FAST detector.
fast_feature_detector = cv2.FastFeatureDetector_create()

# Initialize BRIEF description extractor.
brief_description_extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create()

# Find keypoints with FAST and descriptors of keypoints with BRIEF.
kp1 = fast_feature_detector.detect(image1, None)
kp1, des1 = brief_description_extractor.compute(image1, kp1)

kp2 = fast_feature_detector.detect(image2, None)
kp2, des2 = brief_description_extractor.compute(image2, kp2)

# Find matches between descriptors by brute force!
# TODO Experiment with different distance metrics.
brute_force_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = brute_force_matcher.match(des1, des2)

# TODO draw matches onto new image.
