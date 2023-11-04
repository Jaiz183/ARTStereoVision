import numpy as np
import cv2
import matplotlib.pyplot as plt
from common_functions import *
from brute_force_matcher import *
import math

image1 = cv2.imread("cuboid.jpeg", cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(
    "cuboid_transformed.png", cv2.IMREAD_GRAYSCALE
)  # TODO add this image and test!

# Initialize FAST detector.
fast_feature_detector = cv2.FastFeatureDetector_create()

# Requires opencv-contrib!
# Initialize BRIEF description extractor.
# brief_description_extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create()

# Find keypoints with FAST and descriptors of keypoints with BRIEF.
# kp1 = fast_feature_detector.detect(image1, None)
# kp1, des1 = brief_description_extractor.compute(image1, kp1)

# kp2 = fast_feature_detector.detect(image2, None)
# kp2, des2 = brief_description_extractor.compute(image2, kp2)

# Use ORB instead.
# Initiate ORB detector.
orb = cv2.ORB_create()
# Detect features and get descriptors.
kp1 = orb.detect(image1, None)
kp1, des1 = orb.compute(image1, kp1)

kp2 = orb.detect(image2, None)
kp2, des2 = orb.compute(image2, kp2)

# Find matches between descriptors by brute force!
# TODO Experiment with different distance metrics.
brute_force_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = brute_force_matcher.match(des1, des2)

# Ratio test, David Lowe.
#   - Compare distances of two matches and reject if too close.
filtered_matches = []
# Higher the ratio, more matches we allow because some other point has to be extremely close!
ratio = 2
for m in matches:
    closest_distance = math.inf
    for n in matches:
        # Find smallest difference between n's distance scaled down and m's distance.
        closest_distance = min(closest_distance, ratio * n.distance - m.distance)
        print(closest_distance)

    # Check if closest n's distance is similar after scaling down. If true, reject m.
    if closest_distance > 0:
        filtered_matches.append([m])


image3 = cv2.drawMatchesKnn(
    image1,
    kp1,
    image2,
    kp2,
    filtered_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)

show_image("Matched features", image3)
