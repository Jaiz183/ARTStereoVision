import cv2


def extract_features(image):
    # Initialize FAST detector.
    fast_feature_detector = cv2.FastFeatureDetector_create()

    # Find keypoints with FAST and descriptors of keypoints with BRIEF.
    kp = fast_feature_detector.detect(image, None)
    return kp
