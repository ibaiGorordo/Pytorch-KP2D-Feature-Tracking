import cv2
from imread_from_url import imread_from_url

from kp2d import KP2D
from kp2d.utils import *

input_size = (640, 480)
use_gpu = True
min_score = 0.7
model_path = "models/keypoint_resnet.ckpt"

img = imread_from_url("https://github.com/liruoteng/OpticalFlowToolkit/blob/master/data/example/KITTI/frame1.png?raw=true")

# Initialize model
keypoint_detector = KP2D(model_path, input_size, min_score, use_gpu)

# Detect keypoints
scores, kps, desc = keypoint_detector(img)

# Draw keypoints with confidence
kps_img = draw_keypoints_conf(img, kps, scores, min_score)

cv2.namedWindow("Detected keypoints", cv2.WINDOW_NORMAL)
cv2.imshow("Detected keypoints", kps_img)
cv2.waitKey(0)