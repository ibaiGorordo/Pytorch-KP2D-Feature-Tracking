import cv2
from imread_from_url import imread_from_url

from kp2d import KP2D
from kp2d.utils import *

input_size = (640, 480)
use_gpu = True
min_score = 0.5
top_kps = 500
model_path = "models/keypoint_resnet.ckpt"

img1 = imread_from_url("https://github.com/liruoteng/OpticalFlowToolkit/blob/master/data/example/KITTI/frame1.png?raw=true")
img2 = imread_from_url("https://github.com/liruoteng/OpticalFlowToolkit/blob/master/data/example/KITTI/frame2.png?raw=true")

# Initialize model
keypoint_detector = KP2D(model_path, input_size, min_score, use_gpu)

# Detect keypoints
scores1, kps1, desc1 = keypoint_detector(img1)
scores2, kps2, desc2 = keypoint_detector(img2)

# Match detected keypoints
matches = match_kps(kps1, desc1, 
					kps2, desc2, top_kps)

matched_image = draw_matches_inplace(img1, kps1, img2, kps2, matches)

cv2.namedWindow("Matched keypoints", cv2.WINDOW_NORMAL)
cv2.imshow("Matched keypoints", matched_image)
cv2.waitKey(0)