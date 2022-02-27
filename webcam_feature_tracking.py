import cv2
import pafy
import time
from imread_from_url import imread_from_url

from kp2d import KP2D, FeatureTracker

# Initialize model
model_path = "models/keypoint_resnet.ckpt"
input_size = (640, 480)
use_gpu = True
min_score = 0.7
keypoint_detector = KP2D(model_path, input_size, min_score, use_gpu)

# Initialize feature tracker
buffer_size = 		5   # Number of samples that are kept in memory for the trail plot
max_dissappeared = 	1 	 # Number of frames that a keypoiny can be missing until deleted
min_points = 		100  # Minimum number of points being tracked
keep_k_points = 	200  # Filter the best matches
comp_thres = 		0.7 # For the matching comparison

feature_tracker = FeatureTracker(buffer_size, max_dissappeared, 
					  			 min_points, keep_k_points, comp_thres)

# Initialize webcam
cap = cv2.VideoCapture(0)

cv2.namedWindow("Tracked keypoints", cv2.WINDOW_NORMAL)	
while cap.isOpened():

	# Read frame from the video
	ret, new_frame = cap.read()
	if not ret:	
		continue
	
	# Detect the keypoints in the current frame
	new_scores, new_kps, new_descs = keypoint_detector(new_frame)

	# Update feature tracker
	ret, matches = feature_tracker(new_kps, new_descs)

	if not ret:
		continue

	tracks_img = feature_tracker.draw_tracks(new_frame)

	cv2.imshow("Tracked keypoints", tracks_img)

	# Press key q to stop
	if cv2.waitKey(1) == ord('q'):
		break
