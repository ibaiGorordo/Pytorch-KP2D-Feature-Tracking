import cv2
import pafy
import time
from imread_from_url import imread_from_url

from kp2d import KP2D, FeatureTracker

# Initialize video
# cap = cv2.VideoCapture("test.mp4")

videoUrl = 'https://youtu.be/zP-gTCp5Kac'
videoPafy = pafy.new(videoUrl)
print(videoPafy.streams)
cap = cv2.VideoCapture(videoPafy.streams[-1].url)
start_time = 0 # skip first {start_time} seconds
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time*30)

# Initialize model
model_path = "models/keypoint_resnet.ckpt"
input_size = (640, 480)
use_gpu = True
min_score = 0.7
keypoint_detector = KP2D(model_path, input_size, min_score, use_gpu)

# Initialize feature tracker
buffer_size = 		5   # Number of samples that are kept in memory for the trail plot
max_dissappeared = 	1 	 # Number of frames that a keypoiny can be missing until deleted
min_points = 		50  # Minimum number of points being tracked
keep_k_points = 	200  # Filter the best matches
comp_thres = 		0.7 # For the matching comparison

feature_tracker = FeatureTracker(buffer_size, max_dissappeared, 
					  			 min_points, keep_k_points, comp_thres)

cv2.namedWindow("Tracked keypoints", cv2.WINDOW_NORMAL)	
while cap.isOpened():

	try:
		# Read frame from the video
		ret, new_frame = cap.read()
		if not ret:	
			break
	except:
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
