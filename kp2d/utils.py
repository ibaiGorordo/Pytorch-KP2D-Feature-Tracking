import random
import cv2
import numpy as np
from matplotlib import cm

colormap = (cm.jet(range(256))[:,:3]*255).astype(int)
colors = np.random.randint(0, 255, (100, 3))
color = (0,0,255)

def score_to_color(score, min_score):
	norm_score = int((score-min_score)/(1-min_score)*255)
	color = colormap[norm_score, :]

	return (int(color[2]),int(color[1]),int(color[0]))

def draw_keypoints(img, kps, color):

    size = int(min(img.shape[0], img.shape[1])/250)

    """Draw keypoints on an image"""
    vis = img.copy()
    for kp in kps:
        x, y = int(kp.pt[0]), int(kp.pt[1])

        cv2.circle(vis, (x,y), 3, color, -1)
    return vis

def draw_keypoints_conf(img, kps, scores, min_score):

    size = int(min(img.shape[0], img.shape[1])/250)

    """Draw keypoints on an image"""
    vis = img.copy()
    for kp, score in zip(kps, scores):
        x, y = int(kp.pt[0]), int(kp.pt[1])

        cv2.circle(vis, (x,y), size, score_to_color(score, min_score), -1)
    return vis

def draw_matches_side_to_side(img1, kps1, img2, kps2, matches):

    return cv2.drawMatchesKnn(img1, kps1, img2, kps2, matches, None, flags=2)

def draw_matches_inplace(img1, kps1, img2, kps2, matches):

    size = int(min(img2.shape[0], img2.shape[1])/200)

    vis = img2.copy()
    for match in matches:

        idx1 = match[0].queryIdx 
        idx2 = match[0].trainIdx

        x1, y1 = int(kps1[idx1].pt[0]), int(kps1[idx1].pt[1])
        x2, y2 = int(kps2[idx2].pt[0]), int(kps2[idx2].pt[1])

        color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))


        cv2.line(vis, (x1,y1), (x2,y2), color, size+1)
        circle_color = tuple(channel//2 for channel in color)
        cv2.circle(vis, (x2,y2), size+2, circle_color, -1)

    return vis

def match_kps(kps1, des1, kps2, des2, keep_k_points=1000, comp_thres = 0.75):
    
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    matches = bf.knnMatch(des1, des2, k=2)
    matches = [[m] for m, n in matches if m.distance < comp_thres*n.distance]

    keep_k_points = min(keep_k_points, len(matches))
    matches = matches[:keep_k_points]

    return matches
