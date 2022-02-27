from collections import OrderedDict
import random
import cv2
import numpy as np

class FeatureTracker():

	def __init__(self, buffer_size = 10, max_dissappeared = 10, 
					   min_points=200, keep_top_points=500, comp_thres = 0.75):

		self.buffer_size = buffer_size
		self.max_dissappeared = max_dissappeared
		self.next_object_id = 0
		self.min_points = min_points
		self.keep_top_points = keep_top_points
		self.comp_thres = comp_thres

		self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

		# Initialize data
		self.tracked_kps = OrderedDict()
		self.tracked_descs = OrderedDict()
		self.tracks = OrderedDict()
		self.disappeared = OrderedDict()
		self.colors = OrderedDict()
		
	def __call__(self, new_kps, new_descs):

		return self.update(new_kps, new_descs)

	def update(self, new_kps, new_descs):

		# Update all keypoints as missing
		if(len(new_kps) == 0):
			self.increment_all_missing()
			return False, []

		# Add the keypoint directly if no keypoints are being tracked
		if(len(self.tracked_kps)==0):
			for kp, desc in zip(new_kps, new_descs):
				self.register(kp, desc)
			return False, []

		# Match the new keypoints with the points being tracked
		tracked_descs_array = np.array(list(self.tracked_descs.values()))
		matches = self.match_kps(tracked_descs_array, new_descs)

		# Add the new keypoints that were matched
		self.update_matches(new_kps, new_descs, matches)

		# Check if keypoints have dissapeared
		self.increment_missing_kps(matches)

		# Register the rest of the points if the number of tracked  
		# keypoints is low
		num_long_trails = np.sum(np.array([len(track) for track in self.tracks.values()])>(self.max_dissappeared+self.buffer_size)//2)

		if(num_long_trails < self.min_points):
			self.register_non_visited(new_kps, new_descs, matches)

		# Update tracks
		self.update_tracks()

		return True, matches

	def register(self, kp, desc):

		# When registering an object we use the next available object
		# ID to store the keypoint
		self.tracked_kps[self.next_object_id] = kp
		self.tracked_descs[self.next_object_id] = desc
		self.disappeared[self.next_object_id] = 0
		self.colors[self.next_object_id] = (random.randint(0,255),
											random.randint(0,255),
											random.randint(0,255))
		self.next_object_id += 1

	def deregister(self, object_id):

		# To deregister an object ID we delete the object ID from
		# both of our respective dictionaries
		del self.tracked_kps[object_id]
		del self.tracked_descs[object_id]
		del self.disappeared[object_id]
		del self.colors[object_id]

		if(object_id in self.tracks.keys()):
			del self.tracks[object_id]

	def update_matches(self, new_kps, new_descs, matches):

		tracked_kps_id_list = list(self.tracked_descs.keys())

		for m in matches:

			tracked_kps_id = tracked_kps_id_list[m.queryIdx]
			new_kps_id = m.trainIdx

			# Update tracked keypoint data
			self.tracked_kps[tracked_kps_id] = new_kps[new_kps_id]
			self.tracked_descs[tracked_kps_id] = new_descs[new_kps_id]
			self.disappeared[tracked_kps_id] = 0

	def update_tracks(self):

		# Skip if there is no keypoint being tracked
		if not self.tracked_kps:
			return

		for kp_id, kp in self.tracked_kps.items():

			if(kp_id in self.tracks.keys()):
				if len(self.tracks[kp_id]) > self.buffer_size:
					self.tracks[kp_id].pop(0)
			else:
				self.tracks[kp_id] = []

			self.tracks[kp_id].append((int(kp.pt[0]),int(kp.pt[1])))

	def register_non_visited(self, new_kps, new_descs, matches):

		new_kps_id_set = set(range(len(new_kps)))
		visited_kps_id_set = set([m.trainIdx for m in matches])

		non_visited_id_set = new_kps_id_set - visited_kps_id_set

		for object_id in non_visited_id_set:
			self.register(new_kps[object_id], new_descs[object_id])

	def increment_missing_kp(self, object_id):

		self.disappeared[object_id] += 1

		# If we have reached a maximum number of consecutive
		# frames where a given object has been marked as
		# missing, deregister it
		if self.disappeared[object_id] >= self.max_dissappeared:
			self.deregister(object_id)

	def increment_missing_kps(self, matches):

		tracked_kps_id_list = list(self.tracked_descs.keys())
		tracked_kps_id_set = set(self.tracked_descs.keys())
		visited_kps_id_set = set([tracked_kps_id_list[m.queryIdx] for m in matches])

		non_visited_id_set = tracked_kps_id_set - visited_kps_id_set

		for object_id in non_visited_id_set:
			self.increment_missing_kp(object_id)

	def increment_all_missing(self):

		# Loop over any existing tracked objects and mark them
		# as disappeared
		for object_id in self.disappeared.keys():
			self.increment_missing_kp(object_id)

	def match_kps(self, des1, des2):
		
		matches = self.matcher.knnMatch(des1, des2, k=2)
		matches = [m for m, n in matches if m.distance < self.comp_thres*n.distance]

		# keep_top_points = min(self.keep_top_points, len(matches))
		# matches = sorted(matches, key = lambda x:x.distance)
		# matches = matches[:keep_top_points]

		return matches

	def draw_tracks(self, img):

		size = int(min(img.shape[0], img.shape[1])/300)
		vis = img.copy()
		for object_id, track in self.tracks.items():

			# At least two points are necessary
			if len(track) <= (self.max_dissappeared+self.buffer_size)//2:
				continue

			for i in range(1,len(track)):

				cv2.line(vis, track[i-1], track[i], self.colors[object_id], size)
			circle_color = tuple(channel//2 for channel in self.colors[object_id])
			cv2.circle(vis, track[i], size+2, circle_color, -1)

		return vis

