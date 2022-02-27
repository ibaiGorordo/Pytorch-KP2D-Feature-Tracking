import cv2
import numpy as np
import torch

from .kp2d_original.keypoint_net import KeypointNet
from .kp2d_original.keypoint_resnet import KeypointResnet

from .utils import *

class KP2D():

	def __init__(self, model_path, input_size = (640,480), min_score = 0.0, use_gpu=False):

		self.use_gpu = use_gpu
		self.input_size = (640, 480)
		self.min_score = min_score

		# Initialize model
		self.keypoint_net = self.initialize_model(model_path)

	def __call__(self, image):
		return self.detect_kps(image)

	def initialize_model(self, model_path):

		checkpoint = torch.load(model_path)

		try:
			model_args = checkpoint['config']['model']['params']

			keypoint_net = KeypointNet(use_color=model_args['use_color'],
									do_upsample=model_args['do_upsample'],
									do_cross=model_args['do_cross'])
			keypoint_net.load_state_dict(checkpoint['state_dict'])

		except:
			keypoint_net = KeypointResnet()
			keypoint_net.load_state_dict(checkpoint)

		if (self.use_gpu):
			keypoint_net = keypoint_net.cuda()
		keypoint_net.eval()
		keypoint_net.training = False

		return keypoint_net

	def detect_kps(self, image):

		self.img_height, self.img_width = image.shape[:2]

		# Convert the image to match the input of the model
		input_tensor = self.prepare_input(image)

		# Perform inference on the image
		outputs = self.inference(input_tensor)

		# Process output data
		self.score, self.kps, self.desc1 = self.process_output(outputs)
		
		return self.score, self.kps, self.desc1

	def prepare_input(self, img):

		img_height, img_width = img.shape[:2]

		img_input = cv2.resize(img, self.input_size)  

		# Convert image to tensor
		img_input = img_input.astype(np.float32)/255
		img_input = img_input.transpose(2, 0, 1)
		input_tensor = torch.from_numpy(img_input)

		# Normalize image
		input_tensor -= 0.5
		input_tensor *= 0.225
		input_tensor = torch.unsqueeze(input_tensor, 0)

		if self.use_gpu:
			input_tensor = input_tensor.cuda()

		return input_tensor

	def inference(self, input_tensor):

		with torch.inference_mode():
			outputs = self.keypoint_net(input_tensor)

		return outputs

	def process_output(self, outputs):

		score, coord, desc = outputs	

		B, C, Hc, Wc = desc.shape

		# Scores & Descriptors
		coord = coord.view(2, -1).t().cpu().numpy()
		score = np.squeeze(score.view(1, -1).t().cpu().numpy())
		desc = desc.view(C, Hc, Wc).view(C, -1).t().cpu().numpy()

		# Filter based on confidence threshold
		desc = desc[score > self.min_score, :]
		coord = coord[score > self.min_score, :]
		score = score[score > self.min_score]

		# Resize coordinates
		coord[:,0] *= self.img_width/self.input_size[0]
		coord[:,1] *= self.img_height/self.input_size[1]

		kps = [cv2.KeyPoint(x[0], x[1], 1) for x in coord]

		return score, kps, desc
	

if __name__ == '__main__':

	from imread_from_url import imread_from_url

	input_size = (640, 480)
	use_gpu = True
	min_score = 0.5
	top_kps = 500
	model_path = "../models/keypoint_resnet.ckpt"

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

	# kps_img = draw_keypoints_conf(img2, coord2, scores2, min_score)

	matched_image = draw_matches_inplace(img1, kps1, img2, kps2, matches)

	cv2.namedWindow("Matched keypoints", cv2.WINDOW_NORMAL)
	cv2.imshow("Matched keypoints", matched_image)
	cv2.waitKey(0)