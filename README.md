# Pytorch-KP2D-Feature-Tracking
 Python scripts for performing 2D feature detection and tracking using the KP2D model in Pytorch

![KP2D 2D Feature matching](https://github.com/ibaiGorordo/Pytorch-KP2D-Feature-Tracking/blob/main/doc/img/output.png)
*Original images:https://github.com/liruoteng/OpticalFlowToolkit/blob/master/data/example/KITTI/*

# Requirements

 * Check the **requirements.txt** file. 
 * For Pytorch, check the official website to install the version matching your machine: https://pytorch.org/
 * Additionally, **pafy** and **youtube-dl** are required for youtube video inference.
 
# Installation (Except Pytorch)
```
pip install -r requirements.txt
pip install pafy youtube_dl=>2021.12.17
```

# Pretrained model
Download the original models [KeypointNet](https://tri-ml-public.s3.amazonaws.com/github/kp2d/models/pretrained_models.tar.gz) and [KeypointResnet](https://tri-ml-public.s3.amazonaws.com/github/kp3d/pretrained_models.tar.gz) and extract them into the **[models](https://github.com/ibaiGorordo/Pytorch-KP2D-Feature-Tracking/tree/main/models)** folder. 

# Original Repository
The [original repository](https://github.com/TRI-ML/KP2D) contains additional code to train the models in Pytorch. This repository uses part of that code to make it easier to use the model in videos, images and webcamera.
 
# Examples

 * **Image Feature Matching**:
 
 ```
 python image_feature_matching.py
 ```
 
 * **Image Feature Detection Confidence**:
 
 ```
 python image_feature_detection_conf.py
 ```
 
  * **Video Feature tracking**:
 
 ```
 python video_feature_tracking.py
 ```
 
 * **Webcam Feature Tracking**:
 
 ```
 python webcam_feature_tracking.py
 ```
 
# Inference video Example: https://youtu.be/IeeRWMhpyc0
 ![KP2D 2D Feature tracking drone](https://github.com/ibaiGorordo/Pytorch-KP2D-Feature-Tracking/blob/main/doc/img/kp2d_feature_tracking.gif)

*Original video: https://youtu.be/zP-gTCp5Kac*

# References:
* KP2D original repository: https://github.com/TRI-ML/KP2D
* Original paper: https://openreview.net/pdf?id=Skx82ySYPH
 
