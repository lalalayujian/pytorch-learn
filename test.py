import pandas as pd
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import cv2  # 读取图片是BGR, shape=(W, H, C)
import os
from PIL import Image


landmarks_frame = pd.read_csv(r'D:\DXH\DL_practice\face_pose\data\faces\face_landmarks.csv')
print(landmarks_frame)

n = 65
img_name = landmarks_frame.iloc[n, 0]
landmarks = landmarks_frame.iloc[n, 1:].values
landmarks = landmarks.astype(float).reshape(-1, 2)

def show_landmarks(image, landmarks):
    """可视化人脸和标注点"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1])


plt.figure()
show_landmarks(cv2.imread(os.path.join(r'D:\DXH\DL_practice\face_pose\data\faces', img_name)), landmarks)
plt.show()
