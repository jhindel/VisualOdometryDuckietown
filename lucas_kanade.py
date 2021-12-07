import numpy as np
import cv2 as cv
import argparse
import os
from dataset import DuckietownDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# source: https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html

# parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation. \
#                                              The example file can be downloaded from: \
#                                              https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
# parser.add_argument('image', type=str, help='path to image file')
# args = parser.parse_args()
# params for ShiTomasi corner detection
K = np.array([[373.2779426913342, 0.0, 318.29785021099894],
                  [0.0, 367.9439633567062, 263.9058079734077],
                  [0.0, 0.0, 1.0]])

D = np.array([-0.3017710043972695, 0.07403470502924431, 0.0013028188828223006, 0.00022752165172560925, 0.0])

dataset = DuckietownDataset("alex_2small_loops_ground_truth.txt", "alex_2small_loops_images", K=K, D=D)

loader = DataLoader(dataset, batch_size=1, shuffle=False)

iterator = iter(loader)


feature_params = dict(maxCorners=10,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)
# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0, 255, (100, 3))


while(1):
    frames, pos, newk = next(iterator)
    first_frame = frames.numpy()[0, 0]
    mask = np.zeros_like(first_frame)
    first_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

    p0 = cv.goodFeaturesToTrack(first_gray, mask=None, **feature_params)
    second_frame = frames.numpy()[0,1]
    second_gray = cv.cvtColor(second_frame, cv.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(first_gray, second_gray, p0, None, **lk_params)
    # Select good points
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv.circle(second_frame, (int(a), int(b)), 5, color[i].tolist(), -1)
    img = cv.add(second_frame, mask)
    # print(img.shape, img)
    cv.imshow('frame', img)
    print("new image")
    k = cv.waitKey(0) & 0xFF
    if k == ord('q'):
        break
    elif k == ord('n'):
        continue
