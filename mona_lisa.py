import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import os


def show_matched_keypoints(data, src_pts, dst_pts, mask):
    img2 = data[0].copy()
    img3 = data[1].copy()

    idx = np.where(mask == [1])

    for i in range(len(src_pts[idx[0]])):
        color1 = (list(np.random.choice(range(256), size=3)))
        color = [int(color1[0]), int(color1[1]), int(color1[2])]
        img2 = cv2.circle(img2, (int(src_pts[idx[0]][i][0][0]), int(src_pts[idx[0]][i][0][1])), color=color,
                          radius=10, thickness=2)
        img3 = cv2.circle(img3, (int(dst_pts[idx[0]][i][0][0]), int(dst_pts[idx[0]][i][0][1])), color=color,
                          radius=10, thickness=2)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
    plt.title("Image at time t")
    plt.show()
    plt.imshow(cv2.cvtColor(img3, cv2.COLOR_RGB2BGR))
    plt.title("Image at time t+1")
    plt.show()


img1 = cv2.imread("/Users/julia/Documents/UNI/Master/Montréal/3D/project/P6.png")
img2 = cv2.imread("/Users/julia/Documents/UNI/Master/Montréal/3D/project/P7.png")
data = [img1, img2]
# print(img1, img2)

orb = cv2.ORB_create()
# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(data[0], None)
kp2, des2 = orb.detectAndCompute(data[1], None)

print(kp1, kp2)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1, des2)

# Sort them in the order of their distance.
matches = sorted(matches, key=lambda x: x.distance)
# Draw first 10 matches.
img3 = cv2.drawMatches(cv2.cvtColor(data[0], cv2.COLOR_BGR2GRAY), kp1,
                       cv2.cvtColor(data[1], cv2.COLOR_BGR2GRAY), kp2, matches[:10], None, flags=2)
plt.imshow(img3), plt.show()

# extract the matched keypoints
src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

K = np.asarray([[150, 0, 150], [0, 150, 150], [0, 0, 1]])

E, mask = cv2.findEssentialMat(src_pts, dst_pts, cameraMatrix=K, method=cv2.RANSAC,
                               prob=0.999, threshold=1.0)

show_matched_keypoints(data, src_pts, dst_pts, mask)

points, R, t, mask = cv2.recoverPose(E, src_pts, dst_pts)

print(f" E {E} \n")

H = np.hstack((R, t))
# H = np.vstack((H, [0, 0, 0, 1]))

print(f" H {H} \n")

print(f" t {t} \n")
