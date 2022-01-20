import cv2
import matplotlib.pyplot as plt
import numpy as np


def pad_with(vector, pad_width, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value

# build to random objects
obj1 = np.random.randint(2, size=(8, 8))
obj2 = np.random.randint(2, size=(4, 4))

# pad one object and add other
img_t = np.pad(obj1, 16, pad_with, padder=0)
img_tb = img_t.copy()
img_tb[10:14, 10:14] = obj2

# create 2 more frames by moving objects at different rate
img_ta = np.roll(img_t, -5, axis=1)
img_ta[10:14, 1:5] = obj2
img_tc = np.roll(img_t, 5, axis=1)
img_tc[10:14, 20:24] = obj2

# rotate image to produce different movement (hoirzontal/vertical)
img_ta = np.rot90(img_ta, 3)
img_tb = np.rot90(img_tb, 3)
img_tc = np.rot90(img_tc, 3)

# plot figures
fig, axs = plt.subplots(1, 3)
fig.suptitle('Images across 3 frames')
axs[0].imshow(img_ta, cmap='gray')
axs[1].imshow(img_tb, cmap='gray')
axs[2].imshow(img_tc, cmap='gray')
fig.show()

# retrieve non-zero keypoints
pts_a = np.transpose(img_ta.nonzero())
pts_b = np.transpose(img_tb.nonzero())
pts_c = np.transpose(img_tc.nonzero())
print(pts_a, pts_b, pts_c)

# find essential matrix
E, mask = cv2.findEssentialMat(pts_a, pts_b, method=cv2.RANSAC,
                               prob=0.999, threshold=1.0)

print(f" E {E} \n")
# retrieve rotation and translation
points, R, t, mask = cv2.recoverPose(E, pts_a, pts_b)

# build transformation matrix
H = np.hstack((R, t))

print(f" H {H} \n")

print(f" t {t} \n")

# find rotation angles with rodrigues (but no rotation simulated in this experiment)
rot_angle, _ = cv2.Rodrigues(R)

print(f" Rot_angle {rot_angle} \n")