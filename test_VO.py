import cv2
import matplotlib.pyplot as plt
import numpy as np


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


obj1 = np.random.randint(2, size=(8, 8))
obj2 = np.random.randint(2, size=(4, 4))
"""img_t = np.zeros((16, 16))
img_t[0, 10] = 1
img_t[10, 0] = 1
img_t[10, 10] = 1
img_t[10, 15] = 1
img_t[5, 10] = 1
img_t[15, 15] = 1"""

img_t = np.pad(obj1, 16, pad_with, padder=0)
img_tb = img_t.copy()
img_tb[10:14, 10:14] = obj2

img_ta = np.roll(img_t, -5, axis=1)
img_ta[10:14, 1:5] = obj2
img_tc = np.roll(img_t, 5, axis=1)
img_tc[10:14, 20:24] = obj2

img_ta = np.rot90(img_ta, 3)
img_tb = np.rot90(img_tb, 3)
img_tc = np.rot90(img_tc, 3)

fig, axs = plt.subplots(1, 3)
fig.suptitle('Images across 3 frames')
axs[0].imshow(img_ta, cmap='gray')
axs[1].imshow(img_tb, cmap='gray')
axs[2].imshow(img_tc, cmap='gray')
fig.show()

pts_a = np.transpose(img_ta.nonzero())
pts_b = np.transpose(img_tb.nonzero())
pts_c = np.transpose(img_tc.nonzero())
print(pts_a, pts_b, pts_c)

# K = np.eye(3)

print(pts_a - pts_b)

E, mask = cv2.findEssentialMat(pts_a, pts_b, method=cv2.RANSAC,
                               prob=0.999, threshold=1.0)

print(f" E {E} \n")

points, R, t, mask = cv2.recoverPose(E, pts_a, pts_b)

H = np.hstack((R, t))
# H = np.vstack((H, [0, 0, 0, 1]))

print(f" H {H} \n")

print(f" t {t} \n")

rot_angle, _ = cv2.Rodrigues(R)

print(f" Rot_angle {rot_angle} \n")