# https://stackoverflow.com/questions/33670222/how-to-use-surf-and-sift-detector-in-opencv-for-python
# https://answers.opencv.org/question/221922/how-to-exclude-outliers-from-detected-orb-features/
# https://github.com/kemfic/VOpy/blob/vopy_old/frame.py
# https://learnopencv.com/rotation-matrix-to-euler-angles/

from dataset import DuckietownDataset
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math


def p2e(v):
    v = v / v[[2]]
    return v[:-1]


def e2p(v):
    return np.r_[v, 1]


def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


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


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


K = np.array([[373.2779426913342, 0.0, 318.29785021099894],
              [0.0, 367.9439633567062, 263.9058079734077],
              [0.0, 0.0, 1.0]])

D = np.array([-0.3017710043972695, 0.07403470502924431, 0.0013028188828223006, 0.00022752165172560925, 0.0])

dataloader = DuckietownDataset("alex_2small_loops_ground_truth.txt", "alex_2small_loops_images", K=K, D=D)

for data, rel_pos, scale, K in dataloader:
    orb = cv2.ORB_create()

    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(cv2.cvtColor(data[0], cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = orb.detectAndCompute(cv2.cvtColor(data[1], cv2.COLOR_BGR2GRAY), None)


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

    # print(f" K {K} \n D {D}")
    # src_pts_norm = cv2.undistortPoints(src_pts, cameraMatrix=K, distCoeffs=D)
    # dst_pts_norm = cv2.undistortPoints(src_pts, cameraMatrix=K, distCoeffs=D)

    E, mask = cv2.findEssentialMat(src_pts, dst_pts, cameraMatrix=K, method=cv2.RANSAC, prob=0.9, threshold=0.1)
    # E, mask = cv2.findEssentialMat(src_pts_norm, dst_pts_norm, focal=1.0, pp=(0., 0.),
    #                               method=cv2.RANSAC, prob=0.9, threshold=0.1)
    # print("E", E)

    show_matched_keypoints(data, src_pts, dst_pts, mask)
    print("moving points", src_pts[0], dst_pts[0])

    points, R, t, mask = cv2.recoverPose(E, src_pts, dst_pts)

    # if (scale > 0.1 and abs(t[2][0]) > abs(t[0][0]) and abs(t[2][0]) > abs(t[1][0])):
    #    t = self.R.dot(t)
    #    self.R = R.dot(self.R)


    H = np.hstack((R, t))
    H = np.vstack((H, [0, 0, 0, 1]))

    # print("euler angles", rotationMatrixToEulerAngles(R))
    # R1, R2, t1 = cv2.decomposeEssentialMat(E)
    # print(f" R1 {R1} \n R2 {R2} \n t1 {t1} \n rel_pos {rel_pos}")
    # t_e = p2e(t)

    print(f" H {H} \n rel_pos {rel_pos} \n norm {np.linalg.norm(t)}")

    print(np.linalg.norm(rel_pos[:, 3]))

    rel_pos[:, 3] = rel_pos[:, 3] / np.linalg.norm(rel_pos[:, 3])

    print(rel_pos)

    print("______________________________________________________")
