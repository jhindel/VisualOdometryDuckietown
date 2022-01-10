# https://stackoverflow.com/questions/33670222/how-to-use-surf-and-sift-detector-in-opencv-for-python
# https://answers.opencv.org/question/221922/how-to-exclude-outliers-from-detected-orb-features/
# https://github.com/kemfic/VOpy/blob/vopy_old/frame.py
# https://learnopencv.com/rotation-matrix-to-euler-angles/

from VisualOdometryDuckietown.dataset import DuckietownDataset
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import utils
from itertools import product as cartProduct
import sys
import os


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotationMatrixToEulerAngles(R):
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

    return x, y, z


def isclose(x, y, rtol=1.e-5, atol=1.e-8):
    return abs(x - y) <= atol + rtol * abs(y)


def euler_angles_from_rotation_matrix(R):
    '''
    From a paper by Gregory G. Slabaugh (undated),
    "Computing Euler angles from a rotation matrix
    '''
    phi = 0.0
    if isclose(R[2, 0], -1.0):
        theta = math.pi / 2.0
        psi = math.atan2(R[0, 1], R[0, 2])
    elif isclose(R[2, 0], 1.0):
        theta = -math.pi / 2.0
        psi = math.atan2(-R[0, 1], -R[0, 2])
    else:
        theta = -math.asin(R[2, 0])
        cos_theta = math.cos(theta)
        psi = math.atan2(R[2, 1] / cos_theta, R[2, 2] / cos_theta)
        phi = math.atan2(R[1, 0] / cos_theta, R[0, 0] / cos_theta)
    return psi, theta, phi


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


def eval_absolute_poses(relative_poses_pred, absolute_poses):
    absolute_poses = np.asarray(absolute_poses)
    relative_poses_pred = np.asarray(relative_poses_pred)
    absolute_poses_pred = utils.relative2absolute(relative_poses_pred, absolute_poses[0])

    ma_x = np.absolute(absolute_poses[1:, 0] - absolute_poses_pred[1:, 0]).mean()
    ma_y = np.absolute(absolute_poses[1:, 1] - absolute_poses_pred[1:, 1]).mean()
    ma_angle = np.absolute((absolute_poses[1:, 2] % np.pi) - (absolute_poses_pred[1:, 2] % np.pi)).mean()

    print("MAE", ma_x, ma_y, ma_angle, "MAE")

    fig, (ax1) = plt.subplots(1, 1)
    ax1.plot(absolute_poses_pred[:, 0], absolute_poses_pred[:, 1], label='predicted trajectory')
    ax1.plot(absolute_poses[:, 0], absolute_poses[:, 1], label='ground truth trajectory', c='r')
    ax1.axis('scaled')
    fig.legend()
    fig.show()
    # fig.savefig()

    return [ma_x, ma_y, ma_angle], fig


def calc_trajectory(dataset, trajectory_length, plot=False, debug_mode=False):
    error = []
    relative_poses_pred = []
    relative_poses = []
    absolute_poses = []

    for idx, data, new_pose, old_pose, rel_pose, cleaned_K in dataset:

        if debug_mode:
            print(new_pose, old_pose, rel_pose)

        if len(absolute_poses) == 0:
            absolute_poses.append(old_pose)

        orb = cv2.ORB_create()

        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(data[0], None)
        kp2, des2 = orb.detectAndCompute(data[1], None)

        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bf.match(des1, des2)

        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)

        if plot:
            # Draw first 10 matches.
            img3 = cv2.drawMatches(cv2.cvtColor(data[0], cv2.COLOR_BGR2GRAY), kp1,
                                   cv2.cvtColor(data[1], cv2.COLOR_BGR2GRAY), kp2, matches[:10], None, flags=2)
            plt.imshow(img3), plt.show()

        # extract the matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        E, mask = cv2.findEssentialMat(src_pts, dst_pts, cameraMatrix=cleaned_K, method=cv2.RANSAC,
                                       prob=0.99, threshold=0.9)
        # 0.99, 0.9

        if E is None or E.shape != (3, 3):
            print(idx + 30, len(src_pts), len(dst_pts))
            # plt.imshow(cv2.cvtColor(data[0], cv2.COLOR_RGB2BGR))
            # plt.show()
            # plt.imshow(cv2.cvtColor(data[1], cv2.COLOR_RGB2BGR))
            # plt.show()
            relative_poses_pred.append(rel_pose)
            absolute_poses.append([new_pose[0], new_pose[1], new_pose[2]])
            relative_poses.append(rel_pose)
            continue

        if plot:
            show_matched_keypoints(data, src_pts, dst_pts, mask)

        points, R, t, mask = cv2.recoverPose(E, src_pts, dst_pts)

        H = np.hstack((R, t))
        H = np.vstack((H, [0, 0, 0, 1]))

        if debug_mode:
            print(f" H {H}")

        # adapt t to have a result in world_frame
        # scale = np.linalg.norm(ground_truth[:2] - old_pose[:2])

        if (t[2][0] < 0) and (rel_pose[0] > 0):
            t[2][0] = - t[2][0]
            t[1][0] = - t[1][0]

        change_x = rel_pose[0]
        change_y = rel_pose[1]
        # scale = np.linalg.norm(rel_pose[:2])  # [:2])
        scale = np.sqrt(np.sum((rel_pose[0] - rel_pose[1]) ** 2))

        change_x_pred = t[2][0] * scale
        change_y_pred = t[1][0] * scale * 0.1
        if debug_mode:
            print(f" Scaled relative pose pred {change_x_pred}, {change_y_pred} \n "
                  f"relative pose ground truth {change_x}, {change_y}")
        error_x = (change_x - change_x_pred) ** 2  # z for image
        error_y = (change_y - change_y_pred) ** 2  # x for image
        error_z = (t[0][0] * scale) ** 2

        # extract r from rotation matrix and compare
        if not isRotationMatrix(R):
            print(idx + 30, R)
        x, y, z = rotationMatrixToEulerAngles(R)
        # x, y, z = euler_angles_from_rotation_matrix(R)
        # angles, _ = cv2.Rodrigues(R)
        # theta = np.sqrt(angles[0][0]**2 + angles[1][0]**2 + angles[2][0]**2)
        # y = angles/theta
        change_angle_pred = y
        change_angle = rel_pose[2]
        error_angle = (change_angle - change_angle_pred) ** 2
        if debug_mode:
            print(f" Euler angles {x, y, z} \n relative angle ground truth {change_angle}")
        error.append([error_x, error_y, error_angle, error_z])

        relative_pose_pred = [change_x_pred, change_y_pred, change_angle_pred]
        # relative_pose_pred = [change_x, change_y, change_angle]
        if debug_mode:
            print(relative_pose_pred)
            print("______________________________________________________")
        relative_poses_pred.append(relative_pose_pred)
        absolute_poses.append([new_pose[0], new_pose[1], new_pose[2]])
        relative_poses.append([change_x, change_y, change_angle])

        if idx > trajectory_length:
            print(idx + 30)
            break

    error = np.asarray(error)

    mae, chart = eval_absolute_poses(relative_poses_pred, absolute_poses)

    return error, mae, chart, (idx + 30)

    if debug_mode:
        print("rel", relative_poses, "rel gt", relative_poses_pred)


if __name__ == "__main__":
    K = np.array([[373.2779426913342, 0.0, 318.29785021099894],
                  [0.0, 367.9439633567062, 263.9058079734077],
                  [0.0, 0.0, 1.0]])
    D = np.array([-0.3017710043972695, 0.07403470502924431, 0.0013028188828223006, 0.00022752165172560925, 0.0])

    dataset_dic = {
        "sub_folder": "/Users/julia/Documents/UNI/Master/Montréal/AV/project/duckietown_visual_odometry/data/",
        "filenames": ['alex_2small_8_retest_ground_truth.txt',
                      'alex_2small_loops_ground_truth.txt',
                      'alex_3small_loops_ground_truth.txt',
                      'alex_test_complex_2_ground_truth.txt',
                      'alex_train_complex_2_ground_truth.txt',
                      'razor_1small_8_ground_truth.txt',
                      'razor_2big_loops_ground_truth.txt',
                      'razor_2x3small_loops_ground_truth.txt',
                      'razor_test_incomplet_ground_truth.txt'],
        "dir": ['alex_2small_8_retest_images',
                'alex_2small_loops_images',
                'alex_3small_loops_images',
                'alex_test_complex_2_images',
                'alex_train_complex_2_images',
                'razor_1small_8_images',
                'razor_2big_loops_images',
                'razor_2x3small_loops_images',
                'razor_test_incomplet_images']}

    sub_folder = "/Users/julia/Documents/UNI/Master/Montréal/AV/project/duckietown_visual_odometry/data"
    os.chdir(sub_folder)

    df = pd.DataFrame(columns=["data", "poses", "MSE forward", "MSE horizontal",
                               "MSE vertical", "MSE orientation",
                               "MAE position x", "MAE position y",
                               "MAE orientation", "chart"])

    for i in range(len(dataset_dic["filenames"])):
        print(dataset_dic["filenames"][i])
        dataset = DuckietownDataset(dataset_dic["filenames"][i], dataset_dic["dir"][i], K=K, D=D,
                                    debug_mode=False)

        trajectory_len = len(dataset) - 31

        error, mae, chart, idx = calc_trajectory(dataset, trajectory_len, plot=False, debug_mode=False)

        df = df.append({"data": str(dataset_dic["filenames"][i]),
                        "poses": idx,
                        "MSE forward": np.mean(error[:, 0]),
                        "MSE horizontal": np.mean(error[:, 1]),
                        "MSE vertical": np.mean(error[:, 3]),
                        "MSE orientation": np.mean(error[:, 2]),
                        "MAE position x": mae[0],
                        "MAE position y": mae[1],
                        "MAE orientation": mae[2],
                        "chart": chart}, ignore_index=True)

    df.to_csv("stats_skip2_test.csv", float_format='%.6f')

    print(df)
