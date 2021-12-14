import os
import pandas as pd
# from torchvision.io import read_image
import cv2
from torch.utils.data import Dataset
from os import walk
import numpy as np
import torch


class DuckietownDataset(Dataset):
    def __init__(self, annotations_file, img_dir, K, D):
        self.absolute_pos = pd.read_fwf(annotations_file)
        self.img_dir = img_dir
        filenames_dir = sorted(next(walk(self.img_dir), (None, None, []))[2])
        self.absolute_pos['img'] = filenames_dir
        self.absolute_pos = self.absolute_pos[55:]
        self.skip = 1
        self.K = K
        self.D = D

    def __len__(self):
        return len(self.absolute_pos // self.skip)

    def __getitem__(self, idx):
        data = []
        old_pose_idx = idx*self.skip
        new_pose_idx = (idx+1)*self.skip
        for i in (old_pose_idx, new_pose_idx):
            img_path = os.path.join(self.img_dir, self.absolute_pos.iloc[i]["img"])
            img = cv2.imread(img_path)
            img, newK = self.preprocess(img)
            print(self.absolute_pos.iloc[i]["img"], img.shape)
            data.append(img)
        data = np.array(data)
        old_pose = self.absolute_pos.iloc[old_pose_idx][["x", "y", "theta_correct"]].to_numpy()
        new_pose = self.absolute_pos.iloc[new_pose_idx][["x", "y", "theta_correct"]].to_numpy()
        relative_pos = compute_relative_pose_matrix(old_pose, new_pose)
        scale = np.linalg.norm(new_pose[:2] - old_pose[:2])
        # print("old_pose", old_pose, "new pose", new_pose, "scale", scale, newK)
        return data, relative_pos, scale, new_pose, old_pose, newK


    def preprocess(self, img):
        # img = img[160:480, 0:640]
        correct_img = True
        fisheye = False
        fisheye2 = False
        if correct_img:
            height, width = img.shape[:2]
            # print(img.shape, h, w)
            newK, roi = cv2.getOptimalNewCameraMatrix(self.K, self.D, (width, height), 1,
                                                              (width, height))
            # mapx, mapy = cv2.initUndistortRectifyMap(self.K, self.D, None, newK, dim, 5)
            # undistorted_image = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
            undistorted_image = cv2.undistort(img, self.K, self.D, None, newK)

            x, y, w, h = roi
            undistorted_image = undistorted_image[y:y+h, x:x+w]

            img = undistorted_image
            print("image corrected")

        elif fisheye:

            new_size = img.shape

            print(self.D[:-1], self.K, self.D.dtype)

            self.D = self.D.astype(np.float32)

            newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(self.K, self.D[:-1], (new_size[1], new_size[0]),
                                                                          np.eye(3), balance=1, fov_scale=1)
            unfishmap1, unfishmap2 = cv2.fisheye.initUndistortRectifyMap(K=self.K, D=self.D[:-1], R=np.eye(3), P=newK,
                                                                         size=(new_size[1], new_size[0]),
                                                                         m1type=cv2.CV_32F)
            unfishmap1, unfishmap2 = cv2.convertMaps(unfishmap1, unfishmap2, cv2.CV_16SC2)

            img = cv2.remap(img, unfishmap1, unfishmap2, interpolation=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT)
            print("image corrected fisheye")
        elif fisheye2:
            new_size = img.shape
            newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(self.K, self.D[:-1],
                                                                          (new_size[1], new_size[0]),
                                                                          np.eye(3), balance=1, fov_scale=1)
            img = cv2.fisheye.undistortImage(img, self.K, D=self.D[:-1], Knew=newK)
            print("image corrected fisheye2")
        else:
            newK = self.K
            print("image not corrected")

        # cv2.imshow("undistorted", undistorted_image)
        # cv2.waitKey()

        return img, newK


def rotation(x_n, y_n, theta_n):
    return np.array([[np.cos(theta_n), -np.sin(theta_n), x_n],
                     [np.sin(theta_n), np.cos(theta_n), y_n], [0, 0, 1]])


def inverse_rotation(x_n, y_n, theta_n):
    return np.array([[np.cos(theta_n), np.sin(theta_n), -np.sin(theta_n) * y_n - np.cos(theta_n) * x_n],
                     [-np.sin(theta_n), np.cos(theta_n), -np.cos(theta_n) * y_n + np.sin(theta_n) * x_n]])


def compute_relative_pose(pose1, pose2):
    print("original pose", pose1, pose2)
    relative_theta = pose2[2] - pose1[2]
    pose2_copy = pose2
    pose2_copy[2] = 1
    pose1_copy = pose1
    pose1_copy[2] = 1
    pose2wrt1 = inverse_rotation(*pose1_copy).dot(pose2_copy)
    print("change in pose", pose2wrt1, relative_theta)
    print("check", pose2, np.dot(rotation(*pose1_copy), [*pose2wrt1, 1]))

    return pose2wrt1, relative_theta


def compute_relative_pose_matrix(pose1, pose2):
    matrix1 = get_pose_matrix(pose1)
    matrix2 = get_pose_matrix(pose2)

    prod_matrix = np.dot(np.linalg.inv(matrix1), matrix2)
    # prod_matrix = np.matmul(, )

    # print(matrix1, matrix2)
    # print(prod_matrix)

    return prod_matrix


def get_pose_matrix(pose):
    """matrix = np.asarray([[np.cos(pose[2]), -np.sin(pose[2]), 0, pose[0]],
                        [0, 0, 1, 0],
                        [np.sin(pose[2]), np.cos(pose[2]), 0, pose[1]],
                        [0, 0, 0, 1]])"""

    angle = pose[2]

    matrix = np.asarray([[np.cos(angle), 0, np.sin(angle), pose[0]],
                         [0, 1, 0, 0],
                         [-np.sin(angle), 0, np.cos(angle), pose[1]],
                         [0, 0, 0, 1]])

    return matrix

def get_rotation_matrix(angle):
    """matrix = np.asarray([[np.cos(pose[2]), -np.sin(pose[2]), 0, pose[0]],
                        [0, 0, 1, 0],
                        [np.sin(pose[2]), np.cos(pose[2]), 0, pose[1]],
                        [0, 0, 0, 1]])"""

    matrix = np.asarray([[np.cos(angle), np.sin(angle), 0],
                         [-np.sin(angle), np.cos(angle), 0],
                         [0, 0, 1]])

    return matrix


"""

def absolute2relative(absolute_poses):
        
    n_relative_poses = absolute_poses.shape[0]-1
    relative_poses = np.zeros((n_relative_poses, 3), dtype='float32')
    relative_thetas = np.zeros(n_relative_poses, dtype='float32')
    
    absolute_thetas = absolute_poses[:,-1]
    
    copy_absolute_poses = np.copy(absolute_poses)
    copy_absolute_poses[:,-1] = 1
    
    for i in range(n_relative_poses):
        relative_poses[i] = inverse_rotation(absolute_poses[i]).dot(copy_absolute_poses[i+1])
        relative_thetas[i] = absolute_thetas[i+1]-absolute_thetas[i]
    
    relative_poses[:,-1] = relative_thetas
        
    return relative_poses
    
def relative2absolute(relative_poses, absolute_pose_0):
    n_absolute_poses = relative_poses.shape[0] + 1
    absolute_poses = np.zeros((n_absolute_poses, 3), dtype='float32')
    absolute_thetas = np.zeros(n_absolute_poses, dtype='float32')

    relative_thetas = relative_poses[:, -1]

    copy_relative_poses = np.copy(relative_poses)
    copy_relative_poses[:, -1] = 1

    absolute_poses[0] = absolute_pose_0
    absolute_thetas[0] = absolute_pose_0[-1]

    for i in range(n_absolute_poses - 1):
        absolute_poses[i + 1] = rotation(absolute_poses[i]).dot(copy_relative_poses[i])
        absolute_thetas[i + 1] = relative_thetas[i] + absolute_thetas[i]
        absolute_poses[i + 1][-1] = absolute_thetas[i + 1]

    absolute_poses[:, -1] = absolute_thetas

    return absolute_poses

    def snippet(pose1, pose2):
        pose2wrt1 = np.dot(np.linalg.inv(pose1), pose2)
        R = pose2wrt1[0:3, 0:3]
        t = (torch.from_numpy(pose2wrt1[0:3, 3]).view(-1, 3)).float().cuda()
        axisAngle = (torch.from_numpy(np.asarray(rotMat_to_axisAngle(R))).view(-1, 3)).float()
"""

if __name__ == "__main__":
    """test_dataloader = DuckietownDataset("alex_2small_loops_ground_truth.txt", "alex_2small_loops_images")

    for data, rel_pos in test_dataloader:
        print(data.shape, rel_pos)
        break"""

