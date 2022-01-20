import os
import pandas as pd
import cv2
from torch.utils.data import Dataset
from os import walk
import numpy as np
import utils
import sys

class DuckietownDataset(Dataset):
    def __init__(self, annotations_file, img_dir, K, D, debug_mode=False):
        # read file with ground truth data
        self.absolute_pos = pd.read_fwf(annotations_file)
        self.img_dir = img_dir
        filenames_dir = sorted(next(walk(self.img_dir), (None, None, []))[2])
        self.absolute_pos['img'] = filenames_dir
        # skip first 30 poses as duckiebots were not moving yet
        self.absolute_pos = self.absolute_pos[30:]
        # skipping of frames
        self.skip = 1
        # intrinsic camera matrix K
        self.K = K
        # distorsion vector
        self.D = D
        # enable for print outs
        self.debug_mode = debug_mode
        if debug_mode:
            with pd.option_context('display.max_rows', None, 'display.max_columns',
                               None):  # more options can be specified also
                print(self.absolute_pos)

    def __len__(self):
        return (len(self.absolute_pos) // self.skip) - 1

    def __getitem__(self, idx):
        data = []
        old_pose_idx = idx * self.skip
        new_pose_idx = (idx + 1) * self.skip
        for i in (old_pose_idx, new_pose_idx):
            img_path = os.path.join(self.img_dir, self.absolute_pos.iloc[i]["img"])
            img = cv2.imread(img_path)
            if img.size == 0:
                print(f"ERROR with image {i}")
                sys.exit()
            img, newK = self.preprocess(img)
            if self.debug_mode:
                print(self.absolute_pos.iloc[i]["img"], img.shape)
            data.append(img)
        data = np.array(data)
        absolute_pose = self.absolute_pos.iloc[old_pose_idx:new_pose_idx+1:self.skip][["x", "y", "theta_correct"]].to_numpy()
        rel_pose = utils.absolute2relative(absolute_pose)
        return idx, data, absolute_pose[1], absolute_pose[0], rel_pose, newK

    def preprocess(self, img):
        # img = img[160:480, 0:640]
        # select preprocessing techniques
        correct_img = True
        fisheye = False
        fisheye2 = False
        # undistort image and cropping
        if correct_img:
            height, width = img.shape[:2]
            newK, roi = cv2.getOptimalNewCameraMatrix(self.K, self.D, (width, height), 1,
                                                      (width, height))
            undistorted_image = cv2.undistort(img, self.K, self.D, None, newK)

            x, y, w, h = roi
            undistorted_image = undistorted_image[y:y + h, x:x + w]

            img = undistorted_image
            if self.debug_mode:
                print("image corrected")
        # fisheye specific undistorsion (didn't work as well)
        elif fisheye:
            new_size = img.shape

            print(self.D[:-1], self.K, self.D.dtype)

            self.D = self.D.astype(np.float32)

            newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(self.K, self.D[:-1],
                                                                          (new_size[1], new_size[0]),
                                                                          np.eye(3), balance=1, fov_scale=1)
            unfishmap1, unfishmap2 = cv2.fisheye.initUndistortRectifyMap(K=self.K, D=self.D[:-1], R=np.eye(3), P=newK,
                                                                         size=(new_size[1], new_size[0]),
                                                                         m1type=cv2.CV_32F)
            unfishmap1, unfishmap2 = cv2.convertMaps(unfishmap1, unfishmap2, cv2.CV_16SC2)

            img = cv2.remap(img, unfishmap1, unfishmap2, interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT)
            if self.debug_mode:
                print("image corrected fisheye")

        # fisheye specific undistorsion variation 2 (didn't work as well)
        elif fisheye2:
            new_size = img.shape
            newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(self.K, self.D[:-1],
                                                                          (new_size[1], new_size[0]),
                                                                          np.eye(3), balance=1, fov_scale=1)
            img = cv2.fisheye.undistortImage(img, self.K, D=self.D[:-1], Knew=newK)
            if self.debug_mode:
                print("image corrected fisheye2")
        else:
            newK = self.K
            if self.debug_mode:
                print("image not corrected")

        return img, newK

# used for testing of dataset class
if __name__ == "__main__":
    K = np.array([[373.2779426913342, 0.0, 318.29785021099894],
                  [0.0, 367.9439633567062, 263.9058079734077],
                  [0.0, 0.0, 1.0]])
    D = np.array([-0.3017710043972695, 0.07403470502924431, 0.0013028188828223006, 0.00022752165172560925, 0.0])
    dataset = DuckietownDataset("alex_2small_loops_ground_truth.txt", "alex_2small_loops_images", K=K, D=D,
                                debug_mode=False)

    for all in dataset:
        break
