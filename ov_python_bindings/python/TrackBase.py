import math
from abc import abstractmethod

import PyOpenVINS
import cv2
import numpy as np


class TrackBase(PyOpenVINS.TrackBase):
    def __init__(self, camera_intrinsics, num_pts, max_aruco, use_stereo, histogram):
        super().__init__(camera_intrinsics, num_pts, max_aruco, use_stereo, histogram)

    @abstractmethod
    def feed_new_camera(self, camera_data):
        print(len(camera_data.images))


class TrackDescriptorBase(PyOpenVINS.TrackDescriptorBase):
    def __init__(self, camera_intrinsics, num_pts, max_aruco, use_stereo, histogram):
        super().__init__(camera_intrinsics, num_pts, max_aruco, use_stereo, histogram, 0, 0, 0, 0, 2.0,
                         "BruteForce-Hamming")

    def perform_detection_monocular(self, img0, mask0):
        ret_value = self.perform_detection_monocular_impl(img0, mask0)
        if (len(ret_value) != 3):
            raise ValueError("Return type of abstract method \"perform_detection_monocular_impl\" is not correct.")
        return ret_value

    @abstractmethod
    def perform_detection_monocular_impl(self, img0, mask0):
        pass


class TrackDescriptorOpenCV(PyOpenVINS.TrackDescriptorBase):

    def __init__(self, camera_intrinsics, num_pts, max_aruco, use_stereo, histogram, fast_threshold, grid_x, grid_y,
                 min_px_dist, knn_ratio):
        # super().__init__(camera_intrinsics,num_pts,max_aruco,use_stereo,histogram)
        super().__init__(camera_intrinsics, num_pts, max_aruco, use_stereo, histogram, fast_threshold, grid_x, grid_y,
                         min_px_dist, knn_ratio,
                         "BruteForce-Hamming")
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.num_features = num_pts
        self.fast_threshold = fast_threshold
        self.fast = cv2.FastFeatureDetector_create(threshold=self.fast_threshold)
        self.sub_pix_window_size = 5
        self.sub_pix_zero_zone = -1
        self.orb = cv2.ORB_create()
        self.min_px_dist = min_px_dist
        self.id_count = 0
        self.prev_des = None

    def perform_detection_monocular(self, img0, mask0):
        ret_value = self.perform_detection_monocular_impl(img0, mask0)
        if (len(ret_value) != 3):
            raise ValueError("Return type of abstract method \"perform_detection_monocular_impl\" is not correct.")
        return ret_value

    @abstractmethod
    def perform_detection_monocular_impl(self, img0, mask0):
        cols = img0.shape[1]
        rows = img0.shape[0]
        size_x = int(cols / self.grid_x)
        size_y = int(rows / self.grid_y)
        num_features_grid = int((self.num_features / (self.grid_x * self.grid_y))) + 1

        ct_cols = math.floor(cols / size_x)
        ct_rows = math.floor(rows / size_y)
        detected_keypoints = []
        for r in range(0, ct_rows * ct_cols):
            # Calculate what cell xy value we are in
            x = int(int(r % ct_cols) * size_x)
            y = int(int(r / ct_cols) * size_y)

            if x + size_x > cols or y + size_y > rows:
                continue

            sub_img = img0[y:y + size_y, x:x + size_x]
            keypoints = self.fast.detect(sub_img)
            if len(keypoints) == 0:
                continue
            sorted_keypoints = sorted(keypoints, reverse=True, key=lambda x: x.response)
            for i in range(0, min(num_features_grid, len(sorted_keypoints))):
                kp = sorted_keypoints[i]
                pt_x = kp.pt[0] + float(x)
                pt_y = kp.pt[1] + float(y)

                # Reject if out of bounds(shouldn't be possible...)
                if x + size_x > cols or y + size_y > rows:
                    continue

                if mask0[int(pt_y), int(pt_x)] > 127:
                    continue
                kp.pt = (pt_x, pt_y)
                detected_keypoints.append(kp)

        criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 20, 0.001)
        # Try to detect with subpixel accuracy.
        pts_to_refine = np.empty((len(detected_keypoints), 2), np.float32)
        for idx, kp in enumerate(detected_keypoints):
            pts_to_refine[idx, 0] = kp.pt[0]
            pts_to_refine[idx, 1] = kp.pt[1]

        refined_corners = cv2.cornerSubPix(img0, pts_to_refine,
                                           (self.sub_pix_window_size, self.sub_pix_window_size),
                                           (self.sub_pix_zero_zone, self.sub_pix_zero_zone), criteria)

        # Copy refine pts back
        for idx in range(0, refined_corners.shape[0]):
            detected_keypoints[idx].pt = (refined_corners[idx, 0], refined_corners[idx, 1])
        # detected2 = PyOpenVINS.grid_fast(img0,mask0,self.num_features,self.grid_x,self.grid_y,self.fast_threshold,True)
        # cpp_keypoints = []
        # for tup in detected2:
        #     cpp_keypoints.append(cv2.KeyPoint(x=tup[0], y=tup[1], _size=tup[2],
        #                  _angle=tup[3], _response=tup[4],
        #                  _octave=tup[5], _class_id=tup[6]))
        orb_keypoints, orb_descriptors = self.orb.compute(img0, detected_keypoints)
        # for i in range(0,len(detected_keypoints)):
        #     kpt = detected_keypoints[i]
        #     tup=detected2[i]
        #     if not(math.isclose(kpt.pt[0],tup[0]) and math.isclose(kpt.pt[1],tup[1])):
        #         print("diff")

        width = int(cols / float(self.min_px_dist))
        height = int(rows / float(self.min_px_dist))

        grid = np.zeros((height, width))

        final_kp = []
        final_descriptors = np.empty_like(orb_descriptors)
        final_desriptors_counter = 0
        ids = []
        for idx, kpt in enumerate(orb_keypoints):
            x = int(kpt.pt[0])
            y = int(kpt.pt[1])
            x_grid = int(kpt.pt[0] / float(self.min_px_dist))
            y_grid = int(kpt.pt[1] / float(self.min_px_dist))
            if x_grid < 0 or x_grid >= width or y_grid < 0 or y_grid >= height or x < 0 or x >= cols or y < 0 or y >= rows:
                continue

            if grid[y_grid, x_grid] > 127:
                continue

            grid[y_grid, x_grid] = 255
            final_kp.append(kpt)
            final_descriptors[final_desriptors_counter, :] = orb_descriptors[idx, :]
            final_desriptors_counter += 1
            ids.append(self.get_currid_value())
            self.increment_currid()
        final_descriptors = final_descriptors[0:final_desriptors_counter, :]
        return final_kp, final_descriptors, ids
