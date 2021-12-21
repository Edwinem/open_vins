import PyOpenVINS
import cv2
import click
import numpy as np
from utils import csv_read_matrix
from utils import TimestampSynchronizer
from openvins_config import get_options_from_launch_file
import os
from TrackBase import TrackBase, TrackDescriptorBase, TrackDescriptorOpenCV
from viewer import create_and_run
from multiprocessing import Process, Queue


def hamiltonian_quaternion_to_rot_matrix(q, eps=np.finfo(np.float64).eps):
    """
    Convert a hamiltonian quaternion to a rotation matrix.
    :param q: np array as w,x,y,z
    :param eps:
    :return: Rotation matrix as 3x3 np array

    Used for converting the euroc dataset attitude.
    """
    w, x, y, z = q
    squared_norm = np.dot(q, q)
    if squared_norm < eps:
        return np.eye(3)
    s = 2.0 / squared_norm
    wx = s * w * x
    wy = s * w * y
    wz = s * w * z
    xx = s * x * x
    xy = s * x * y
    xz = s * x * z
    yy = s * y * y
    yz = s * y * z
    zz = s * z * z
    return np.array([[1.0 - (yy + zz), xy - wz, xz + wy], [xy + wz, 1.0 - (xx + zz), yz - wx],
                     [xz - wy, yz + wx, 1.0 - (xx + yy)]])


LEFT_CAMERA_FOLDER = "cam0"
RIGHT_CAMERA_FOLDER = "cam1"
IMU_FOLDER = "imu0"
GT_FOLDER = "state_groundtruth_estimate0"
DATA_FILE = "data.csv"

TIMESTAMP_INDEX = 0

NANOSECOND_TO_SECOND = 1e-9

# Euroc fastest running sensor is the IMU at about 200 hz or 0.005 seconds.(Not exact). This is the actual value in
# nanoseconds.
EUROC_DELTA_TIME = 5000192


@click.command()
@click.option('--euroc_folder', required=True, help="Path to a folder containing Euroc data. Typically called mav0")
@click.option('--start_timestamp', required=True, help="Timestamp of where we want to start reading data from.")
@click.option('--launch_file', required=True,
              help="Path to a ros launch file from OpenVINS. Contains the settings for the algorithm")
@click.option('--use_viewer', is_flag=True, default=True, help="Use a 3D viwer to view the camera path")
def run_on_euroc(euroc_folder, start_timestamp, launch_file, use_viewer):
    # import os
    #
    # print(os.getpid())
    #
    # input("Enter to continue ...")

    options = get_options_from_launch_file(launch_file)
    manager = PyOpenVINS.VioManager(options)
    num_cameras = options.state_options.num_cameras
    # cam_intrins = manager.get_state().cam_intrinsics_cameras
    # histogram_method=PyOpenVINS.HistogramMethod.HISTOGRAM
    # tracker=TrackBase(cam_intrins,200,0,False,histogram_method)
    # desc_tracker = TrackDescriptorBase(cam_intrins,200,0,False,histogram_method)
    # orb_tracker = TrackDescriptorOpenCV(cam_intrins,250,0,False,histogram_method,options.fast_threshold, options.grid_x, options.grid_y, options.min_px_dist, options.knn_ratio)
    # vins_orb_tracker = PyOpenVINS.TrackDescriptor(cam_intrins,options.num_pts, options.state_options.max_aruco_features, options.use_stereo, histogram_method,
    #     options.fast_threshold, options.grid_x, options.grid_y, options.min_px_dist, options.knn_ratio)
    # manager.set_feature_tracker(orb_tracker)

    imu_data = csv_read_matrix(os.path.join(euroc_folder, IMU_FOLDER, DATA_FILE))
    camera_data = csv_read_matrix(os.path.join(euroc_folder, LEFT_CAMERA_FOLDER, DATA_FILE))
    imu_timestamps = [int(data[0]) for data in imu_data]
    camera_timestamps = [int(data[0]) for data in camera_data]
    ground_truth_data = csv_read_matrix(os.path.join(euroc_folder, GT_FOLDER, DATA_FILE))
    ground_truth_timestamps = [int(data[0]) for data in ground_truth_data]

    if use_viewer:
        est_pose_queue = None
        ground_truth_queue = None
        est_pose_queue = Queue()
        ground_truth_queue = Queue()
        viewer_process = Process(target=create_and_run, args=(est_pose_queue, ground_truth_queue))
        viewer_process.start()

    time_syncer = TimestampSynchronizer(int(EUROC_DELTA_TIME / 2))

    time_syncer.add_timestamp_stream("camera", camera_timestamps)
    time_syncer.add_timestamp_stream("imu", imu_timestamps)
    time_syncer.add_timestamp_stream("gt", ground_truth_timestamps)
    time_syncer.set_start_timestamp(int(start_timestamp))
    last_imu_timestamp = -1
    first_time = True

    while time_syncer.has_data():

        cur_data = time_syncer.get_data()
        if "imu" in cur_data:
            imu_index = cur_data["imu"].index
            imu_line = imu_data[imu_index]
            measurements = np.array([imu_line[1:]]).astype(np.float64).squeeze()

            gyro = np.array(measurements[0:3])
            acc = np.array(measurements[3:])
            timestamp = int(imu_line[TIMESTAMP_INDEX])
            if last_imu_timestamp != -1:
                dt = timestamp - last_imu_timestamp
            else:
                dt = EUROC_DELTA_TIME
            last_imu_timestamp = timestamp
            dt_seconds = dt * NANOSECOND_TO_SECOND
            timestamp_seconds = timestamp * NANOSECOND_TO_SECOND
            vins_imu_data = PyOpenVINS.ImuData()
            vins_imu_data.am = acc
            vins_imu_data.wm = gyro
            vins_imu_data.timestamp = timestamp_seconds
            manager.feed_measurement_imu(vins_imu_data)

        if "gt" in cur_data and first_time:
            gt_index = cur_data["gt"].index
            gt_line = ground_truth_data[gt_index]
            gt = np.array([gt_line[1:]]).astype(np.float64).squeeze()
            gt_pos = gt[0:3]
            gt_quat = gt[3:7]
            gt_vel = gt[7:10]
            gt_bias_gyro = gt[10:13]
            gt_bias_acc = gt[13:16]
            gt_timestamp = cur_data["gt"].timestamp * NANOSECOND_TO_SECOND
            gt_rot_matrx = hamiltonian_quaternion_to_rot_matrix(gt_quat)
            gt_jpl_quat = PyOpenVINS.rot_2_quat(np.asfortranarray(gt_rot_matrx.transpose()))
            gt_state = np.concatenate(
                [np.array([gt_timestamp]), gt_jpl_quat, gt_pos, gt_vel, gt_bias_gyro, gt_bias_acc])
            manager.initialize_with_gt(gt_state)

            first_time = False
            continue

        if "camera" in cur_data:
            index = cur_data["camera"].index
            image_name = camera_data[index][1]
            img = cv2.imread(os.path.join(euroc_folder, LEFT_CAMERA_FOLDER, "data", image_name), 0)
            img2 = cv2.imread(os.path.join(euroc_folder, RIGHT_CAMERA_FOLDER, "data", image_name), 0)
            cam_data = PyOpenVINS.CameraData()
            if num_cameras == 2:
                imgs = [img, img2]
                sensor_ids = [0, 1]
                mask = np.zeros_like(img)
                mask = mask.astype(np.uint8)
                masks = [mask, mask]
            else:  # 1 camera
                imgs = [img]
                sensor_ids = [0]
                mask = np.zeros_like(img)
                mask = mask.astype(np.uint8)
                masks = [mask]

            cam_data.images = imgs
            cam_data.sensor_ids = sensor_ids
            cam_data.timestamp = cur_data["camera"].timestamp * NANOSECOND_TO_SECOND
            cam_data.masks = masks

            manager.feed_measurement_camera(cam_data)
            state = manager.get_state()
            imu = state.imu
            est_pose = np.eye(4, dtype=np.float32)
            est_pose[0:3, 0:3] = imu.Rot().transpose()
            est_pose[0:3, 3] = imu.pos()
            if use_viewer and est_pose_queue:
                est_pose_queue.put(est_pose)

        if use_viewer and ground_truth_queue and "gt" in cur_data:
            gt_index = cur_data["gt"].index
            gt_line = ground_truth_data[gt_index]
            gt = np.array([gt_line[1:]]).astype(np.float64).squeeze()
            gt_pos = gt[0:3]
            gt_quat = gt[3:7]
            gt_rot_mat = hamiltonian_quaternion_to_rot_matrix(gt_quat)
            gt_transform = np.eye(4, dtype=np.float32)
            gt_transform[0:3, 0:3] = gt_rot_mat
            gt_transform[0:3, 3] = gt_pos
            ground_truth_queue.put(gt_transform)


if __name__ == '__main__':
    run_on_euroc()
