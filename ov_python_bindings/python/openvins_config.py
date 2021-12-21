import xml.etree.ElementTree as ET
import numpy as np
import PyOpenVINS

map_type_to_python_type = {"string": str, "str": str, 'bool': bool, 'int': int, "double": float}

def convert_bool(str):
    if str == "false":
        return False
    return True

def get_openvins_options_from_yaml(yaml_filepath):
    """
    Get the VioManager options from a Yaml config file.
    Args:
        yaml_filepath: File path to yaml config

    Returns:
        A VioManager filled with the parameters from the config file.

    """
    yaml_parser=PyOpenVINS.YamlParser(yaml_filepath,True)
    params = PyOpenVINS.VioManagerOptions()
    params.print_and_load(yaml_parser)
    return params


def parse_launch_file_to_dict(launch_filepath):
    """
    Parses a ros launch file to a python dictionary
    Args:
        launch_filepath:

    Returns:

    """
    tree = ET.parse(launch_filepath)
    root = tree.getroot()

    dict = {}
    for elem in root.iter():
        if (elem.tag == "arg"):
            arg_name = elem.attrib.get("name")
            value_tag = elem.attrib.get("default")
            dict[arg_name] = value_tag

        if (elem.tag == "param"):
            name = elem.attrib.get("name")
            type = elem.attrib.get("type")
            value_tag = elem.attrib.get("value")
            if value_tag.find("$(arg") != -1:
                fixed_str = value_tag
                fixed_str = fixed_str.replace("$(arg ", "")
                fixed_str = fixed_str.replace(")", "")
                dict[name] = dict[fixed_str]
            else:
                python_type = map_type_to_python_type[type]
                if type == "bool":
                    value = convert_bool(value_tag)
                else:
                    value = python_type(value_tag)
                dict[name] = value
        if (elem.tag == "rosparam"):
            name = elem.attrib.get("param")
            dict[name] = elem.text
    return dict

def get_openvins_options_from_dict(dict):
    params = PyOpenVINS.VioManagerOptions()

    params.state_options.do_fej = dict["use_fej"]
    params.state_options.imu_avg = dict["use_imuavg"]
    params.state_options.use_rk4_integration = bool(dict["use_rk4int"])
    params.state_options.do_calib_camera_pose = bool(dict["calib_cam_extrinsics"])
    params.state_options.do_calib_camera_intrinsics = bool(dict["calib_cam_intrinsics"])
    params.state_options.do_calib_camera_timeoffset = bool(dict["calib_cam_timeoffset"])
    params.state_options.max_clone_size = int(dict["max_clones"])
    params.state_options.max_slam_features = int(dict["max_slam"])
    params.state_options.max_slam_in_update = int(dict["max_slam_in_update"])
    params.state_options.max_msckf_in_update = int(dict["max_msckf_in_update"])
    params.state_options.max_aruco_features = int(dict["num_aruco"])
    params.state_options.num_cameras = int(dict["max_cameras"])
    params.dt_slam_delay = float(dict["dt_slam_delay"])

    # Filter initialization
    params.init_options.init_window_time = float(dict["init_window_time"])
    params.init_options.init_imu_thresh = float(dict["init_imu_thresh"])

    # Zero velocity update
    params.try_zupt = bool(dict["try_zupt"])
    params.zupt_options.chi2_multipler = float(dict["zupt_chi2_multipler"])
    params.zupt_max_velocity = float(dict["zupt_max_velocity"])
    params.zupt_noise_multiplier = float(dict["zupt_noise_multiplier"])
    params.zupt_max_disparity = float(dict["zupt_max_disparity"])
    params.zupt_only_at_beginning = bool(dict["zupt_only_at_beginning"])

    params.state_options.feat_rep_msckf = PyOpenVINS.LandmarkRepresentation.from_string(dict["feat_rep_msckf"])
    params.state_options.feat_rep_slam = PyOpenVINS.LandmarkRepresentation.from_string(dict["feat_rep_slam"])
    params.state_options.feat_rep_aruco = PyOpenVINS.LandmarkRepresentation.from_string(dict["feat_rep_aruco"])

    # Our noise values for inertial sensor
    params.imu_noises.sigma_w = float(dict["gyroscope_noise_density"])
    params.imu_noises.sigma_a = float(dict["accelerometer_noise_density"])
    params.imu_noises.sigma_wb = float(dict["gyroscope_random_walk"])
    params.imu_noises.sigma_ab = float(dict["accelerometer_random_walk"])

    # Read in update parameters

    params.msckf_options.sigma_pix = float(dict["up_msckf_sigma_px"])
    params.msckf_options.chi2_multipler = float(dict["up_msckf_chi2_multipler"])
    params.slam_options.sigma_pix = float(dict["up_slam_sigma_px"])
    params.slam_options.chi2_multipler = float(dict["up_slam_chi2_multipler"])
    params.aruco_options.sigma_pix = float(dict["up_aruco_sigma_px"])
    params.aruco_options.chi2_multipler = float(dict["up_aruco_chi2_multipler"])
    params.use_stereo = bool(dict["use_stereo"])
    params.use_klt = bool(dict["use_klt"])
    params.use_aruco = bool(dict["use_aruco"])
    params.downsize_aruco = bool(dict["downsize_aruco"])
    params.downsample_cameras = bool(dict["downsample_cameras"])
    params.use_multi_threading = bool(dict["multi_threading"])

    # General parameters
    params.num_pts = int(dict["num_pts"])
    params.fast_threshold = int(dict["fast_threshold"])
    params.grid_x = int(dict["grid_x"])
    params.grid_y = int(dict["grid_y"])
    params.min_px_dist = int(dict["min_px_dist"])
    params.knn_ratio = float(dict["knn_ratio"])

    # params.params.featinit_options.triangulate_1d = bool(dict["fi_triangulate_1d"])
    # params.featinit_options.refine_features= bool(dict["fi_refine_features"])
    # params.featinit_options.max_runs= int(dict["fi_max_runs"])
    # params.featinit_options.init_lamda = float(dict["fi_init_lamda"])
    # params.featinit_options.max_lamda = float(dict["fi_max_lamda"])
    # params.featinit_options.min_dx = float(dict["fi_min_dx"])
    # params.featinit_options.min_dcost = float(dict["fi_min_dcost"])
    # params.featinit_options.lam_mult = float(dict["fi_lam_mult"])
    # params.featinit_options.min_dist = float(dict["fi_min_dist"])
    # params.featinit_options.max_dist = float(dict["fi_max_dist"])
    # params.featinit_options.max_baseline = float(dict["fi_max_baseline"])
    # params.featinit_options.max_cond_number = float(dict["fi_max_cond_number"])

    def extract_to_numpy(dict, key):
        data_as_str = dict[key]
        # Remove the brackets [ and ]
        data_as_str = data_as_str.replace("[", "")
        data_as_str = data_as_str.replace("]", "")
        return np.fromstring(data_as_str, sep=",")

    for i in range(0, params.state_options.num_cameras):

        # If our distortions are fisheye or not !
        is_fisheye = bool(dict[f"cam{i}_is_fisheye"])

        # If the desired fov we should simulate
        width_height_np = extract_to_numpy(dict, f"cam{i}_wh")
        if params.downsample_cameras:
            width_height_np = width_height_np / 2
        wh = (int(width_height_np[0]), int(width_height_np[1]))

        intrinsics_np = extract_to_numpy(dict, f"cam{i}_k")
        distortion_np = extract_to_numpy(dict, f"cam{i}_d")
        if params.downsample_cameras:
            intrinsics_np /= 2.0

        camera_calib = np.concatenate([intrinsics_np, distortion_np])

        extrinsics_np = extract_to_numpy(dict, f"T_C{i}toI")
        extrinsics_np = extrinsics_np.reshape(4, 4)
        extrinsics = np.asfortranarray(extrinsics_np)
        rot_matrix = extrinsics[0:3, 0:3]
        quat = PyOpenVINS.rot_2_quat(rot_matrix.transpose())
        trans = -rot_matrix.transpose() @ extrinsics[0:3, 3]
        quat_and_trans = np.concatenate([quat, trans])
        params.camera_fisheye[i] = is_fisheye
        params.camera_intrinsics[i] = camera_calib
        params.camera_extrinsics[i] = quat_and_trans
        params.camera_wh[i] = wh
    return params

def get_options_from_launch_file(launch_filepath):
    dict=parse_launch_file_to_dict(launch_filepath)
    return get_openvins_options_from_dict(dict)