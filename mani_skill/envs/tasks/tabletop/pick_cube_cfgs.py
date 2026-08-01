"""
PickCube-v1 is a basic/common task which defaults to using the panda robot. It is also used as a testing task to check whether a robot with manipulation
capabilities can be simulated and trained properly. The configs below set the pick cube task differently to ensure the cube is within reach of the robot tested
and the camera angles are reasonable.
"""

import math

# Match robust-rearrangement-custom's simulated front camera in the Panda base
# frame. RR places the Panda base at (-0.3, 0, 0.415), its camera at
# (0.9, 0, 0.65), and its look-at target at (-1, 0, 0.3). ManiSkill places the
# Panda base at (-0.615, 0, 0), so preserving the two relative transforms gives
# the world-space eye and target below. RR uses a square image, so its 40-degree
# horizontal FOV is also the vertical FOV expected by CameraConfig.
RR_ALIGNED_PANDA_FRONT_CAMERA = {
    "sensor_cam_eye_pos": [0.585, 0.0, 0.235],
    "sensor_cam_target_pos": [-1.315, 0.0, -0.115],
    "sensor_cam_width": 224,
    "sensor_cam_height": 224,
    "sensor_cam_fov": math.radians(40.0),
    "sensor_cam_near": 0.001,
    "sensor_cam_far": 2.0,
}

PICK_CUBE_CONFIGS = {
    "panda": {
        "cube_half_size": 0.02,
        "goal_thresh": 0.025,
        "cube_spawn_half_size": 0.1,
        "cube_spawn_center": (0, 0),
        "max_goal_height": 0.3,
        # Sensor cam is the camera used for visual observation generation.
        **RR_ALIGNED_PANDA_FRONT_CAMERA,
        "human_cam_eye_pos": [
            0.6,
            0.7,
            0.6,
        ],  # human cam is the camera used for human rendering (i.e. eval videos)
        "human_cam_target_pos": [0.0, 0.0, 0.35],
    },
    "fetch": {
        "cube_half_size": 0.02,
        "goal_thresh": 0.025,
        "cube_spawn_half_size": 0.1,
        "cube_spawn_center": (0, 0),
        "max_goal_height": 0.3,
        "sensor_cam_eye_pos": [0.3, 0, 0.6],
        "sensor_cam_target_pos": [-0.1, 0, 0.1],
        "human_cam_eye_pos": [0.6, 0.7, 0.6],
        "human_cam_target_pos": [0.0, 0.0, 0.35],
    },
    "xarm6_robotiq": {
        "cube_half_size": 0.02,
        "goal_thresh": 0.025,
        "cube_spawn_half_size": 0.1,
        "cube_spawn_center": (0, 0),
        "max_goal_height": 0.3,
        "sensor_cam_eye_pos": [0.3, 0, 0.6],
        "sensor_cam_target_pos": [-0.1, 0, 0.1],
        "human_cam_eye_pos": [0.6, 0.7, 0.6],
        "human_cam_target_pos": [0.0, 0.0, 0.35],
    },
    "so100": {
        "cube_half_size": 0.0125,
        "goal_thresh": 0.0125 * 1.25,
        "cube_spawn_half_size": 0.05,
        "cube_spawn_center": (-0.46, 0),
        "max_goal_height": 0.08,
        "sensor_cam_eye_pos": [-0.27, 0, 0.4],
        "sensor_cam_target_pos": [-0.56, 0, -0.25],
        "human_cam_eye_pos": [-0.1, 0.3, 0.4],
        "human_cam_target_pos": [-0.46, 0.0, 0.1],
    },
    "widowxai": {
        "cube_half_size": 0.018,
        "goal_thresh": 0.018 * 1.25,
        "cube_spawn_half_size": 0.05,
        "cube_spawn_center": (-0.25, 0),
        "max_goal_height": 0.2,
        "sensor_cam_eye_pos": [0.0, 0, 0.35],
        "sensor_cam_target_pos": [-0.2, 0, 0.1],
        "human_cam_eye_pos": [0.45, 0.5, 0.5],
        "human_cam_target_pos": [-0.2, 0.0, 0.2],
    },
}

# The wrist-camera Panda has identical task geometry and uses the same
# RR-aligned fixed front camera as the standard Panda.
PICK_CUBE_CONFIGS["panda_wristcam"] = PICK_CUBE_CONFIGS["panda"].copy()
