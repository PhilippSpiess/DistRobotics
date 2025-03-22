# Copyright (c) 2024 Philipp Spiess
# All rights reserved.

import os
import cv2

# THE "" are placeholders so that new tasks can be added without changing the dimensions of the neural network
TASKS = ["standing", "standing_up", "walking", "jumping","squat","","","","","","","","","","","","","","",""]

NUM_BOTS = 256
FRICTION = 0.9
PARALLEL_SIM = False
SIMULATION_FREQUENCY = 30       # 30Hz for velocity - 120 Hz for torque
BLOCKING = False
CONTROL = "velocity"              # "torque" , "position", "velocity"

MAX_VELOCITY = 1.6              # All velocities are between 0 and 1.6 - 16RPM is equivalent to 1.6 rad/s
MAX_RPM = 16

# Domain randomization
HEIGHT_SCALE = 0.0
FRICTION_LOW, FRICTION_HIGH = 0.9, 1
NOISE = 0.3

NOISE_ANGLE_STANDING_UP = 0.75
DUCK_X_POS = 10
ENERGY_FACTOR = 500
SMOOTH_FACTOR = 500

MIMIC_CONSTANTS = {
    "ORIENTATION_WEIGHT": 0.65 * 5,
    "ORIENTATION_EXPONENT": -2,

    "ANGLE_VELOCITY_WEIGHT": 0.15 * 0,
    "ANGLE_VELOCITY_EXPONENT": -0.1,

    "END_EFFECTOR_WEIGHT": 0.15 * 0,
    "END_EFFECTOR_EXPONENT": -40,

    "CENTER_MASS_WEIGHT": 0.15 * 0,
    "CENTER_MASS_EXPONENT": -10,
}

WEIGHT_REWARD = 0.3
WEIGHT_REWARD_MIMIC = 0.7
HEIGHT_ROOT = 0.97

# Define robot types as constants
ROBOT_TYPE_LEGS = "legs"
ROBOT_TYPE_HALF = "half"

# Define a mapping for action and state sizes
ROBOT_CONFIGS = {
    ROBOT_TYPE_LEGS: {"action_size": 8, "state_size": (48,)},
    ROBOT_TYPE_HALF: {"action_size": 18, "state_size": (78,)},
}

FORCE_MULTIPLIER = 1

ORIENTATION_OFFSET_PITCH = -0.1

#########            FIXED PARAMETERS           ##########

STEP_FREQUENCY = 240
FRAMES_FREQUENCY = 30
GRAVITY = -9.81
WAIT_TIME = 1 / STEP_FREQUENCY

relative_path = os.getcwd()
PATH = os.path.abspath(os.path.expanduser(relative_path))
VISION_PATH = 'vision'

MAX_TORQUES = {
    "root_chest": 7,
    "neck": 7,
    "right_shoulder_pitch": 7,
    "right_shoulder_roll": 7,
    "right_elbow": 7,
    "right_wrist": 7,
    "right_hand_to_thumb_base": 0.5,
    "right_thumb_base_to_center": 0.5,
    "right_hand_to_index_base": 0.5,
    "right_index_base_to_center": 0.5,
    "right_hand_to_middle_base": 0.5,
    "right_middle_base_to_center": 0.5,
    "right_hand_to_ring_base": 0.5,
    "right_ring_base_to_center": 0.5,
    "right_hand_to_pinky_base": 0.5,
    "right_pinky_base_to_center": 0.5,
    "left_shoulder_pitch": 7,
    "left_shoulder_roll": 7,
    "left_elbow": 7,
    "left_wrist": 7,
    "left_hand_to_thumb_base": 0.5,
    "left_thumb_base_to_center": 0.5,
    "left_hand_to_index_base": 0.5,
    "left_index_base_to_center": 0.5,
    "left_hand_to_middle_base": 0.5,
    "left_middle_base_to_center": 0.5,
    "left_hand_to_ring_base": 0.5,
    "left_ring_base_to_center": 0.5,
    "left_hand_to_pinky_base": 0.5,
    "left_pinky_base_to_center": 0.5,
    "right_hip_yaw": 10,
    "right_hip_pitch": 20,
    "right_knee": 20,
    "right_ankle": 20,
    "left_hip_yaw": 10,
    "left_hip_pitch": 20,
    "left_knee": 20,
    "left_ankle": 20
}

MIN_POSITIONS = {
    "root_chest": -0.5,
    "neck": -1.4,
    "right_shoulder_roll": -1.57,
    "right_shoulder_pitch": -1.57,
    "right_elbow": 0,
    "right_wrist": -1,
    "right_hand_to_thumb_base": 0,
    "right_thumb_base_to_center": -1,
    "right_hand_to_index_base": 0,
    "right_index_base_to_center": 0,
    "right_hand_to_middle_base": 0,
    "right_middle_base_to_center": 0,
    "right_hand_to_ring_base": 0,
    "right_ring_base_to_center": 0,
    "right_hand_to_pinky_base": 0,
    "right_pinky_base_to_center": 0,
    "left_shoulder_roll": 0,
    "left_shoulder_pitch": -1,
    "left_elbow": 0,
    "left_wrist": -1,
    "left_hand_to_thumb_base": 0,
    "left_thumb_base_to_center": -1,
    "left_hand_to_index_base": 0,
    "left_index_base_to_center": 0,
    "left_hand_to_middle_base": 0,
    "left_middle_base_to_center": 0,
    "left_hand_to_ring_base": 0,
    "left_ring_base_to_center": 0,
    "left_hand_to_pinky_base": 0,
    "left_pinky_base_to_center": 0,
    "right_hip_yaw": 0,
    "right_hip_pitch": -1.57,
    "right_knee": -3.14,
    "right_ankle": -1,
    "left_hip_yaw": -0.7,
    "left_hip_pitch": -1.57,
    "left_knee": -3.14,
    "left_ankle": -1
}

MAX_POSITIONS = {
    "root_chest": 0.5,
    "neck": 1.4,
    "right_shoulder_roll": 0,
    "right_shoulder_pitch": 2.5,
    "right_elbow": 2.5,
    "right_wrist": 1,
    "right_hand_to_thumb_base": 1,
    "right_thumb_base_to_center": 0,
    "right_hand_to_index_base": 1,
    "right_index_base_to_center": 1,
    "right_hand_to_middle_base": 1,
    "right_middle_base_to_center": 1,
    "right_hand_to_ring_base": 1,
    "right_ring_base_to_center": 1,
    "right_hand_to_pinky_base": 1,
    "right_pinky_base_to_center": 1,
    "left_shoulder_roll": 1.57,
    "left_shoulder_pitch": 2.5,
    "left_elbow": 2.5,
    "left_wrist": 1,
    "left_hand_to_thumb_base": 1,
    "left_thumb_base_to_center": 0,
    "left_hand_to_index_base": 1,
    "left_index_base_to_center": 1,
    "left_hand_to_middle_base": 1,
    "left_middle_base_to_center": 1,
    "left_hand_to_ring_base": 1,
    "left_ring_base_to_center": 1,
    "left_hand_to_pinky_base": 1,
    "left_pinky_base_to_center": 1,
    "right_hip_yaw": 0.7,
    "right_hip_pitch": 1.57,
    "right_knee": 0,
    "right_ankle": 1,
    "left_hip_yaw": 0,
    "left_hip_pitch": 1.57,
    "left_knee": 0,
    "left_ankle": 1
}

# "root" should be accessed with getBodyInfo
LINKS = [
    "chest",
    "head",
    "right_shoulder_intermediate",
    "right_humerus",
    "right_radius",
    "right_hand",
    "right_thumb_base",
    "right_thumb_center",
    "right_index_base",
    "right_index_center",
    "right_middle_base",
    "right_middle_center",
    "right_ring_base",
    "right_ring_center",
    "right_pinky_base",
    "right_pinky_center",
    "left_shoulder_intermediate",
    "left_humerus",
    "left_radius",
    "left_hand",
    "left_thumb_base",
    "left_thumb_center",
    "left_index_base",
    "left_index_center",
    "left_middle_base",
    "left_middle_center",
    "left_ring_base",
    "left_ring_center",
    "left_pinky_base",
    "left_pinky_center",
    "right_hip_intermediate",
    "right_femur",
    "right_tibia",
    "right_foot",
    "left_hip_intermediate",
    "left_femur",
    "left_tibia",
    "left_foot"
]

LINKS_HALF = [
    "chest",
    "head",
    "right_shoulder_intermediate",
    "right_humerus",
    "right_radius",
    "right_hand",
    "left_shoulder_intermediate",
    "left_humerus",
    "left_radius",
    "left_hand",
    "right_hip_intermediate",
    "right_femur",
    "right_tibia",
    "right_foot",
    "left_hip_intermediate",
    "left_femur",
    "left_tibia",
    "left_foot"
]

LINKS_LEGS = [
    "right_hip_intermediate",
    "right_femur",
    "right_tibia",
    "right_foot",
    "left_hip_intermediate",
    "left_femur",
    "left_tibia",
    "left_foot"]

JOINTS = [
    "root_chest",
    "neck",
    "right_shoulder_roll",
    "right_shoulder_pitch",
    "right_elbow",
    "right_wrist",
    "right_hand_to_thumb_base",
    "right_thumb_base_to_center",
    "right_hand_to_index_base",
    "right_index_base_to_center",
    "right_hand_to_middle_base",
    "right_middle_base_to_center",
    "right_hand_to_ring_base",
    "right_ring_base_to_center",
    "right_hand_to_pinky_base",
    "right_pinky_base_to_center",
    "left_shoulder_roll",
    "left_shoulder_pitch",
    "left_elbow",
    "left_wrist",
    "left_hand_to_thumb_base",
    "left_thumb_base_to_center",
    "left_hand_to_index_base",
    "left_index_base_to_center",
    "left_hand_to_middle_base",
    "left_middle_base_to_center",
    "left_hand_to_ring_base",
    "left_ring_base_to_center",
    "left_hand_to_pinky_base",
    "left_pinky_base_to_center",
    "right_hip_yaw",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
    "left_hip_yaw",
    "left_hip_pitch",
    "left_knee",
    "left_ankle"
]

JOINTS_HALF = [
    "root_chest",
    "neck",
    "right_shoulder_roll",
    "right_shoulder_pitch",
    "right_elbow",
    "right_wrist",
    "left_shoulder_roll",
    "left_shoulder_pitch",
    "left_elbow",
    "left_wrist",
    "right_hip_yaw",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
    "left_hip_yaw",
    "left_hip_pitch",
    "left_knee",
    "left_ankle"
]

JOINTS_LEGS = [
    "right_hip_yaw",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
    "left_hip_yaw",
    "left_hip_pitch",
    "left_knee",
    "left_ankle"
]

JOINT_NAMES_POSE_ESTIMATES_ORIGINAL = [
    "Nose",
    "Left Eye Inner",
    "Left Eye",
    "Left Eye Outer",
    "Right Eye Inner",
    "Right Eye",
    "Right Eye Outer",
    "Left Ear",
    "Right Ear",
    "Mouth Left",
    "Mouth Right",
    "Left Shoulder",
    "Right Shoulder",
    "Left Elbow",
    "Right Elbow",
    "Left Wrist",
    "Right Wrist",
    "Left Pinky",
    "Right Pinky",
    "Left Index",
    "Right Index",
    "Left Thumb",
    "Right Thumb",
    "Left Hip",
    "Right Hip",
    "Left Knee",
    "Right Knee",
    "Left Ankle",
    "Right Ankle",
    "Left Heel",
    "Right Heel",
    "Left Foot Index",
    "Right Foot Index"
]

JOINT_NAMES_POSE_ESTIMATES_EQUIVALENT_TO_SIMULATION = [
    "head",
    "Left Eye Inner",
    "Left Eye",
    "Left Eye Outer",
    "Right Eye Inner",
    "Right Eye",
    "Right Eye Outer",
    "Left Ear",
    "Right Ear",
    "Mouth Left",
    "Mouth Right",
    "left_shoulder_pitch",
    "right_shoulder_pitch",
    "left_elbow",
    "right_elbow",
    "left_hand",
    "right_hand",
    "left_pinky_base_to_center",
    "right_pinky_base_to_center",
    "left_index_base_to_center",
    "right_index_base_to_center",
    "left_thumb_base_to_center",
    "right_thumb_base_to_center",
    "left_hip_pitch",
    "right_hip_pitch",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "Left Heel",
    "Right Heel",
    "left_foot",
    "right_foot"
]
