# Copyright (c) 2024 Philipp Spiess
# All rights reserved.

import cv2
import mediapipe as mp
import json
import math
import numpy as np
import argparse
import os

from settings import JOINT_NAMES_POSE_ESTIMATES_EQUIVALENT_TO_SIMULATION

ROBOT_HEIGHT = 1.48  # Foot to Head - For mimicking properly, the demo must start with a straight position
ROBOT_WIDTH = 0.4    # Shoulder to shoulder
EPSILON = 1e-6
DEFAULT_FPS = 30
MIMIC_DIR = "mimic"

def get_video_path(task):
    return os.path.join(MIMIC_DIR, f"{task}.mp4")

def get_json_path(task):
    return os.path.join(MIMIC_DIR, f"{task}.json")

def calculate_angle(a, b, c):
    """ Calculate the angle between three points a, b, and c (where b is the vertex point). """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    # Normalize vectors
    ba_norm = ba / (np.linalg.norm(ba) + EPSILON)
    bc_norm = bc / (np.linalg.norm(bc) + EPSILON)

    cosine_angle = np.dot(ba_norm, bc_norm)
    angle = np.arccos(cosine_angle)

    return angle

def calculate_angular_velocity(current_angle, previous_angle, time_interval=1 / DEFAULT_FPS):
    """ Calculate the angular velocity given current and previous angles and the time interval. """
    return (current_angle - previous_angle) / time_interval


def calculate_linear_velocity(current_position, previous_position, time_interval=1 / DEFAULT_FPS):
    """ Calculate linear velocity based on the current and previous positions. """
    current_position = np.array(current_position)
    previous_position = np.array(previous_position)

    return (current_position - previous_position) / time_interval

def calculate_yaw(left_shoulder, right_shoulder):
    """Calculate yaw (rotation around the vertical axis) from the shoulders."""
    dx = right_shoulder["x"] - left_shoulder["x"]
    dz = right_shoulder["z"] - left_shoulder["z"]
    return math.atan2(dz, dx) + math.pi/2

def calculate_pitch(left_shoulder, right_shoulder, hips):
    """Calculate pitch (tilt forward or backward) using head and hips positions."""
    dx = ( right_shoulder["x"] + left_shoulder["x"] ) / 2 - hips["x"]
    dy = ( right_shoulder["y"] + left_shoulder["y"] ) / 2 - hips["y"]
    return math.atan2(dx, dy)

def calculate_roll(left_shoulder, right_shoulder):
    """Calculate roll (tilt left or right) using the vertical difference of the shoulders."""
    dy = left_shoulder["y"] - right_shoulder["y"]

    # Roll based only on vertical difference
    roll = math.atan2(dy, 1)
    return roll

def store_frames(task):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    cap = cv2.VideoCapture(get_video_path(task))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Frames per second: {fps}")  # USUALLY 30

    results_save = []

    initial_orientation = None
    translation_offset = None
    scaling = None
    previous_angles = {}

    previous_orientation = None
    previous_translation = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform pose estimation
        results = pose.process(rgb_frame)

        if results.pose_landmarks:

            for landmark in results.pose_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # Extract pose landmarks
        if results.pose_landmarks:
            landmarks = dict()
            for landmark, element in zip(results.pose_landmarks.landmark, JOINT_NAMES_POSE_ESTIMATES_EQUIVALENT_TO_SIMULATION):
                landmarks[element] = {
                    'coordinates': {"x": landmark.x,  "y": 1 - landmark.y, "z": landmark.z,},  # Transform for X, Y to be conventionnal
                    'visibility': landmark.visibility
                }

            # Calculate yaw, pitch, and roll
            left_shoulder = landmarks['left_shoulder_pitch']["coordinates"]
            right_shoulder = landmarks['right_shoulder_pitch']["coordinates"]
            head = landmarks['head']["coordinates"]
            left_hip = landmarks['left_hip_pitch']["coordinates"]
            right_hip = landmarks['right_hip_pitch']["coordinates"]
            hips_center = {
                "x": (left_hip["x"] + right_hip["x"]) / 2,
                "y": (left_hip["y"] + right_hip["y"]) / 2,
                "z": (left_hip["z"] + right_hip["z"]) / 2
            }

            yaw = calculate_yaw(left_shoulder, right_shoulder)
            pitch = calculate_pitch(left_shoulder, right_shoulder, hips_center)
            roll = calculate_roll(left_shoulder, right_shoulder)

            # Store orientation as yaw, pitch, roll
            landmarks["orientation"] = {"yaw": yaw, "pitch": pitch, "roll": roll}

            orientation = [roll, pitch, yaw]

            if initial_orientation is None:
                initial_orientation = orientation

                # Get the initial foot coordinates (e.g., left foot)
                left_foot, right_foot = landmarks['left_foot']["coordinates"], landmarks['right_foot']["coordinates"]

                # scaling
                head = landmarks['head']["coordinates"]
                scaling = ROBOT_HEIGHT / (head["y"] - np.mean([left_foot["y"], right_foot["y"]]))
                scaling_depth = ROBOT_WIDTH /  np.abs((right_shoulder["z"] - left_shoulder["z"]))
                print(f"Scaling is: {scaling} for height and {scaling_depth} for depth")

            if previous_orientation is not None:
                landmarks["angular_rotation_velocity"] = {
                    "roll": calculate_angular_velocity(orientation[0], previous_orientation[0], 1 / fps),
                    "pitch": calculate_angular_velocity(orientation[1], previous_orientation[1], 1 / fps),
                    "yaw": calculate_angular_velocity(orientation[2], previous_orientation[2], 1 / fps)
                }
            else:
                landmarks["angular_rotation_velocity"] = {"roll": 0, "pitch": 0, "yaw": 0}

            previous_orientation = orientation

            # Linear translation of body (center of mass, using hips or feet)
            center_of_body = [scaling *  (left_hip["x"] + right_hip["x"]) / 2,
                              scaling * (left_hip["y"] + right_hip["y"]) / 2,
                              scaling_depth * (left_hip["z"] + right_hip["z"]) / 2]


            # Calculate linear translation velocity
            if previous_translation is not None:
                linear_translation_velocity = calculate_linear_velocity(center_of_body, previous_translation, 1 / fps)
                landmarks["linear_translation_velocity"] = {"x": linear_translation_velocity[0],
                                                            "y": linear_translation_velocity[1],
                                                            "z": linear_translation_velocity[2]}
            else:
                landmarks["linear_translation_velocity"] = {"x": 0, "y": 0, "z": 0}

            previous_translation = center_of_body

            for element in JOINT_NAMES_POSE_ESTIMATES_EQUIVALENT_TO_SIMULATION:
                x, y, z = landmarks[element]['coordinates']["x"], landmarks[element]['coordinates']["y"], landmarks[element]['coordinates']["z"]

                # scaling
                x, y, z  = scaling * x, scaling * y, scaling_depth * z

                landmarks[element]['coordinates']["x"], landmarks[element]['coordinates']["y"], landmarks[element]['coordinates']["z"] = x, z, y # Invest y and z here to match pybullet

                # Calculate joint angles
            joint_combinations = [
                ('left_shoulder_pitch', 'left_elbow', 'left_hand', 'left_elbow'),
                ('right_shoulder_pitch', 'right_elbow', 'right_hand', 'right_elbow'),
                ('left_hip_pitch', 'left_knee', 'left_ankle', 'left_knee'),
                ('right_hip_pitch', 'right_knee', 'right_ankle', 'right_knee'),
                ('left_shoulder_pitch', 'left_hip_pitch', 'left_knee', 'left_hip_pitch'),
                ('right_shoulder_pitch', 'right_hip_pitch', 'right_knee', 'right_hip_pitch'),
                ('left_knee', 'Left Heel', 'left_foot', 'left_ankle'),
                ('right_knee', 'Right Heel', 'right_foot', 'right_ankle'),
                ('left_hip_pitch', 'left_shoulder_pitch', 'left_elbow', 'left_shoulder_pitch'),
                ('right_hip_pitch', 'right_shoulder_pitch', 'left_elbow', 'right_shoulder_pitch'),
            ]

            for a, b, c, ele in joint_combinations:
                angle = calculate_angle(
                    [landmarks[a]['coordinates']['x'], landmarks[a]['coordinates']['y'],
                     landmarks[a]['coordinates']['z']],
                    [landmarks[b]['coordinates']['x'], landmarks[a]['coordinates']['y'],
                     landmarks[b]['coordinates']['z']],
                    [landmarks[c]['coordinates']['x'], landmarks[a]['coordinates']['y'],
                     landmarks[c]['coordinates']['z']]
                )

                # Adjust angles based on specific joint requirements (hip, knee, elbow, ankle)
                if "knee" in ele:
                    angle = angle - 3.14
                    landmarks[ele]["angle"] = angle
                if "elbow" in ele:
                    angle = 3.14 - angle
                    landmarks[ele]["angle"] = angle
                if "ankle" in ele:
                    angle = 1.8 - angle
                    landmarks[ele]["angle"] = angle
                if "hip" in ele:
                    angle = 3.14 - angle
                    landmarks[ele]["angle"] = angle
                if "shoulder" in ele:
                    landmarks[ele]["angle"] = angle

                # Calculate angular velocity for joints
                if ele in previous_angles:
                    angular_velocity = calculate_angular_velocity(angle, previous_angles[ele])
                    landmarks[ele]["angle_velocity"] = angular_velocity
                else:
                    landmarks[ele]["angle_velocity"] = 0  # Initial frame has no angular velocity

                previous_angles[ele] = angle

            results_save.append(landmarks)

            # Display the frame with landmarks
        cv2.imshow('Pose Estimation', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("Number of frames: ", len(results_save))

    data = {"num_frames": len(results_save), "frames": results_save}

    # Save the results in JSON format
    with open(f'mimic/{task}.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)


def load_mimic(task):
    with open(get_json_path(task), 'r') as json_file:
        data_loaded = json.load(json_file)
    return data_loaded


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--TASK", type=str)
    args = parser.parse_args()
    TASK = args.TASK

    store_frames(TASK)