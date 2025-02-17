# Copyright (c) 2024 Philipp Spiess
# All rights reserved.

import pybullet_utils.bullet_client as bc
import pybullet as p
import time
import pybullet_data
import numpy as np
import json
import random
import math
import os
import argparse

random.seed(42)
np.random.seed(42)

from mimic import load_mimic
from settings import *
from helpers import timeit, create_random_terrain

class SimulationEnv:

    def __init__(self, render=False, robot_type="half", task="standing", print_reward=False):
        # robot: full, half, legs
        # task: standing, walking, walking_low_energy, etc.

        self.PRINT_REWARD = print_reward
        self.state = None
        self.reward = None
        self.FTP = None
        self.done = False
        self.robot_type = robot_type
        self.render = render
        self.robot_id = None
        self.upright = None
        self.links_data = dict()
        self.joints_data = dict()
        self.vision_object = ""
        self.task = task
        self.task_state = [1 if self.task == t else 0 for t in TASKS]
        self.frame_counter = 0
        self.start_frame = None
        self.end_frame = None
        self.mimic_frames = [None]
        self.is_mimicking = self.task in MIMICKING_TASKS
        if self.is_mimicking:
            self.mimic_frames = load_mimic(self.task)["frames"]

        self.NOISE = NOISE
        self.NOISE_ANGLE = NOISE_ANGLE_STANDING_UP if self.task == "standing_up" else 0.1
        self.robust = False
        if "robust" in task:
            self.robust = True

        if self.robot_type == "half":
            self.LINKS = LINKS_HALF
            self.HUMANOID_FILE = "humanoid_half.urdf"
            self.JOINTS = JOINTS_HALF
            self.dim_action = len(self.JOINTS)
        elif self.robot_type == "legs":
            self.LINKS = LINKS_LEGS
            self.HUMANOID_FILE = "humanoid_lower.urdf"
            self.JOINTS = JOINTS_LEGS
            self.dim_action = len(self.JOINTS)
        else:
            self.LINKS = LINKS
            self.HUMANOID_FILE = "humanoid.urdf"
            self.JOINTS = JOINTS
            self.dim_action = len(self.JOINTS)

        # Joints (angle, velocity, torques applied) - root orientation - one-hot encoded task - mimic phase variable
        self.dim_state = len(self.JOINTS) * 3 + 3 + len(TASKS) + 1

        self.MAX_TORQUES = [value * FORCE_MULTIPLIER for key, value in MAX_TORQUES.items() if key in self.JOINTS]
        self.MAX_POSITIONS = [value for key, value in MAX_POSITIONS.items() if key in self.JOINTS]
        self.MIN_POSITIONS = [value for key, value in MIN_POSITIONS.items() if key in self.JOINTS]

        self.action = np.zeros(self.dim_action)
        self.action_transformed = self.transform_action(self.action)

        # Start PyBullet
        if not self.render:
            self.p = bc.BulletClient(connection_mode=p.DIRECT)
            self.sleep_time = 0.0
        else:
            self.p = bc.BulletClient(connection_mode=p.GUI)
            self.sleep_time = WAIT_TIME

        # Load URDFs
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())

        planeId =self.p.loadURDF("plane.urdf")
        # planeId = create_random_terrain(self.p)

        duck_position = [DUCK_X_POS, 0, 1.1]
        duck_id = p.loadURDF("duck_vhacd.urdf", basePosition=duck_position)
        # planeId =self.p.loadURDF(PATH + "pybullet_data/plane.urdf")

        self.p.changeDynamics(planeId, -1, lateralFriction= FRICTION * (1 + random.uniform(- self.NOISE, NOISE)) )
        self.p.setAdditionalSearchPath(PATH)

        # Simulation settings
        self.p.setGravity(0, 0, GRAVITY)
        self.p.setTimeStep(1.0 / STEP_FREQUENCY)

        # Set initial camera parameters
        camera_distance = 1.0
        camera_yaw = 50
        camera_pitch = -35
        camera_target_position = [0, 0, 0]
        new_camera_distance = camera_distance * 3
        self.p.resetDebugVisualizerCamera(cameraDistance=new_camera_distance, cameraYaw=camera_yaw, cameraPitch=camera_pitch,
                                     cameraTargetPosition=camera_target_position)

        cube_collision = self.p.createCollisionShape(self.p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.5])
        cube_visual = self.p.createVisualShape(self.p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.5],
                                               rgbaColor=[0.5, 0.5, 0.5, 1])
        cube_id = self.p.createMultiBody(baseMass=0, baseCollisionShapeIndex=cube_collision,
                                         baseVisualShapeIndex=cube_visual, basePosition=[DUCK_X_POS, 0, 0.5])
        cube_back_id = self.p.createMultiBody(baseMass=0, baseCollisionShapeIndex=cube_collision,
                                         baseVisualShapeIndex=cube_visual, basePosition=[-DUCK_X_POS, 0, 0.5])

        self.tpv_camera_pos = [1, -1, 1]
        self.tpv_camera_target = [0, 0, 0.5]

        self.start_pos = [0, 0, 0.97]

    def get_position_joints(self, robot_id, joint_index):

        joint_info = self.p.getJointInfo(robot_id, joint_index)
        parent_link_index = joint_info[16]

        # Get the parent link state
        if parent_link_index == -1:
            # This is a base joint
            parent_state = self.p.getBasePositionAndOrientation(robot_id)
            parent_pos, parent_orn = parent_state
        else:
            parent_state = self.p.getLinkState(robot_id, parent_link_index)
            parent_pos, parent_orn = parent_state[0], parent_state[1]

        # Get the joint position in parent link's frame
        joint_pos_parent_frame = joint_info[14]
        # Transform joint position to world frame
        joint_pos_world_frame = self.p.multiplyTransforms(parent_pos, parent_orn, joint_pos_parent_frame, [0, 0, 0, 1])[0]

        return joint_pos_world_frame

    def transform_action(self, values):

        def project_value(x):
            return 1.25 * x + 0.25 if x <= -0.2 else 1.25 * x - 0.25

        if CONTROL == "torque":

            if BLOCKING:
                values = [0 if -0.2 < x < 0.2 else project_value(x) for x in values]  # Force middle values to be a rigid joint : 0

            values = list(values)
            result = [value * effort for value, effort in zip(values, self.MAX_TORQUES)]

        elif CONTROL == "position":

            result = []
            for j, v in enumerate(values):
                position = self.MIN_POSITIONS[j] + ((v + 1) / 2) * (self.MAX_POSITIONS[j] - self.MIN_POSITIONS[j])
                result.append(position)

        elif CONTROL == "velocity":

            result = np.array(values) * MAX_VELOCITY

        return result

    def initialize_angles(self, robot, init_target_frames=None, start_pos_robot=None):

        # Set specific angles with random perturbation for each joint
        if self.task not in MIMICKING_TASKS:

            base_angles = [0] * len(self.JOINTS)
            for joint_index in range(len(self.JOINTS)):

                joint_info =self.p.getJointInfo(robot, joint_index)
                joint_name = joint_info[1].decode("utf-8")

                if joint_name in self.JOINTS:

                    factor = (self.MAX_POSITIONS[joint_index] - self.MIN_POSITIONS[joint_index])
                    perturbation = random.uniform(-factor * self.NOISE_ANGLE / 2, factor * self.NOISE_ANGLE / 2)
                    target_angle = base_angles[joint_index] + perturbation
                    target_angle = max(self.MIN_POSITIONS[joint_index], target_angle)
                    target_angle = min(self.MAX_POSITIONS[joint_index], target_angle)

                    # Directly set the joint state
                    if joint_index == self.JOINTS.index("right_knee") or joint_index ==self.JOINTS.index("left_knee"):
                       self.p.resetJointState(robot, joint_index, target_angle - random.uniform(0, factor * self.NOISE_ANGLE)) # bent knees
                    elif joint_index == self.JOINTS.index("right_hip_pitch") or joint_index ==self.JOINTS.index("left_hip_pitch"):
                       self.p.resetJointState(robot, joint_index, target_angle + random.uniform(0, factor * self.NOISE_ANGLE)) # high unlocked
                    else:
                       self.p.resetJointState(robot, joint_index, target_angle)

                    # Set velocity control to zero
                    self.p.setJointMotorControl2(robot, joint_index,self.p.VELOCITY_CONTROL, force=0)

        else:

            # Init the angles available as ref
            for element in JOINT_NAMES_POSE_ESTIMATES_EQUIVALENT_TO_SIMULATION:

                if element in self.JOINTS and element in init_target_frames:

                    if "angle" in init_target_frames[element]:

                        self.p.resetJointState(robot, self.JOINTS.index(element), init_target_frames[element]["angle"], init_target_frames[element]["angle_velocity"])

            # To do - rotation of the entire body
            vx, vy, vz = init_target_frames["linear_translation_velocity"]["x"], init_target_frames["linear_translation_velocity"]["y"], init_target_frames["linear_translation_velocity"]["z"]
            wx, wy, wz = init_target_frames["angular_rotation_velocity"]["roll"], init_target_frames["angular_rotation_velocity"]["pitch"], init_target_frames["angular_rotation_velocity"]["yaw"]
            self.p.resetBaseVelocity(robot, linearVelocity=[vx, vy, vz], angularVelocity=[wx, wy, wz])

    def get_upright_pos(self, robot):

        pos, ori =self.p.getBasePositionAndOrientation(robot)
        roll, pitch, yaw = self.p.getEulerFromQuaternion(ori)
        upright = math.cos(pitch) * math.sin(roll)
        rotation = math.cos(yaw)

        return upright, rotation

    def get_robot_position_angle(self, robot):

        pos, ori =self.p.getBasePositionAndOrientation(robot)
        roll, pitch, yaw = self.p.getEulerFromQuaternion(ori)

        return pos[0], pos[1], pos[2], roll, pitch, yaw

    def get_distance_to_object(self, robot):

        redObjectPos, redObjectOrn =self.p.getBasePositionAndOrientation(self.red_objectId)
        x_redObjectPos = redObjectPos[0]
        y_redObjectPos = redObjectPos[1]
        robotPos, robotOrn =self.p.getBasePositionAndOrientation(robot)
        x_robot = robotPos[0]
        y_robot = robotPos[1]
        euclidean_distance = math.sqrt((x_redObjectPos - x_robot) ** 2 + (y_redObjectPos - y_robot) ** 2)

        return euclidean_distance

    def get_foot_distance_to_object(self, robot):

        redObjectPos, redObjectOrn =self.p.getBasePositionAndOrientation(self.red_objectId)
        x_redObjectPos = redObjectPos[0]
        y_redObjectPos = redObjectPos[1]

        foot_pos, foot_ori =self.p.getLinkState(robot, self.LINKS.index("right_foot"))[:2]
        x_right_foot = foot_pos[0]
        y_right_foot = foot_pos[1]
        foot_pos, foot_ori =self.p.getLinkState(robot, self.LINKS.index("left_foot"))[:2]
        x_left_foot = foot_pos[0]
        y_left_foot = foot_pos[1]

        euclidean_distance_right = math.sqrt((x_redObjectPos - x_right_foot) ** 2 + (y_redObjectPos - y_right_foot) ** 2)
        euclidean_distance_left = math.sqrt((x_redObjectPos - x_left_foot) ** 2 + (y_redObjectPos - y_left_foot) ** 2)

        return euclidean_distance_left, euclidean_distance_right

    def get_first_person_view(self, robot):

        # Get the position and orientation of the head joint for FPV camera
        head_pos, head_ori =self.p.getLinkState(robot, self.LINKS.index("head"))[:2]
        # Compute the forward vector for the head orientation to simulate human-like forward gaze
        forward_vector = [1, 0, 0]  # Assuming the head's forward direction is along the robot's x-axis
        head_forward_pos =self.p.multiplyTransforms(head_pos, head_ori, forward_vector, [0, 0, 0, 1])[0]

        # First-person view camera
        fpv_view_matrix =self.p.computeViewMatrix(cameraEyePosition=head_pos,
                                              cameraTargetPosition=head_forward_pos,
                                              cameraUpVector=[0, 0, 1])
        fpv_projection_matrix = self.p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.1, farVal=100.0)
        width, height, fpv_rgbImg, _, _ =self.p.getCameraImage(width=224, height=224, viewMatrix=fpv_view_matrix, projectionMatrix=fpv_projection_matrix)
        # Process and print FPV camera pixel data
        self.FPV = np.reshape(np.array(fpv_rgbImg), (height, width, 4))

        np.save(VISION_PATH+'/FPV.npy', self.FPV)
        with open(VISION_PATH+"/result.json", 'r') as f:
            data = json.load(f)
        coordinates = np.array(data['coordinates'])
        labels = data['labels']
        depths = data['depths']
        values = np.array(data['values'])
        self.vision_object = labels[0]
        box = np.array(coordinates[0]) * 2 - 1
        x_center = (box[0] + box[2]) / 2
        y_center = (box[1] + box[3]) / 2
        z = depths[0]

        return self.vision_object, x_center, y_center, z

    def get_reward(self, robot, upright, joints_data, links_data, previous_upright, previous_joints_data, previous_links_data):

        done = 0
        reward = 0

        distance_forward_left_foot = links_data["left_foot"]["x"] - previous_links_data["left_foot"]["x"]
        distance_forward_right_foot = links_data["right_foot"]["x"] - previous_links_data["right_foot"]["x"]
        distance_forward_root = links_data["root"]["x"] - previous_links_data["root"]["x"]

        energy = np.sum([np.abs(joints_data[j]["torque"]) for j in self.JOINTS])
        non_smoothness = np.sum([np.abs(previous_joints_data[j]["torque"] - joints_data[j]["torque"]) for j in self.JOINTS])

        reward_stable = 0                 # upright ** 2 + 50 * (upright ** 2 - previous_upright ** 2)
        penalty_not_smooth = 0
        penalty_energy = 0
        penalty_fall = -50
        penalty_knees_not_bend = 0        #- np.abs(0.5 + right_knee_angle_radian) * 2 - np.abs( 0.5 + left_knee_angle_radian) * 2
        reward_walking = 0
        reward_height = 0
        penalty_ankles = 0                #- np.abs(-0.3 + right_ankle_angle_radian) * 2 - np.abs(-0.3 + left_ankle_angle_radian) * 2
        penalty_floor = 0

        # SPECIFIC REWARDS
        reward_stable = upright ** 2 + 50 * (upright ** 2 - previous_upright ** 2)

        if "walking" in self.task:
            reward_walking += 4*min(1, SIMULATION_FREQUENCY * distance_forward_left_foot) if distance_forward_right_foot <= 1 / SIMULATION_FREQUENCY / 5 else 0
            reward_walking += 4*min(1, SIMULATION_FREQUENCY * distance_forward_right_foot) if distance_forward_left_foot <= 1 / SIMULATION_FREQUENCY / 5 else 0
            reward_walking += 4*min(1, SIMULATION_FREQUENCY * distance_forward_root)

        if "low_energy" in self.task:
            penalty_energy -= energy / ENERGY_FACTOR
            penalty_not_smooth -= non_smoothness / SMOOTH_FACTOR

        if self.task == "standing_up":
            if self.robot_type == "legs":
                reward_height += links_data["root"]["z"]
            else:
                reward_height += links_data["head"]["z"]

        # TERMINATION STATES
        if self.robot_type == "legs":
            if joints_data["right_knee"]["z"] < 0.1 or joints_data["left_knee"]["z"]< 0.1 or links_data["right_hip_intermediate"]["z"] < 0.3 or links_data["left_hip_intermediate"]["z"]< 0.3:
                if self.task == "standing_up":
                    penalty_floor = -1
                else:
                    reward, done = penalty_fall, 1
        else:
            if links_data["head"]["z"] < 0.5 or joints_data["right_knee"]["z"] < 0.1 or joints_data["left_knee"]["z"]< 0.1 or links_data["right_hand"]["z"] < 0.15 or links_data["left_hand"]["z"] < 0.15:
                if self.task == "standing_up":
                    penalty_floor = -1
                else:
                    reward, done = penalty_fall, 1

        if done==0:

            reward += reward_stable + reward_height + penalty_not_smooth + penalty_energy + penalty_floor
            reward += penalty_knees_not_bend + reward_walking + penalty_ankles

            if self.PRINT_REWARD:
                print("")
                print("Rewards: ")
                print("reward_stable ", reward_stable)
                print("reward_height ", reward_height)
                print("penalty_not_smooth ", penalty_not_smooth)
                print("penalty_energy ", penalty_energy)
                print("penalty_knees_not_bend ", penalty_knees_not_bend)
                print("reward_walking ", reward_walking)
                print("penalty_ankles ", penalty_ankles)
                print("penalty_floor ", penalty_floor)
        else:
            if VIDEO_RECORDING:
                video_writer.release()

        return reward, done

    def get_reward_mimic(self, robot, frame_counter, end_frame, joints_data, links_data, offset_xy = [0,0]):
        # DeepMimic implementation

        if self.task not in MIMICKING_TASKS:
            return 0
        if frame_counter >= end_frame:
            return None

        frames = self.mimic_frames[int(frame_counter)]

        link_indices = {element: self.LINKS.index(element) for element in
                        JOINT_NAMES_POSE_ESTIMATES_EQUIVALENT_TO_SIMULATION if element in self.LINKS}
        joint_indices = {element: self.JOINTS.index(element) for element in
                         JOINT_NAMES_POSE_ESTIMATES_EQUIVALENT_TO_SIMULATION if element in self.JOINTS}

        # joints orientations - 0.65 - joints angle velocity - 0.1 - end effector xyz positions - 0.15 - Center of mass - 0.1 (For now unused)
        orientation_penalty = []
        angle_velocity_penalty = []
        end_effector_penalty = []
        center_mass_penalty = []

        for element in JOINT_NAMES_POSE_ESTIMATES_EQUIVALENT_TO_SIMULATION:
            if element in joint_indices and element in frames:
                if "angle" in frames[element]:
                    ref_angle = frames[element]["angle"]
                    ref_angle_velocity = frames[element]["angle_velocity"]

                    angle, angle_velocity = joints_data[element]["angle"], joints_data[element]["angle_velocity"]

                    diff_angle = ( ref_angle - angle )     # / self.MAX_POSITIONS[element] - self.MIN_POSITIONS[element]

                    orientation_penalty.append(np.linalg.norm(diff_angle))
                    angle_velocity_penalty.append(np.linalg.norm(ref_angle_velocity - angle_velocity))

                    if self.PRINT_REWARD:
                        print(element, angle, ref_angle)

            if element in link_indices and element in ["right_foot", "left_foot", "right_hand", "left_hand"]:
                pos = [links_data[element]["x"], links_data[element]["y"], links_data[element]["z"]]
                pos[0] -= offset_xy[0]
                pos[1] -= offset_xy[1]
                ref = frames[element]["coordinates"]
                ref_pos = np.array([ref["x"], ref["y"], ref["z"]])

                end_effector_penalty.append(np.linalg.norm(ref_pos - pos))

            if element in joint_indices and element in ["right_hip_pitch", "left_hip_pitch"]:
                pos = [joints_data[element]["x"], joints_data[element]["y"], joints_data[element]["z"]]
                pos[0] -= offset_xy[0]
                pos[1] -= offset_xy[1]
                ref = frames[element]["coordinates"]
                ref_pos = np.array([ref["x"], ref["y"], ref["z"]])

                center_mass_penalty.append(np.linalg.norm(ref_pos - pos))

        orientation_penalty_res = (
                MIMIC_CONSTANTS["ORIENTATION_WEIGHT"]
                * np.exp(MIMIC_CONSTANTS["ORIENTATION_EXPONENT"] * np.linalg.norm(orientation_penalty))
                * MIMIC_CONSTANTS["ORIENTATION_SCALING"]
        )

        angle_velocity_penalty_res = (
                MIMIC_CONSTANTS["ANGLE_VELOCITY_WEIGHT"]
                * np.exp(MIMIC_CONSTANTS["ANGLE_VELOCITY_EXPONENT"] * np.linalg.norm(angle_velocity_penalty))
                * MIMIC_CONSTANTS["ANGLE_VELOCITY_SCALING"]
        )

        end_effector_penalty_res = (
                MIMIC_CONSTANTS["END_EFFECTOR_WEIGHT"]
                * np.exp(MIMIC_CONSTANTS["END_EFFECTOR_EXPONENT"] * np.linalg.norm(end_effector_penalty))
                * MIMIC_CONSTANTS["END_EFFECTOR_SCALING"]
        )

        center_mass_penalty_res = (
                MIMIC_CONSTANTS["CENTER_MASS_WEIGHT"]
                * np.exp(MIMIC_CONSTANTS["CENTER_MASS_EXPONENT"] * np.linalg.norm(center_mass_penalty))
                * MIMIC_CONSTANTS["CENTER_MASS_SCALING"]
        )

        if self.PRINT_REWARD:
            print(f"Rewards mimic: frame {frame_counter}")
            print("Orientation: ", orientation_penalty_res)
            print("Angle velocity: ", angle_velocity_penalty_res)
            print("Center mass xyz: ", end_effector_penalty_res)
            print("End effector xyz: ", center_mass_penalty_res)

        return orientation_penalty_res + angle_velocity_penalty_res + end_effector_penalty_res + center_mass_penalty_res

    def get_state(self, robot, idx=None):

        action = self.action if idx is None else self.action[idx]
        next_state = []
        for j in range(len(self.JOINTS)):

            state_noise = np.random.uniform(1 - self.NOISE, 1 + self.NOISE) if self.robust else 1

            joint_info =self.p.getJointInfo(robot, j)
            joint_name = joint_info[1].decode("utf-8")
            joint_state = self.p.getJointState(robot, j)
            position = 2 * ((joint_state[0] - self.MIN_POSITIONS[j]) / (self.MAX_POSITIONS[j] - self.MIN_POSITIONS[j])) - 1 #  ( joint_state[0] - self.MIN_POSITIONS[j] ) / (self.MAX_POSITIONS[j] - self.MIN_POSITIONS[j])     # normalised [0, 1]
            velocity = joint_state[1]                                                                                   # radian/s [-1, 1]
            joint_action = action[j] # joint_state[3] # torque + external force                                         # between [-1, 1]
            next_state += [position * state_noise, velocity * state_noise, joint_action * state_noise]
            # The reaction force is not used : joint_state[2]

        state_noise = np.random.uniform(1 - self.NOISE, 1 + self.NOISE) if self.robust else 1

        _,_,_, roll, pitch, yaw = self.get_robot_position_angle(robot)
        next_state += [roll * state_noise, pitch * state_noise, yaw * state_noise] # yaw will be 0 for the real robot
        next_state += self.task_state

        return next_state

    def set_target_frames(self):

        if self.is_mimicking:
            start_frame = random.randint(0, len(self.mimic_frames)-2)
            end_frame = random.randint(start_frame+1, len(self.mimic_frames)-1)  # RSI and early_termination
        else:
            start_frame, end_frame = 0, -1

        # return start_frame, end_frame
        return 0, len(self.mimic_frames)-2                                       # NO RSI FOR NOW!

    def get_links_joints_data(self, robot):

        upright, rotation = self.get_upright_pos(robot)
        joints_data, links_data = dict(), dict()

        for joint_index, ele in enumerate(self.JOINTS):

            state_noise = np.random.uniform(-self.NOISE, self.NOISE) if self.robust else 0

            x,y,z = self.get_position_joints(robot, joint_index)

            joint_state = self.p.getJointState(robot, joint_index)
            # real angle here - (joint_state[0] - self.MIN_POSITIONS[joint_index] ) / (self.MAX_POSITIONS[joint_index] - self.MIN_POSITIONS[joint_index])   # to normalise to [0, 1]
            angle = joint_state[0]
            angle_velocity = joint_state[1]
            torque = joint_state[3]

            if ele not in joints_data:
                joints_data[ele] = {}

            joints_data[ele]["x"] = x
            joints_data[ele]["y"] = y
            joints_data[ele]["z"] = z
            joints_data[ele]["angle"] = angle * (1+ state_noise)
            joints_data[ele]["angle_velocity"] = angle_velocity * (1+ state_noise)
            joints_data[ele]["torque"] = torque * (1+ state_noise)

        for link_index, ele in enumerate(self.LINKS):

            state_noise = np.random.uniform(-self.NOISE, self.NOISE) if self.robust else 0

            pos, ori = self.p.getLinkState(robot,link_index)[:2] # Adapt ORI to quaternions

            if ele not in links_data:
                links_data[ele] = {}

            links_data[ele]["x"] = pos[0]
            links_data[ele]["y"] = pos[1]
            links_data[ele]["z"] = pos[2]
            links_data[ele]["roll"] = ori[0] * (1+ state_noise)
            links_data[ele]["pitch"] = ori[1] * (1+ state_noise)
            links_data[ele]["yaw"] = ori[2] * (1+ state_noise)

        # add root info
        if "root" not in links_data:
            links_data["root"] = {}

        state_noise = np.random.uniform(-self.NOISE, self.NOISE) if self.robust else 0

        x, y, z,  roll, pitch, yaw = self.get_robot_position_angle(robot)
        links_data["root"]["x"] = x
        links_data["root"]["y"] = y
        links_data["root"]["z"] = z
        links_data["root"]["roll"] = roll * (1+ state_noise)
        links_data["root"]["pitch"] = pitch * (1+ state_noise)
        links_data["root"]["yaw"] = yaw * (1+ state_noise)

        return upright, joints_data, links_data

    def generate_robot(self, frame_counter=0, robot=None):

        if self.is_mimicking:
            init_target_frames = self.mimic_frames[int(frame_counter)]
            start_pos_robot = np.array(self.start_pos) + np.array([init_target_frames['head']['coordinates']["x"] , init_target_frames['head']['coordinates']["y"], 0]) # Translate on the xy plane
            init_orientation_x, init_orientation_y, init_orientation_z = init_target_frames["orientation"]["roll"], init_target_frames["orientation"]["pitch"], init_target_frames["orientation"]["yaw"]
        else:
            init_target_frames = None
            start_pos_robot = self.start_pos
            init_orientation_x, init_orientation_y, init_orientation_z = 0, 0 ,0
        perturbation = np.random.uniform(-0.01, 0.01, size=3)
        self.upright_orientation =self.p.getQuaternionFromEuler([1.57+perturbation[0]+init_orientation_x, perturbation[1]+init_orientation_y, perturbation[2]+init_orientation_z])
        if robot is None:
            robot = self.p.loadURDF(self.HUMANOID_FILE, start_pos_robot, self.upright_orientation, useFixedBase=False)
        else:
            self.p.removeBody(robot)
            robot =self.p.loadURDF(self.HUMANOID_FILE, start_pos_robot, self.upright_orientation, useFixedBase=False)
            # self.p.resetBasePositionAndOrientation(robot, self.start_pos, self.upright_orientation)

        self.initialize_angles(robot, init_target_frames, np.array([self.start_pos[0], self.start_pos[1], 0]))

        if self.robust:

            for joint_index in range(len(self.JOINTS)):
                # Get the current mass and inertia values of the link
                dynamics_info = self.p.getDynamicsInfo(robot, joint_index)
                current_mass = dynamics_info[0]
                current_inertia = dynamics_info[2]

                # Generate random mass and inertia
                new_mass = current_mass * random.uniform(1-self.NOISE, 1+self.NOISE)
                new_inertia = [inertia * random.uniform(1-self.NOISE, 1+self.NOISE) for inertia in current_inertia]

                # Set the new mass and inertia to the link
                self.p.changeDynamics(robot, joint_index, mass=new_mass, localInertiaDiagonal=new_inertia)

        self.p.changeDynamics(robot, self.LINKS.index("right_foot"), lateralFriction=FRICTION * (1 + random.uniform(- self.NOISE, NOISE)))
        self.p.changeDynamics(robot, self.LINKS.index("left_foot"), lateralFriction=FRICTION * (1 + random.uniform(- self.NOISE, NOISE)))

        upright, joints_data, links_data = self.get_links_joints_data(robot)

        return robot, upright, joints_data, links_data

    def apply_actions(self, robot, action_transformed, start_frame, frame_counter):

        for j, action_value in enumerate(action_transformed):
            joint_info =self.p.getJointInfo(robot, j)
            joint_name = joint_info[1].decode("utf-8")

            current_action_value = action_value
            current_max_torque = MAX_TORQUES[joint_name] * FORCE_MULTIPLIER

            velocity_gain = 1
            if self.robust:
                current_action_value = current_action_value * np.random.uniform(1-self.NOISE, 1)
                current_max_torque = current_max_torque * np.random.uniform(1-self.NOISE, 1)
                velocity_gain = velocity_gain * (1 + np.random.uniform(-self.NOISE, self.NOISE))

            if action_value==0 and BLOCKING:
                # Lock the position
                joint_state = self.p.getJointState(robot, j)
                current_position = joint_state[0]
                stall_torque = 25
                self.p.setJointMotorControl2(robot, j, controlMode=self.p.POSITION_CONTROL,
                                             targetPosition=current_position, # positionGain=0, velocityGain=0,
                                             force=stall_torque)
            elif CONTROL == "torque":
                self.p.setJointMotorControl2(bodyUniqueId=robot,
                                        jointIndex=j, # positionGain=0, velocityGain=0,
                                        controlMode=self.p.TORQUE_CONTROL,
                                        force=action_value)
            elif CONTROL == "position":
                self.p.setJointMotorControl2(robot, j, controlMode=self.p.POSITION_CONTROL,
                                             targetPosition=action_value, # positionGain=0, velocityGain=0, # To adjust
                                             force=current_max_torque)
            elif CONTROL == "velocity":
                self.p.setJointMotorControl2(robot, j, self.p.VELOCITY_CONTROL,
                                             targetVelocity=action_value, positionGain=0, velocityGain=velocity_gain, # MUST BE ADJUSTED TO MATCH THE REAL MOTOR
                                             force=current_max_torque)

    def close(self):
       self.p.disconnect()

    def reset(self):

        self.start_frame, self.end_frame = self.set_target_frames()
        self.frame_counter = self.start_frame
        if self.robot_id is None:
            self.robot_id, self.upright, self.joints_data, self.links_data = self.generate_robot(self.frame_counter)
        else:
            _ , self.upright, self.joints_data, self.links_data = self.generate_robot(self.frame_counter, self.robot_id)
        self.state = self.get_state(self.robot_id)
        self.state.append(self.frame_counter / len(self.mimic_frames))
        self.done = False
        self.reward = None
        action = np.zeros(self.dim_action)
        self.action = action
        self.action_transformed = self.transform_action(self.action)

        return self.state, {}

    # Function to apply random force to a random link at a random position
    def apply_random_force(self, robot_id):

        # Randomly select a link (joint)
        link_index = random.randint(0, len(self.JOINTS)-1)

        # Get the link state (position, orientation)
        link_state = self.p.getLinkState(robot_id, link_index)

        # Extract the world position of the center of mass of the link
        link_position = link_state[0]

        # Get the axis-aligned bounding box (AABB) of the link to determine its dimensions
        aabb_min, aabb_max = self.p.getAABB(robot_id, link_index)

        # Calculate the dimensions of the link based on the AABB
        link_dimensions = [aabb_max[i] - aabb_min[i] for i in range(3)]

        # Randomly choose a position offset based on -50% to +50% of the link's dimensions
        position_offset = [random.uniform(-0.5, 0.5) * link_dimensions[i] for i in range(3)]
        random_position = [link_position[i] + position_offset[i] for i in range(3)]

        # Apply a random force in a random direction
        random_force = [random.uniform(-self.NOISE*100, self.NOISE*100) for _ in range(3)]

        # Apply the force to the link at the random position
        self.p.applyExternalForce(objectUniqueId=robot_id,
                                  linkIndex=link_index,
                                  forceObj=random_force,
                                  posObj=random_position,
                                  flags=self.p.WORLD_FRAME)

    def step(self, raw_action):

        current_action_transformed = self.transform_action(raw_action)

        for _ in range(int(STEP_FREQUENCY / SIMULATION_FREQUENCY)):
             self.apply_actions(self.robot_id, current_action_transformed, self.start_frame, self.frame_counter)
             if self.robust:
                 self.apply_random_force(self.robot_id)
             self.p.stepSimulation()
             time.sleep(self.sleep_time)

        if VIDEO_RECORDING:
            width, height = 640*2, 480*2
            _, _, rgba_pixels, _, _ = p.getCameraImage(width, height)
            frame = np.reshape(np.array(rgba_pixels, dtype=np.uint8), (height, width, 4))
            frame = frame[:, :, :3]  # Remove the alpha channel if present
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to OpenCV format
            frame = cv2.resize(frame, (width, height))
            video_writer.write(frame)

        if self.robot_type == "full":
            vision_object, x, y, z = self.get_first_person_view(self.robot_id)

        new_upright, new_joints_data, new_links_data = self.get_links_joints_data(self.robot_id)

        self.state = self.get_state(self.robot_id)
        self.state.append(self.frame_counter / len(self.mimic_frames))

        reward, self.done = self.get_reward(self.robot_id, new_upright, new_joints_data, new_links_data, self.upright, self.joints_data, self.links_data)
        reward_mimic = self.get_reward_mimic(self.robot_id, self.frame_counter, self.end_frame, self.joints_data, self.links_data)
        if reward_mimic is None:
            self.done = 1
            reward_mimic = 0

        self.reward = 0.3 * reward + 0.7 * reward_mimic

        self.action = raw_action
        self.action_transformed = current_action_transformed
        self.upright = new_upright
        self.joints_data = new_joints_data
        self.links_data = new_links_data
        if self.is_mimicking:
            self.frame_counter += FRAMES_FREQUENCY / SIMULATION_FREQUENCY

        return self.state, self.reward, self.done, {}, {}


class SimulationParallelEnv(SimulationEnv):

    def __init__(self, render=False, robot_type="full", task="standing"):

        super().__init__(render=render, robot_type=robot_type, task=task)

        self.state = []
        self.action = []
        self.action_transformed = []
        self.reward = []
        self.done = []
        self.FPV = []
        self.robot_id = []
        self.upright = []
        self.links_data = []
        self.joints_data = []
        self.frame_counter = []
        self.init_offset_xy = []
        self.total_robots = 0
        self.start_frame = []
        self.end_frame = []

    def reset(self, idx=None):

        if idx is None:
            idx = self.total_robots
            start_frame, end_frame = self.set_target_frames()
            self.end_frame.append(end_frame)
            self.start_frame.append(start_frame)
            self.frame_counter.append(start_frame)
            self.total_robots += 1
            self.start_pos = [3*int(idx%16)-20, 3*int(idx/16)-20, 0.97]
            robot, upright, joints_data, links_data = self.generate_robot(self.frame_counter[idx])
            self.upright.append(upright)
            self.joints_data.append(joints_data)
            self.links_data.append(links_data)
            self.robot_id.append(robot)
            action =np.zeros(self.dim_action)
            self.action.append(action)
            self.action_transformed.append(self.transform_action(action))
            self.reward.append(0)
            self.done.append(False)
            state = self.get_state(self.robot_id[idx], idx)
            state.append( start_frame / len(self.mimic_frames))
            self.state.append(state)
            self.init_offset_xy.append(self.start_pos[:2])
        else:
            self.start_pos = [3*int(idx%16)-20, 3*int(idx/16)-20, 0.97]
            self.start_frame[idx], self.end_frame[idx] = self.set_target_frames()
            self.frame_counter[idx] = self.start_frame[idx]
            self.robot_id[idx], self.upright[idx],  self.joints_data[idx], self.links_data[idx] = self.generate_robot(self.frame_counter[idx], self.robot_id[idx])
            self.reward[idx] = 0
            action = np.zeros(self.dim_action)
            self.action[idx] = action
            self.action_transformed[idx] = self.transform_action(action)
            self.done[idx] = False
            state = self.get_state(self.robot_id[idx], idx)
            state.append(self.frame_counter[idx] / len(self.mimic_frames))
            self.state[idx] = state

        return state, idx

    def step(self, raw_actions):

        current_action_transformed_list = [self.transform_action(raw_action) for raw_action in raw_actions]

        for _ in range(int(STEP_FREQUENCY / SIMULATION_FREQUENCY)):
            for idx, current_action_transformed in enumerate(current_action_transformed_list):
                self.apply_actions(self.robot_id[idx], current_action_transformed, self.start_frame[idx], self.frame_counter[idx])
                if self.robust:
                    self.apply_random_force(self.robot_id[idx])
            self.p.stepSimulation()
            time.sleep(self.sleep_time)

        for idx, robot_id in enumerate(self.robot_id):
            current_action_transformed = current_action_transformed_list[idx]

            # VISION : NOT FAST ENOUGH FOR PARALLEL

            new_upright, new_joints_data, new_links_data = self.get_links_joints_data(robot_id)

            self.state[idx] = self.get_state(robot_id, idx)
            self.state[idx].append(self.frame_counter[idx] / len(self.mimic_frames))

            reward, self.done[idx] = self.get_reward(robot_id, new_upright, new_joints_data, new_links_data, self.upright[idx], self.joints_data[idx], self.links_data[idx])

            reward_mimic = self.get_reward_mimic(robot_id, self.frame_counter[idx], self.end_frame[idx], self.joints_data[idx], self.links_data[idx], self.init_offset_xy[idx])
            if reward_mimic is None:
                self.done[idx] = 1
                reward_mimic = 0

            self.reward[idx] = 0.3 * reward + 0.7 * reward_mimic

            self.action[idx] = raw_actions[idx]
            self.action_transformed[idx] = current_action_transformed
            self.upright[idx] = new_upright
            self.joints_data[idx] = new_joints_data
            self.links_data[idx] = new_links_data

            if self.is_mimicking:
                self.frame_counter[idx] += FRAMES_FREQUENCY / SIMULATION_FREQUENCY

        return self.state, self.reward, self.done, {}, {}


class RandomAgent:

    def __init__(self, dim_action=16):

        self.dim_action = dim_action

    def act(self, state):
        # random_array = np.random.uniform(-1, 1, size=self.dim_action)
        random_array = np.full(self.dim_action, -1)
        array = [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        return random_array


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ROBOT_TYPE", type=str, default=ROBOT_TYPE)
    parser.add_argument("--NUM_BOTS", type=int, default=NUM_BOTS)
    args = parser.parse_args()
    ROBOT_TYPE = args.ROBOT_TYPE
    NUM_BOTS = args.NUM_BOTS

    if not PARALLEL_SIM:

        env = SimulationEnv(render=True, robot_type=ROBOT_TYPE)
        agent = RandomAgent(dim_action=env.dim_action)
        state, _ = env.reset()

        rewards = []
        actions = []
        states = []

        while True:

            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
            rewards.append(reward)
            actions.append(action)
            states.append(state)
            if done:
                state, _ = env.reset()
    else:

        env = SimulationParallelEnv(render=True, robot_type=ROBOT_TYPE)
        agent = RandomAgent(dim_action = env.dim_action)

        states = [[] for _ in range(NUM_BOTS)]
        actions = [[] for _ in range(NUM_BOTS)]
        rewards = [[] for _ in range(NUM_BOTS)]

        for i in range(NUM_BOTS):
            state, idx = env.reset()
            states[idx].append(state)

        while True:

            for idx in range(NUM_BOTS):

                action = agent.act(states[idx])
                actions[idx] = action

            next_states, rewards, dones, _, _ = env.step(actions)
            states = next_states

            for idx in range(NUM_BOTS):
                if dones[idx]:
                    states[idx], _ = env.reset(idx)
