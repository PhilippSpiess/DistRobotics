# Copyright (c) 2024 Philipp Spiess
# All rights reserved.

import RPi.GPIO as GPIO
import time
import random
import math
from mpu6050 import mpu6050
import numpy as np
import board
import busio
from adafruit_pca9685 import PCA9685
from adafruit_tca9548a import TCA9548A
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
from digitalio import Direction
import smbus
from adafruit_motor import servo

from config_simulation import TASKS
from learning import PPOAgent, TRAINING_BATCH

MANUAL_CALIBRATION = False
AUTO_CALIBRATION = True

ROBOT_TYPE = "legs"
TASK = "standing"
TASK_STATE = [1 if TASK == t else 0 for t in TASKS]

SIMPLE_TEST = False
TEST_FLEXION = True
PRODUCTION = False

STATES = []  # NEXT_STATES is STATES Shifted by one
ACTIONS = []
PREDICTIONS = []
REWARDS = []
DONES = []

PRINTING = True

# from learning import agent
MAX_SPEED = 1.6
NUM_MOTORS = 8

MOTORS = ["right_hip_yaw", "left_hip_yaw", "right_hip_pitch", "left_hip_pitch", "right_knee", "left_knee",
          "right_ankle", "left_ankle"]

# According to the trained model
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

ANGLE_OFFSET = [4.799, 4.763, 1.95, 3.77, 5.94, 1.41, 5.63, 2.57]
ANGLE_DIRECTION_ENCODER = [1, 1, 1, -1, -1, 1, -1, 1]

MIN_ANGLES = [-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1]
MAX_ANGLES = [0.1, 0.1, 0.1, 0.1, 0, 0, 0.1, 0.1]

ACTIVE_MOTORS = [1, 1, 1, 1, 1, 1, 1, 1]

if TEST_FLEXION:
    MIN_ANGLES = [0, 0, 0, 0, -0.8, -0.8, 0, 0]
    MAX_ANGLES = [0.1, 0.1, 0.5, 0.5, 0, 0, 0.5, 0.5]

if PRODUCTION:
    MIN_ANGLES = [0.1, 0.7, -0.7, -0.7, -1.5, -1.5, -0.7, -0.7]
    MAX_ANGLES = [-0.7, -0.1, 0.7, 0.7, 0, 0, 0.7, 0.7]

PINS_ENCODER = [17, 27, 22, 5, 6, 13, 19, 26]
PINS_DIRECTION = [18, 23, 24, 25, 12, 16, 20, 21]

FREQUENCY = 30

MPU6050_ADDRESS = 0x68
AS5600_ADDRESS = 0x36

agent = PPOAgent("real", render=False, task=TASK, robot_type=ROBOT_TYPE)

angles = np.full(NUM_MOTORS, 0)
orientation_motors = np.array([1, 1, -1, 1, 1, -1, 1, -1])
goal_speeds = np.full(NUM_MOTORS, 0)
duty_cycles = np.full(NUM_MOTORS, 0)

current_directions = np.full(NUM_MOTORS, 1)

bus = smbus.SMBus(1)

i2c = busio.I2C(board.SCL, board.SDA)

TCA9548A_ADDRESS = 0x70  # Motors encoders lower body
# TCA9548A_ADDRESS_2 = 0x71 # A0                  # Motors encdoers upper body
# TCA9548A_ADDRESS_2 = 0x72 # A0 and A1           # 6 ads1115 - 4 Servo encoders each

tca = TCA9548A(i2c)
# tca2 = TCA9548A(i2c, address=TCA9548A_ADDRESS_2)
# tca3 = TCA9548A(i2c, address=TCA9548A_ADDRESS_2)

mpu = mpu6050(MPU6050_ADDRESS)

# Display the i2c devices on the TCA9548A
for channel in range(8):
    if tca[channel].try_lock():
        print("Channel {}:".format(channel), end="")
        addresses = tca[channel].scan()
        print([hex(address) for address in addresses if address != 0x70])
        tca[channel].unlock()


# Function to select a specific channel on the TCA9548A
def select_tca9548a_channel(channel, address=TCA9548A_ADDRESS):  # or TCA9548A_ADDRESS_2 or TCA9548A_ADDRESS_3
    if channel < 0 or channel > 7:
        raise ValueError("Channel must be between 0 and 7")
    # Write to the TCA9548A to select the channel
    bus.write_byte(address, 1 << channel)


# Function to read raw angle from AS5600
def read_as5600_angle(num_channel):
    select_tca9548a_channel(num_channel)

    # AS5600 register for RAW ANGLE is 0x0C (high byte) and 0x0D (low byte)
    high_byte = bus.read_byte_data(AS5600_ADDRESS, 0x0C)
    low_byte = bus.read_byte_data(AS5600_ADDRESS, 0x0D)

    # Combine the two bytes to form the raw angle
    raw_angle = (high_byte << 8) | low_byte

    angle_in_degrees = (raw_angle / 4096.0) * 360.0

    angle_radian = angle_in_degrees * math.pi / 180

    return angle_radian


# Function to read from an ADS1115 connected through the TCA9548A - To get feedback from all the fingers
def read_ads1115_on_tca_channel(tca_channel):
    # Check if the selected channel is active
    if not tca[tca_channel].try_lock():
        return

    # Initialize ADS1115 on the selected TCA channel
    ads = ADS.ADS1115(tca[tca_channel])

    # Create analog input channels for A0, A1, A2, A3
    chan0 = AnalogIn(ads, ADS.P0)
    chan1 = AnalogIn(ads, ADS.P1)
    chan2 = AnalogIn(ads, ADS.P2)
    chan3 = AnalogIn(ads, ADS.P3)

    # Read and print values from each channel
    print(f"ADS1115 on TCA channel {tca_channel} -")
    print(f"Channel 0: Raw Value: {chan0.value}, Voltage: {chan0.voltage}V")
    print(f"Channel 1: Raw Value: {chan1.value}, Voltage: {chan1.voltage}V")
    print(f"Channel 2: Raw Value: {chan2.value}, Voltage: {chan2.voltage}V")
    print(f"Channel 3: Raw Value: {chan3.value}, Voltage: {chan3.voltage}V")

    # Release the lock for the channel
    tca[tca_channel].unlock()

    return [chan0.value, chan1.value, chan2.value, chan3.value]  #


def get_root_orientation():
    # Get accelerometer and gyroscope data
    accel_data = mpu.get_accel_data()
    gyro_data = mpu.get_gyro_data()

    # Calculate pitch and roll from the accelerometer data
    accel_x = accel_data['x']
    accel_y = accel_data['y']
    accel_z = accel_data['z']

    pitch = math.atan2(accel_y, math.sqrt(accel_x ** 2 + accel_z ** 2))
    roll = math.atan2(-accel_x, accel_z)

    return pitch, roll


class PDController:
    def __init__(self, kp, kd):
        self.kp = kp  # Proportional gain
        self.kd = kd  # Derivative gain
        self.prev_error = np.full(NUM_MOTORS, 0)  # Previous error, initially 0

    def update(self, rpm_goal, current_rpm, dt):
        # Calculate the error
        error = rpm_goal - current_rpm

        # Calculate the derivative of the error
        derivative = (error - self.prev_error) / dt if dt > 0 else 0

        # Calculate the output from the PD controller
        duty_cycle_change = self.kp * error + self.kd * derivative

        # Save the error for the next iteration
        self.prev_error = error

        return duty_cycle_change


# Example of how to use the PD controller for bidirectional control
def get_duty_cycle(rpm_goal, current_rpm, current_duty_cycle, dt, pd_controller):
    # Update the duty cycle using the PD controller
    duty_cycle_change = pd_controller.update(rpm_goal, current_rpm, dt)

    # Adjust the current duty cycle based on the PD controller's output
    new_duty_cycle = current_duty_cycle + duty_cycle_change

    # Clamp the duty cycle to stay within the valid range [-100, 100]
    new_duty_cycle = np.clip(new_duty_cycle, -100, 100)

    return new_duty_cycle


def basic_duty_cycle(speed_goal):
    return speed_goal / MAX_SPEED * 100


GPIO.setmode(GPIO.BCM)

# pins_direction = []
for pin_encoder, pin_direction in zip(PINS_ENCODER, PINS_DIRECTION):
    GPIO.setup(pin_direction, GPIO.OUT)

# Drivers
pca = PCA9685(i2c)  # Lower body motors control
# pca2 = PCA9685(i2c, address=0x41) # A0                   # Upper body motors control + thumbs servos + neck servo + wrist servos
# pca3 = PCA9685(i2c, address=0x42) # A0 and A1            # Servos for all other fingers

pca.frequency = 60

# Set up the PWM channel for speed control (e.g., channel 0)
speed_channels = [pca.channels[i] for i in range(NUM_MOTORS)]
# speed_channels_upper = [pca2.channels[i] for i in range(NUM_MOTORS)]
# servos = [servo.Servo(pca3.channels[i]) for i in [14,15]]

# for servo in servos:
#    servo.angle = 180 # Basic way to set the angle
#    time.sleep(1)

# Test the servo encoders
if False:
    read_ads1115_on_tca_channel(0)  # this gets the values from first 4 servo encoders
    read_ads1115_on_tca_channel(1)
    time.sleep(0.5)


def map_duty_cycle_to_motor(value):
    if value > 0:
        if value < 50:
            # Gradual increase from 31 to 75 as the input goes from 31 to 50
            return 31 + (value / 50) * (92 - 31)
        else:
            # Gradual increase from 75 to 100 as the input goes from 50 to 100
            return 92 + ((value - 50) / 50) * (100 - 92)
    elif value < 0:
        if value > -50:
            # Gradual decrease from -31 to -75 as the input goes from -31 to -50
            return -31 + (value / 50) * (92 - 31)
        else:
            # Gradual decrease from -75 to -100 as the input goes from -50 to -100
            return -92 + ((value + 50) / 50) * (100 - 92)
    else:
        return 0


# Function to control motor
def control_motors(duty_cycles):
    for num in range(NUM_MOTORS):
        if duty_cycles[num] * orientation_motors[num] <= 0:
            GPIO.output(PINS_DIRECTION[num], GPIO.LOW)
            # pins_direction[num].value = False
        else:
            GPIO.output(PINS_DIRECTION[num], GPIO.HIGH)
            # pins_direction[num].value = True

        duty_cycle = duty_cycles[num] * ACTIVE_MOTORS[num]

        duty_cycle = int(abs(duty_cycle) / 100 * 65535)
        speed_channels[num].duty_cycle = duty_cycle


def quit_gpio():
    control_motors(np.full(NUM_MOTORS, 0))
    GPIO.cleanup()


try:
    control_motors(np.full(NUM_MOTORS, 0))
    input("Turn power on and press enter: ")
    time.sleep(3)
    control_motors(np.full(NUM_MOTORS, 0))

    if MANUAL_CALIBRATION:
        # manual calibration

        for num in range(NUM_MOTORS):

            print(f"Motor {num} manual calibration")

            duty_direction = np.full(NUM_MOTORS, 1)
            duty_control = np.full(NUM_MOTORS, 0)
            duty_control[num] = 75

            while True:

                angles_encoder = [(read_as5600_angle(val) - ANGLE_OFFSET[val]) * ANGLE_DIRECTION_ENCODER[val] for val in
                                  range(NUM_MOTORS)]
                angles = np.mod(np.array(angles_encoder) + np.pi, 2 * np.pi) - np.pi
                print("Angles encoder", angles_encoder)

                user_input = input("Press '' for forward, 'a' to change direction and x if ok: ")
                if user_input == '':
                    control_motors(duty_control * np.full(NUM_MOTORS, duty_direction))
                    time.sleep(0.05)
                    control_motors(np.full(NUM_MOTORS, 0))
                elif user_input == 'a':
                    duty_direction = -duty_direction
                elif user_input == 'x':
                    break

    if AUTO_CALIBRATION:

        calibrating = [True] * NUM_MOTORS  # Track which motors are still calibrating

        while any(calibrating):  # Continue until all motors are calibrated
            duty_control = np.full(NUM_MOTORS, 0)  # Initialize duty control for all motors

            for num in range(NUM_MOTORS):
                if calibrating[num]:  # Only calibrate motors that are not yet done
                    angle_encoder = (read_as5600_angle(num) - ANGLE_OFFSET[num]) * ANGLE_DIRECTION_ENCODER[num]
                    angle_encoder = -2 * math.pi + angle_encoder if angle_encoder > math.pi else angle_encoder
                    angle_encoder = 2 * math.pi - angle_encoder if angle_encoder < -math.pi else angle_encoder

                    if abs(angle_encoder) > 0.005:
                        # Set the duty control direction based on angle
                        duty_control[num] = 50 if angle_encoder < 0 else -50
                    else:
                        calibrating[num] = False  # Motor is calibrated
                        print(f"Motor {num} done calibrating, angle: {angle_encoder}")

            # Apply control to all motors simultaneously
            print(duty_control)
            control_motors(duty_control)
            time.sleep(0.03)

    # Control
    current_time = time.time()
    angles_encoder = [(read_as5600_angle(num) - ANGLE_OFFSET[num]) * ANGLE_DIRECTION_ENCODER[num] for num in
                      range(NUM_MOTORS)]
    angles_encoder = [- 2 * math.pi + angle_radian if angle_radian > math.pi else angle_radian for angle_radian in
                      angles_encoder]
    angles = [2 * math.pi - angle_radian if angle_radian < -math.pi else angle_radian for angle_radian in
              angles_encoder]

    if angles[2] > 0.05:
        DIRECTION_TEST = -1
    else:
        DIRECTION_TEST = 1

    pd_controller = PDController(kp=5, kd=0)  # Adjust kp to match the simulation motor

    while True:
        previous_angles = angles

        angles_encoder = [(read_as5600_angle(num) - ANGLE_OFFSET[num]) * ANGLE_DIRECTION_ENCODER[num] for num in
                          range(NUM_MOTORS)]  # Replace with : NUM_MOTORS # Takes 0.0085 s

        angles = np.mod(np.array(angles_encoder) + np.pi, 2 * np.pi) - np.pi

        interval = time.time() - current_time
        current_time = time.time()

        current_speeds = (np.array(angles) - np.array(previous_angles)) / interval

        print("Step time: ", np.round(interval, 4))
        if PRINTING:
            print("Angles; ", np.round(angles, 2))

        angles_normalized = 2 * ((angles - np.array(MIN_ANGLES)) / (np.array(MAX_ANGLES) - np.array(MIN_ANGLES))) - 1
        state_velocity_normalized = np.array(current_speeds) / MAX_SPEED
        set_speed_normalized = np.array(goal_speeds) / MAX_SPEED

        # reorder according to the model
        angles_normalized = {motor: value for motor, value in zip(MOTORS, angles_normalized)}
        angles_normalized = [angles_normalized[joint] for joint in JOINTS_LEGS]
        state_velocity_normalized = {motor: value for motor, value in zip(MOTORS, state_velocity_normalized)}
        state_velocity_normalized = [state_velocity_normalized[joint] for joint in JOINTS_LEGS]
        set_speed_normalized = {motor: value for motor, value in zip(MOTORS, set_speed_normalized)}
        set_speed_normalized = [set_speed_normalized[joint] for joint in JOINTS_LEGS]

        start_time = time.time()
        pitch, roll = get_root_orientation()  # Takes 0.006 to 0.01 s
        execution_time = time.time() - start_time
        print(f"Execution time: {execution_time:.6f} seconds")
        # print(f"Orientation (pitch, roll): {np.round(pitch, 3)} rad,{np.round(roll, 3)} rad")

        if PRODUCTION:
            state = np.concatenate(
                [angles_normalized, state_velocity_normalized, set_speed_normalized, [pitch, roll, 0], TASK_STATE, [0]])
            action, prediction = agent.act(state)
            goal_speeds = np.array(action) * MAX_SPEED
            if PRINTING:
                print("Actions: ", np.round(action, 2), "Predictions: ", np.round(prediction, 2))
                print("Goal speeds: ", goal_speeds)
                print("State: ", np.round(state, 2))

            STATES.append(state)

            if len(STATES) == TRAINING_BATCH:
                # TODO - SEND THE DATA TO THE OMEN FOR TRAINING: STATES, ACTIONS, PREDICTIONS, REWARDS, DONES
                STATES, ACTIONS, PREDICTIONS, REWARDS, DONES = [state], [], [], [], []

            ACTIONS.append(action)
            PREDICTIONS.append(prediction)

            reward = 0  # TODO - GET THE REWARD IN REAL TIME
            REWARDS.append(reward)

            done = 0  # TODO - PAUSE THE ITERATIONS AND REPLACE THE ROBOT AND GIVE -50 PENALTY OR CUSTOM
            DONES.append(done)

        elif TEST_FLEXION:
            goal_speeds = np.array([0, 0, 0.8, 0.8, -1.2, -1.2, 0.8, 0.8]) * DIRECTION_TEST
        elif SIMPLE_TEST:
            goal_speeds = np.array([0.2, -0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]) * DIRECTION_TEST

        goal_speeds = np.clip(goal_speeds, -1, 1)
        duty_cycles = basic_duty_cycle(goal_speeds)
        duty_cycles = [map_duty_cycle_to_motor(d) for d in duty_cycles]

        # Safety
        duty_cycles = np.where(
            (np.array(angles) < np.array(MIN_ANGLES)) & (np.array(duty_cycles) < np.full(NUM_MOTORS, 0)), 0,
            duty_cycles)  # Prevents from going too far !!
        duty_cycles = np.where(
            (np.array(angles) > np.array(MAX_ANGLES)) & (np.array(duty_cycles) > np.full(NUM_MOTORS, 0)), 0,
            duty_cycles)  # Prevents from going too far !!

        if PRINTING:
            print(f"Current speed: {np.round(current_speeds, 2)}")
            print(f"Duty Cycles: {np.round(duty_cycles, 2)}")
            print(f"Goal speed: {np.round(goal_speeds, 2)}")
            print(f"Filtered duty Cycles: {np.round(duty_cycles, 2)}")

        control_motors(duty_cycles)
        print("")

except KeyboardInterrupt:
    print("Caught KeyboardInterrupt (Ctrl+C)")
    quit_gpio()
finally:
    quit_gpio()