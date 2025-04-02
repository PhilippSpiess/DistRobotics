# Dist Robotics

Traditional methods for training robots require powerful infrastructure and high similarity between simulated and real environments, often in the form of a digital twin, where a control model is learned in simulation and then transferred to the real robot.

However, creating an accurate digital twin is both difficult and time-consuming.

In this project, we prioritize development speed and flexibility over accuracy. Rather than investing in a highly precise digital twin, we explore techniques such as domain randomization to enable policy transfer. The project also aims to evaluate the impact of fine-tuning (i.e., applying reinforcement learning in the real environment) and to assess how quickly a robot can adapt to the real world.

Other key aspects of the project include building intelligence through teaching: the robot will learn by mimicking human behavior, with the community contributing to the development of tasks that robots can learn.

The end goal is a system where anyone can assemble their own robot and contribute to a shared global model by writing reward functions, making demonstrations, evaluating real-world robot trajectories, and validating or rejecting new tasks proposed by others.

**Challenge:** Build a humanoid robot from scratch for under **$1000** within **12 months**.

**Success Criteria:** The robot should autonomously pick up a light object (e.g., an apple) from **point A** and deliver it to **point B**.

#### Files

- Printable parts are located in `freecad/ROBOT.FCStd` and should be assembled using **M3, M4, and M6** screws/bolts.
- All electronic components are listed in `parts.csv`.
- The electronics schematic is in `robot.kicad_sch`.
- Code for offline training is in `/src`, and code for running on the real robot is in `/pi`.

More information at [www.distrobotics.com](https://www.distrobotics.com/)

---

## Infra and Tools

- Python 3.10
- cuda-toolkit-12-4, nvidia-driver-535
- GPU Nvidia RTX 4090: simulations & deep learning
- Raspberry PIs 5: robot compute engine
- FreeCad 0.21.2: Parts design
- Prusa & Ender 3 V3 (Creality): 3D print robot parts
- PyBullet (stable release 3.2.4) - 2022: Physics simulation engine

---

## Models

#### Learning (custom built using TensorFlow)

- A Feedforward critic model
- A Feedforward actor Model
- A Feedforward curiosity Model: For model-based learning (dreaming)
- A RLHF mechanism: training in the real world - TODO
- A diffusion model: mimicking - TODO

#### Vision

- Inception-ResNet v2 - 2019
- Pre-trained on the Open Images V4 dataset - 600 classes
- Download the model: https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1?tf-hub-format=compressed
- Intel Midas 2 - For depth estimation: https://tfhub.dev/intel/midas/v2/2
- Get the model: https://www.kaggle.com/models/intel/midas/tensorFlow1/v2/2?tfhub-redirect=true

#### Pose estimation from videos

- Google mediapipe solution though the mediapipe library ( uses a combination of CNNs ) - 2020

#### Automatic speech recognition (ASR)

- Transformer: TODO

#### Sentences to tasks

- LLM: TODO

#### Tasks verification

- Voting mechanism: TODO

---

## Main methods from the following papers

| Source | Description |
|-------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| **Introduction to Reinforcement Learning** (Sutton & Barto, 1998/2018) | Key reinforcement learning concepts, algorithms, and applications in AI and robotics. |
| **Proximal Policy Optimization (PPO) Paper** (Schulman et al., 2017) | PPO, a stable and efficient policy gradient method widely used in robotic control and AI training. |
| **DeepMimic** (Peng et al., 2018) | An imitation learning framework that trains robots to mimic human-like motions from motion capture data. |
| **Continuous Control with Deep Reinforcement Learning** (Lillicrap et al., 2016) | Concepts for reinforcement learning in continuous action spaces, crucial for robotics and autonomous control. |
| **Understanding Domain Randomization for Sim-to-Real Transfer** (Chen et al., 2021) | How randomizing simulation parameters helps RL-trained models generalize better to real-world robotics. |

## Sensors and actuators

```
- DC Motor 5840-3650                           # Control
- Hall effect encoders: AS5600                 # Joint angles
- Gyroscope and accelerometer: mpu6050         # Equillibrium
- PiCamera                                     # Vision
- Microphone: Mini USB Microphone Audio        # Receive audio commands
- Speakers: Max98357 I2S 3W Class D Amplifier  # Respond to audio commands
```

---

## Using this code

### 1. Installation

```
git pull
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Test the robot simulation
```
python simulation.py --ROBOT_TYPE full
```
### 2. Define a new task to learn and add it to the array of TASKS in **settings.py**

Replace one of the placeholder value "" with the new task. (To avoid changing the dimension of the state)

### 3. Create a new reward logic in the reward function in **simulation.py**

Define rewards for each step (e.g +1 if moving closer to a target and -50 if the robot falls)

### 4. (Optional) - Use a video of a subject performing the task

- Store a video (e.g TASK_NAME.mp4) in /src/mimic/
- Create the frames / pose estimations (e.g) by running the following:
```
# For example
python mimic.py --TASK task_name
```

### 5. (Optional) - Define the pretraining level in **learning.py**

For example walking uses the standing weights, so if the standing weights have already been trained,
one can specify in learning.py that walking should use the weights of standing. For example, one could add:

```
if self.task == "walking":
    self.load_pretrained("standing")
```

### 6. Learn

Train the weights (and specify certain parameters), e.g:

```
python learning.py --TASK walking --LEARNING_RATE 0.0001 --ROBOT_TYPE legs
```

Then test:

```
python learning.py --TESTING True --TASK walking --LEARNING_RATE 0.0001 --ROBOT_TYPE legs
```

Reinforce the models by adding perturbations (to make it transferable to the real robot)

```
python learning.py --ROBUST True
```

If available for the task, use pose estimations to improve learning  

```
python learning.py --TASK task_name --MIMIC True
```

Save the changes to the code and the newly generate weights:

```
git push origin main
```

### 7. Control the robot with the new weights:

- Connect the robot to the 24V power supply.
- Then run the following (The robot should first automatically calibrate itself)

```
git pull               # Updates the code and the weights
python control.py      # Easiest is to connect remotely with VStudio
```

### 8. Reinforce the model weights using real robot data (Human Feedback)

TODO: Sending arrays of state-action-rewards tuples to a dedicated server for centralised training

### 9. Use vision

TODO: Connect the cameras

Setup 2 background processes on the PI:

- Take picture on a regular basis
```
libcamera-still -t 0 --timelapse 1000 -n -o "image_%04d.jpg"
```
- Analyse the content of the pictures as often as possible
```
python vision_depth.py
```
This will constantly analyse the view of the robot an create a additional vision features that the robot can use to improve its state.

---

## Pictures

For now, only the lower body has been built to prove the concept.
Once proper locomotion is achieved, the upper body will be added.

| Robot V2 | Robot V3 |
|----------|----------|
| ![Robot V2](images/v2.png) | ![Robot V3](images/v3.png) |
