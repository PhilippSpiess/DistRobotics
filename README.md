# Dist Robotics

Traditional methods to train robots require powerful infrastructure and extremely high similarity between the simulated and real environments, a digital twin, where a control model is learnt and then transfered to the real robot.

But creating a precise digital twin is both very difficult and time consuming.

This projects here settles for a low level of digital resemblance but uses technics like domain randomization to allow the transfer the policy.
The goal here is also to evaluate the impact of fine-tuning (applying reinforcement learning in the real environment) and see how fast a robot adapt to the real world.

Another aspect of this project is to build intelligence through teaching (the robot will mimic individuals).

<strong>Challenge:</strong> Building a humanoid robot from scratch for less than <strong>1000 CHF</strong> within <strong>12 months</strong>. (Currently at month 7)

<strong>Success Criteria:</strong> The robot should autonomously pick up a light object, like an apple, from point A and deliver it to point B, with under 50 hours of finetuning.

More information at [Link Text](https://distrobotics.com)

## Infra and Tools

- Python 3.10.4
- cuda-toolkit-12-4, nvidia-driver-535
- GPU Nvidia RTX 4090: simulations & deep learning
- Raspberry PIs 5: robot compute engine
- FreeCad 0.21.2: Parts design
- Prusa & Ender 3 V3 (Creality): 3D print robot parts
- PyBullet (stable release 3.2.4) - 2022: Physics simulation engine

## Models

#### Learning (custom built using TensorFlow)

- A Feedforward critic model
- A Feedforward actor Model
- A Feedforward curiosity Model: For model-based learning (dreaming)
- A RLHF mechanism: training in the real world - TODO
- A diffusion model: mimicking - TODO

#### Vision

- Faster R-CNN, a well-known object detection framework, combined with Inception-ResNet v2 - 2019
- Pre-trained on the Open Images V4 dataset - 600 classes - XYZ box coordinates
- Download the model: https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1?tf-hub-format=compressed
- Intel Midas 2 - For depth estimation: https://tfhub.dev/intel/midas/v2/2
- Get the model: https://www.kaggle.com/models/intel/midas/tensorFlow1/v2/2?tfhub-redirect=true
- A VLM transformer: more powerful vision features extraction - TODO

#### Pose estimation from videos

- Google mediapipe solution though the mediapipe library ( uses a combination of CNNs ) - 2020

#### Automatic speech recognition (ASR)

- Transformer: TODO

#### Sentences to tasks

- LLM: TODO

#### Tasks verification

- Voting mechanism: TODO

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

## Using this code

### 1. Installation

```
git pull
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Try the robot in the simulation
```
# For example
python simulation.py --ROBOT_TYPE half
```
### 2. Define a task and add it to the array of TASKS in **settings.py**

Replace one of the placeholder value "" with the new task. (To avoid changing the dimension of the state)

### 3. Create a new reward logic in the reward function in **simulation.py**

Define rewards for each step (e.g +1 if closer to the goal and -50 if the robot fell)

### 4. (Optional) - Use a video of a subject performing the task and add it to TASKS_MIMIC in **settings.py**

- Store a video (e.g TASK_NAME.mp4) in /src/mimic/
- Create the frames (e.g):
```
# For example
python mimic.py --TASK squat
```

### 5. Define the pretraining level in **learning.py**

For example walking uses the standing weights, so if the standing weights have already been trained,
one can specify in learning.py that walking should use the weights of standing. For instance, one could add:

```
if self.task == "walking":
    self.load_pretrained("standing")
```

### 6. Learn

Train the weights (and specify certain parameters), e.g:

```
python learning.py --TASK walking --LEARNING_RATE 0.0001 --ROBOT_TYPE full
```

Then test:

```
python learning.py --TESTING True --TASK walking --LEARNING_RATE 0.0001 --ROBOT_TYPE full
```

Reinforce the models by adding perturbations (to make it transferable to the real robot)

```
python learning.py --ROBUST True
```

If available for the task, use pose estimations to improve learning  

```
python learning.py --MIMIC True
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

TODO: Sending arrays of state-action-rewards tuples to a dedicated server for centralized training

### 9. Use vision

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

## Pictures

For now, only the lower body has been built to prove the concept.
Once proper locomotion is achieved, the upper body will be added.

| Robot V2 | Robot V3 |
|----------|----------|
| ![Robot V2](images/v2.png) | ![Robot V3](images/v3.png) |
