# Dist Robotics

Traditional methods to train robots require powerful infrastructure and extremely high similarity between the simulated and real environments.

Creating the digital twin of the robot can be very time consuming. 
The goal here is to settle for a low level of digital ressemblance but use technics like domain randomization to allow the transfer the policy and evaluate the impact of fine-tuning in the real environment.

Through reinforcement learning from human feedback (RLHF), the robot should continuously refine its behavior and build intelligence through through teaching (the robot will mimic individuals).

<strong>Challenge:</strong> Building a humanoid robot from scratch for less than <strong>1000 CHF</strong> within <strong>12 months</strong>. (Currently at month 7)

<strong>Success Criteria:</strong> The robot should autonomously pick up a light object, like an apple, from point A and deliver it to point B, with under 50 hours of finetuning.

## Infra and Tools

- Python 3.10.4
- cuda-toolkit-12-4
- nvidia-driver-535
- https://www.liberiangeek.net/2024/04/install-cuda-on-ubuntu-24-04/ (or see install_cuda.md)

- GPU Nvidia RTX 4090: simulations & parameter tuning
- Raspberry PIs 5: compute engine of the robot
- FreeCad 0.21.2: Parts design
- Prusa & Ender 3 V3 (Creality): 3D print Plasic parts
- PyBullet (stable release 3.2.4) - 2022: Physics simulation engine

## Models

#### Learning (custom built using Tensorflow)

- A Feedforward critic model
- A Feedforward actor Model
- A Feedforward curiosity Model: For model-based learning (dreaming)
- A RLHF mechanism - TODO
- A large Transformer (VLM) based on all past robot-wold interactions - TODO

#### Vision

- Faster R-CNN, a well-known object detection framework, combined with Inception-ResNet v2 - 2019
- Pre-trained on the Open Images V4 dataset - 600 classes - XYZ box coordinates
- Stored locally: efficient_openimages
- Download the model: https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1?tf-hub-format=compressed
- Intel Midas 2 - For depth estimation: https://tfhub.dev/intel/midas/v2/2
- Get the model: https://www.kaggle.com/models/intel/midas/tensorFlow1/v2/2?tfhub-redirect=true
- Take a picture on the PI for the analysis:

```
libcamera-still -t 0 --timelapse 1000 -n -o "image_%04d.jpg"
```

#### Pose estimation from videos

- Google mediapipe solution though the mediapipe library installed and accessible in venv
( uses a combination of CNNs ) - 2020

#### Speech recognition and audio processing

- TODO

#### Sentences to tasks and dialogues

- LLM - TODO

#### Tasks verification

- Voting mechanism to evaluate tasks - TODO

## Methods from the following papers

| Source | Description |
|-------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| **Introduction to Reinforcement Learning** (Sutton & Barto, 1998/2018) | Foundational textbook covering reinforcement learning concepts, algorithms, and applications in AI and robotics. |
| **Proximal Policy Optimization (PPO) Paper** (Schulman et al., 2017) | Introduces PPO, a stable and efficient policy gradient method widely used in robotic control and AI training. |
| **DeepMimic** (Peng et al., 2018) | Presents an imitation learning framework that trains robots to mimic human-like motions from motion capture data. |
| **Continuous Control with Deep Reinforcement Learning** (Lillicrap et al., 2016) | Introduces concepts for reinforcement learning in continuous action spaces, crucial for robotics and autonomous control. |
| **Understanding Domain Randomization for Sim-to-Real Transfer** (Chen et al., 2021) | Explores how randomizing simulation parameters helps RL-trained models generalize better to real-world robotics. |

## Sensors and actuators

- Motor 5840-3650                              # Control
- Hall effect encoders                         # Joint angles
- Gyroscope and accelerometer : mpu6050        # Equillibrium
- PiCamera                                     # Vision
- Microphone: Mini USB Microphone Audio        # Receive audio commands
- Speakers: Max98357 I2S 3W Class D Amplifier  # Respond to audio commands

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

### 4. (optional) - use a video of a subject performing the task and add it to TASKS_MIMIC in **settings.py**

- First store a video (e.g TASK_NAME_mimic.mp4) in /src/mimic/
- Second: add TASK_NAME to the MIMICKING_TASKS in settings.py
- Create the frames (e.g):
```
# For example
python mimic.py --TASK squat_mimic
```

### 5. Define the pretraining level in **learning.py** 

For example walking uses the standing weights, so if the standing weights have already been trained, 
one can specify in learning.py that walking should use the weights of standing. For instance, one could add:

```
elif self.task == "walking_low_energy" or self.task =="walking_mimic" or self.task == "walking_robust":
    self.load_pretrained("walking")
```

### 6. Learn

Ensure cuda path is defined:

```
export CUDA_DIR=/usr/local/cuda-12.4
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_DIR

echo 'export CUDA_DIR=/usr/local/cuda-12.4' >> ~/.bashrc
echo 'export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_DIR' >> ~/.bashrc
source ~/.bashrc
```

Train the weights (and specify custom parameters), e.g:

```
python learning.py --TASK walking --LEARNING_RATE 0.00001 --ROBOT_TYPE full
```

Then test:

```
python learning.py --TESTING True --TASK walking --LEARNING_RATE 0.00001 --ROBOT_TYPE full
```

Reinforce the models by adding perturbations (to make it transferable to the real robot)

```
python learning.py --TASK walking_robust --LEARNING_RATE 0.00001 --ROBOT_TYPE full
```

Save the changes to the code and the newly generate weights:

```
git push origin main
```

### 7. Upload the weights in the robot (the raspberry pi)

```
git pull 
```

### 8. Try to control the robot with the new weights

- Connect the robot to the 24V power supply.
- Then run the following (it should automatically calibrate itself)

```
python control.py      # Easiest is to connect remotely with VStudio
```

### 9. Reinforce the model weights using real robot data (Human Feedback)

TODO: Sending arrays of state-action-rewards to a server for centralized training

### 10. Full size robot with vision

TODO: running vision_depth.py to constantly save and analyse pictures of the environment 
to enhance the state of the robot.

## Pictures and Videos

For now, only the lower body has been built to prove the concept. 
Once proper locomotion is achieved, the upper body will be added. 

| Robot V2 | Robot V3 |
|----------|----------|
| ![Robot V2](images/v2.png) | ![Robot V3](images/v3.png) |