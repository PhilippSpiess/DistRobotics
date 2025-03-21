# Copyright (c) 2024 Philipp Spiess
# All rights reserved.

import os
import argparse
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K
import copy
from threading import Thread, Lock
from multiprocessing import Process, Pipe
import time

from simulation import SimulationEnv, SimulationParallelEnv
from helpers import timeit, create_folder_if_not_exists, add_log_entry
from parameters import *

plt.figure(figsize=(20, 10))
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

tf.random.set_seed(42)
random.seed(42)
np.random.seed(42)

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.set_logical_device_configuration(
        gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
    )

class ActorModel:

    def __init__(self, input_shape, action_space, optimizer_cls):
        self.action_space = action_space
        self.initial_entropy_bonus = ENTROPY_BONUS
        self.entropy_bonus = tf.Variable(self.initial_entropy_bonus, trainable=False, dtype=tf.float32)

        X_input = Input(shape=input_shape)
        X = Dense(LAYER_SIZE, activation='relu', kernel_initializer=tf.random_normal_initializer(stddev=0.03))(X_input)
        X = Dense(LAYER_SIZE, activation='relu', kernel_initializer=tf.random_normal_initializer(stddev=0.03))(X)
        X = Dense(LAYER_SIZE, activation='relu', kernel_initializer=tf.random_normal_initializer(stddev=0.03))(X)
        output = Dense(2 * self.action_space, activation='tanh')(X)

        self.model = Model(inputs=X_input, outputs=output)
        self.model.compile(loss=self.actor_loss, optimizer=optimizer_cls)

    @tf.function
    def actor_loss(self, y_true, y_pred):
        advantage_start = 0
        means_start = 1
        stds_start = means_start + self.action_space
        actions_start = stds_start + self.action_space
        curiosity_start = actions_start + self.action_space

        advantages = y_true[:, advantage_start:means_start]
        pred_means = y_true[:, means_start:stds_start]
        pred_stds = y_true[:, stds_start:actions_start] + 1e-10
        actions = y_true[:, actions_start:curiosity_start]
        curiosity = y_true[:, curiosity_start:]

        new_means = y_pred[:, :self.action_space]
        new_stds = (y_pred[:, self.action_space:] + 1) / 2 + 1e-10

        new_prob = self.normal_pdf(new_means, new_stds, actions)
        old_prob = self.normal_pdf(pred_means, pred_stds, actions)

        ratio = new_prob / (old_prob + 1e-10)
        p1 = ratio * advantages
        p2 = tf.clip_by_value(ratio, 1 - LOSS_CLIPPING, 1 + LOSS_CLIPPING) * advantages

        actor_loss = -tf.reduce_mean(tf.minimum(p1, p2))
        entropy_bonus = self.entropy_bonus * tf.reduce_mean(0.5 * tf.math.log(2 * np.pi * np.e * (tf.square(new_stds) + 1e-10)))
        curiosity_bonus = CURIOSITY_BONUS * curiosity

        return actor_loss - entropy_bonus - curiosity_bonus

    def normal_pdf(self, mean, std, action):
        return 1 / (std * tf.sqrt(2 * np.pi)) * tf.exp(-0.5 * ((action - mean) / std) ** 2)

    def update_entropy_bonus(self, new_value):
        self.entropy_bonus.assign(new_value)


class CriticModel:

    def __init__(self, input_shape, optimizer_cls):
        X_input = Input(shape=input_shape)

        V = Dense(LAYER_SIZE, activation="relu", kernel_initializer='he_uniform')(X_input)
        V = Dense(LAYER_SIZE, activation="relu", kernel_initializer='he_uniform')(V)
        V = Dense(LAYER_SIZE, activation="relu", kernel_initializer='he_uniform')(V)
        value = Dense(1, activation=None)(V)

        self.model = Model(inputs=X_input, outputs=value)
        self.model.compile(loss=self.critic_loss, optimizer=optimizer_cls)

    @tf.function
    def critic_loss(self, y_true, y_pred):
        values, target = y_true[:, :1], y_true[:, 1:]
        value_loss = K.mean((target - y_pred) ** 2)
        return value_loss


class CuriosityModel:
    # Useful for "dreaming": generating next_states with a neural network instead of using a simulation

    def __init__(self, input_shape, optimizer_cls):
        X_input = Input(shape=input_shape)

        V = Dense(LAYER_SIZE, activation="relu", kernel_initializer='he_uniform')(X_input)
        V = Dense(LAYER_SIZE, activation="relu", kernel_initializer='he_uniform')(V)
        V = Dense(LAYER_SIZE, activation="relu", kernel_initializer='he_uniform')(V)
        value = Dense(input_shape[0], activation=None)(V)

        self.model = Model(inputs=X_input, outputs=value)
        self.model.compile(loss="mse", optimizer=optimizer_cls, metrics=["mae"])


class PPOAgent:

    def __init__(self, env_name, render=False, task="standing", parallel=False, robot_type="full", mimic=False, robust=False, low_energy=False):

        if REAL:
            if robot_type in ROBOT_CONFIGS:
                self.action_size = ROBOT_CONFIGS[robot_type]["action_size"]
                self.state_size = ROBOT_CONFIGS[robot_type]["state_size"]
            else:
                raise ValueError(f"Unknown ROBOT_TYPE: {ROBOT_TYPE}")
        else:
            if parallel and not TESTING:
                self.env = SimulationParallelEnv(render, robot_type, task, mimic, robust, low_energy)
                render=False
                self.test_env = SimulationEnv(render, robot_type, task, mimic, robust, low_energy)
            elif TESTING:
                self.env = SimulationEnv(True, robot_type, task, mimic, robust, low_energy, print_reward=True)
                self.test_env = self.env
            else:
                self.env = SimulationEnv(render, robot_type, task, mimic, robust, low_energy)
                self.test_env = self.env
            self.action_size = self.env.dim_action
            self.state_size = (self.env.dim_state,)
        self.env_name = env_name
        self.task = task
        self.episode = 0
        self.max_episodes = MAX_EPISODES
        self.max_average = GOAL
        self.lr = LEARNING_RATE
        self.entropy_bonus = ENTROPY_BONUS
        self.epochs = EPOCHS
        self.shuffle = SHUFFLE
        self.Training_batch = TRAINING_BATCH
        self.optimizer_actor = Adam(self.lr)
        self.optimizer_critic = Adam(self.lr)
        self.optimizer_curiosity = Adam(self.lr)
        self.action_stds_average = 0.5

        self.replay_count = 0

        self.scores_, self.scores_deterministic_, self.episodes_, self.average_ = [], [], [], []

        self.Actor = ActorModel(input_shape=self.state_size, action_space=self.action_size,
                                 optimizer_cls=self.optimizer_actor)
        self.Critic = CriticModel(input_shape=self.state_size,
                                   optimizer_cls=self.optimizer_critic)
        self.Curiosity = CuriosityModel(input_shape=self.state_size,
                                   optimizer_cls=self.optimizer_curiosity)

        self.Actor_name = f"{self.env_name}_PPO_Actor.weights.h5"
        self.Critic_name = f"{self.env_name}_PPO_Critic.weights.h5"
        self.Curiosity_name = f"{self.env_name}_PPO_Curiosity.weights.h5"

        if REAL:
            self.load_pretrained(self.task)
        if robust or low_energy:
            self.load_pretrained(self.task)
        elif self.task == "walking":
            self.load_pretrained("standing")

    def act(self, state):

        state = np.array(state)
        state = np.squeeze(state).reshape(1, -1)

        prediction = self.Actor.model(state, training=False)[0].numpy()

        means = prediction[:self.action_size]
        stds = ((prediction[self.action_size:] + 1) / 2)

        action = np.random.normal(means, stds)

        return action, np.concatenate([means, stds])

    def get_gaes(self, rewards, dones, values, next_values, normalize=True):

        deltas = [r + GAMMA * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * GAMMA * LAMDA * gaes[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-10)
        return np.vstack(gaes), np.vstack(target)

    def replay(self, states, actions, rewards, predictions, dones, next_states):

        states = np.vstack(states)
        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        predictions = np.vstack(predictions)

        values = self.Critic.model(states)
        next_values = np.concatenate((values[1:], self.Critic.model(np.array([next_states[-1]]).reshape((1, self.state_size[0])))))

        next_states_predictions = self.Curiosity.model(states)
        curiosity = np.mean((next_states_predictions - next_states) ** 2, axis=1)
        curiosity = [[c] for c in curiosity]

        advantages, targets = self.get_gaes(rewards, dones, np.squeeze(values), np.squeeze(next_values))

        y_true = np.hstack([advantages, predictions, actions, curiosity])
        y_true_ = np.hstack([values, targets])

        start_time = time.time()
        a_loss = self.Actor.model.fit(states, y_true, epochs=self.epochs, verbose=0, shuffle=self.shuffle, batch_size=MINI_BATCH_SIZE)
        end_time = time.time()
        print(f"Execution time of actor: {end_time - start_time:.4f} seconds")
        start_time = time.time()
        c_loss = self.Critic.model.fit(states, y_true_, epochs=self.epochs, verbose=0, shuffle=self.shuffle, batch_size=MINI_BATCH_SIZE)
        end_time = time.time()
        print(f"Execution time of critic: {end_time - start_time:.4f} seconds")
        start_time = time.time()
        # cu_loss = self.Curiosity.model.fit(states, next_states, epochs=self.epochs, verbose=0, shuffle=self.shuffle, batch_size=MINI_BATCH_SIZE)
        end_time = time.time()
        print(f"Execution time of curiosity: {end_time - start_time:.4f} seconds")

        self.replay_count += 1

    def load(self):
        self.Actor.model.load_weights(FOLDER_NAME+"/"+self.task+"_"+self.Actor_name)
        self.Critic.model.load_weights(FOLDER_NAME+"/"+self.task+"_"+self.Critic_name)
        self.Curiosity.model.load_weights(FOLDER_NAME+"/"+self.task+"_"+self.Curiosity_name)

    def load_pretrained(self, prefix):
        self.Actor.model.load_weights("models/"+prefix+"_"+ROBOT_TYPE+"_"+str(LAYER_SIZE)+"_actor.weights.h5")
        self.Critic.model.load_weights("models/"+prefix+"_"+ROBOT_TYPE+"_"+str(LAYER_SIZE)+"_critic.weights.h5")
        self.Curiosity.model.load_weights("models/"+prefix+"_"+ROBOT_TYPE+"_"+str(LAYER_SIZE)+"_curiosity.weights.h5")

    def save(self, suffix=""):
        self.Actor.model.save_weights(FOLDER_NAME+"/"+self.task+suffix+"_"+self.Actor_name)
        self.Critic.model.save_weights(FOLDER_NAME+"/"+self.task+suffix+"_"+self.Critic_name)
        self.Curiosity.model.save_weights(FOLDER_NAME+"/"+self.task+suffix+"_"+self.Curiosity_name)

    def save_pretrained(self):
        self.Actor.model.save_weights("models/" + self.task+"_"+ROBOT_TYPE+"_"+str(LAYER_SIZE)+"_actor.weights.h5")
        self.Critic.model.save_weights("models/" + self.task+"_"+ROBOT_TYPE+"_"+str(LAYER_SIZE)+ "_critic.weights.h5")
        self.Curiosity.model.save_weights("models/" + self.task+"_"+ROBOT_TYPE+"_"+str(LAYER_SIZE)+"_curiosity.weights.h5")

    def overview(self, score, episode):

        if episode % 200 == 0:
            self.save()
            self.lr *= LEARNING_DECAY
            self.entropy_bonus *= ENTROPY_DECAY
            self.optimizer_actor.learning_rate.assign(self.lr)
            self.optimizer_critic.learning_rate.assign(self.lr)
            self.optimizer_curiosity.learning_rate.assign(self.lr)
            self.Actor.update_entropy_bonus(new_value=self.entropy_bonus)
            average_deterministic_score = self.test()
        else:
            average_deterministic_score = self.scores_deterministic_[-1]

        self.scores_deterministic_.append(average_deterministic_score)
        self.scores_.append(score)
        self.episodes_.append(episode)
        self.average_.append(sum(self.scores_[-50:]) / len(self.scores_[-50:]))

        if episode % 500 == 0:
            plt.plot(self.episodes_, self.average_, 'blue')
            plt.plot(self.episodes_, self.scores_deterministic_, 'black')
            plt.title(FOLDER_NAME.split("/")[-1].replace("_"," ") + " - PPO training cycle", fontsize=14)
            plt.ylabel('Score', fontsize=18)
            plt.xlabel('Episodes', fontsize=18)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.grid(True)
            plt.legend(['Exploring_avg', 'Deterministic'])
            plt.savefig(FOLDER_NAME+"/"+self.env_name + ".png")

        if episode==MAX_EPISODES:
            add_log_entry(FOLDER_NAME + " score " + str(self.average_[-1]) + " deterministic " + str(self.scores_deterministic_[-1]) + " episode " + str(episode), file_name="results.log")
        if self.scores_deterministic_[-1] > self.max_average:
            self.max_average = self.scores_deterministic_[-1]
            add_log_entry(FOLDER_NAME+" score "+str(self.average_[-1]) + " deterministic " + str(self.scores_deterministic_[-1])+" episode "+str(episode), file_name="success.log")
            self.save(suffix="_solved")
            self.save_pretrained()

        return self.average_[-1], self.scores_deterministic_[-1]

    def run_real(self, states, actions, rewards, predictions, dones):

        # states comes with one extra value
        states = states[:-1]
        next_states = states[1:]

        self.replay(states, actions, rewards, predictions, dones, next_states)
        self.save()

    def run_batch(self):
        state, info = self.env.reset()
        done, score = False, 0
        num_steps = 0
        while True:
            states, next_states, actions, rewards, predictions, dones = [], [], [], [], [], []
            for _ in range(self.Training_batch):
                action, prediction = self.act(state)
                next_state, reward, done, _, _ = self.env.step(action)
                states.append(state)
                next_states.append(next_state)
                actions.append(action)
                rewards.append(reward)
                predictions.append(prediction)
                state = next_state
                score += reward
                if num_steps==MAX_EPISODE_LENGTH:
                    done=1
                dones.append(done)
                if done:
                    average, deterministic = self.overview(score, self.episode)
                    self.episode += 1
                    print(f"episode: {self.episode}/{self.max_episodes}, score: {score}, average: {average:.2f} and {deterministic} in {num_steps} steps")

                    (state, info), done, score = self.env.reset(), False, 0
                    num_steps = 0
                num_steps += 1

            self.replay(states, actions, rewards, predictions, dones, next_states)
            if self.episode >= self.max_episodes:
                break
        self.env.close()

    def run_parallel(self, num_workers=64):

        states = [[] for _ in range(num_workers)]
        next_states = [[] for _ in range(num_workers)]
        rewards = [[] for _ in range(num_workers)]
        dones = [[] for _ in range(num_workers)]
        actions = [[] for _ in range(num_workers)]
        predictions = [[] for _ in range(num_workers)]

        score = [0 for _ in range(num_workers)]
        num_steps = [0 for _ in range(num_workers)]

        state = [self.env.reset()[0] for _ in range(num_workers)]
        while self.episode <= self.max_episodes:

            predictions_list = self.Actor.model(np.reshape(state, [num_workers, self.state_size[0]]))
            predictions_list = [ (prediction[:self.action_size] , ((prediction[self.action_size:] + 1) / 2) ) for prediction in predictions_list]
            actions_list = [np.random.normal(means, stds) for (means, stds) in predictions_list]
            predictions_list = [np.concatenate([means, stds]) for (means, stds) in predictions_list]

            for idx in range(num_workers):
                actions[idx].append(actions_list[idx])
                predictions[idx].append(predictions_list[idx])

            next_states_, rewards_, dones_, _, _ = self.env.step(actions_list)

            for idx, (next_state, reward, done) in enumerate(zip(next_states_, rewards_, dones_)):

                states[idx].append(state[idx])
                next_states[idx].append(next_state)
                rewards[idx].append(reward)
                state[idx] = next_state
                score[idx] += reward

                if num_steps[idx] == MAX_EPISODE_LENGTH:
                    done = 1
                dones[idx].append(done)

                if done:
                    average, deterministic = self.overview(score[idx], self.episode)
                    print(
                        f"episode: {self.episode}/{self.max_episodes}, worker: {idx}, score: {score[idx]}, average: {average:.2f} and {deterministic} in {num_steps[idx]} steps")
                    next_state, _ = self.env.reset(idx)
                    state[idx] = next_state
                    score[idx] = 0
                    num_steps[idx] = 0
                    self.episode += 1
                num_steps[idx] += 1

            total_data_points = sum(len(states[idx]) for idx in range(num_workers))

            if total_data_points >= self.Training_batch:

                concatenated_states = np.concatenate([np.array(states[idx]) for idx in range(num_workers)])
                concatenated_actions = np.concatenate([np.array(actions[idx]) for idx in range(num_workers)])
                concatenated_rewards = np.concatenate([np.array(rewards[idx]) for idx in range(num_workers)])
                concatenated_predictions = np.concatenate([np.array(predictions[idx]) for idx in range(num_workers)])
                concatenated_dones = np.concatenate([np.array(dones[idx]) for idx in range(num_workers)])
                concatenated_next_states = np.concatenate([np.array(next_states[idx]) for idx in range(num_workers)])

                self.replay(concatenated_states, concatenated_actions, concatenated_rewards, concatenated_predictions,
                            concatenated_dones, concatenated_next_states)

                for idx in range(num_workers):
                    states[idx], next_states[idx], actions[idx], rewards[idx], dones[idx], predictions[idx] = [],[],[],[],[],[]

        self.env.close()

    def test(self, test_episodes=5):
        scores = []
        self.load()
        for e in range(test_episodes):
            state, info = self.test_env.reset()
            done = False
            score = 0
            num_steps = 0
            rewards = []
            actions = []
            actions_stds_averages = []
            actions_means = []
            actions_stds = []
            while not done:
                state = np.array(state)
                state = np.squeeze(state).reshape(1, -1)
                action = self.Actor.model(state, training=False)[0].numpy()
                actions.append(action)
                action_means = action[:self.action_size]
                action_stds = ((action[self.action_size:] + 1) / 2)
                state, reward, done, _, _ = self.test_env.step(action_means)
                score += reward
                rewards.append(round(reward,2))
                actions_stds_averages.append(np.mean(action_stds))
                actions_means.append(action_means)
                actions_stds.append(action_stds)
                num_steps+=1
                if num_steps==MAX_EPISODE_LENGTH:
                    done=1
                if done:
                    break
            scores.append(score)
            random_index = random.randint(0, len(actions) - 1)
            action_means_to_eval = actions_means[random_index]
            action_stds_to_eval = actions_stds[random_index]
            print(f"Action step {random_index} means: ", action_means_to_eval)
            print(f"Action step {random_index} stds: ", action_stds_to_eval)
            print("Episode rewards", rewards)
            print(f"episode: {e + 1}/{test_episodes}, score: {score} in {num_steps} steps")
        self.action_stds_average = np.mean(actions_stds_averages)
        print("Avearge action STDs: ", self.action_stds_average)
        return round(np.mean(scores),2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--LEARNING_RATE", type=float, default=LEARNING_RATE)
    parser.add_argument("--LAYER_SIZE", type=int, default=LAYER_SIZE)
    parser.add_argument("--MAX_EPISODE_LENGTH", type=int, default=MAX_EPISODE_LENGTH)
    parser.add_argument("--ENTROPY_BONUS", type=float, default=ENTROPY_BONUS)
    parser.add_argument("--CURIOSITY_BONUS", type=float, default=CURIOSITY_BONUS)
    parser.add_argument("--LEARNING_DECAY", type=float, default=LEARNING_DECAY)
    parser.add_argument("--ENTROPY_DECAY", type=float, default=ENTROPY_DECAY)
    parser.add_argument("--LOSS_CLIPPING", type=float, default=LOSS_CLIPPING)
    parser.add_argument("--GAMMA", type=float, default=GAMMA)
    parser.add_argument("--TRAINING_BATCH", type=int, default=TRAINING_BATCH)
    parser.add_argument("--MINI_BATCH_SIZE", type=int, default=MINI_BATCH_SIZE)
    parser.add_argument("--SHUFFLE", type=str, default=SHUFFLE)
    parser.add_argument("--PARALLEL", type=str, default=PARALLEL)
    parser.add_argument("--TESTING", type=str, default=TESTING)
    parser.add_argument("--TASK", type=str, default=TASK)
    parser.add_argument("--ROBOT_TYPE", type=str, default=ROBOT_TYPE)
    parser.add_argument("--NUM_WORKERS", type=int, default=NUM_WORKERS)
    parser.add_argument("--ROBUST", type=str, default=ROBUST)
    parser.add_argument("--MIMIC", type=str, default=MIMIC)
    parser.add_argument("--LOW_ENERGY", type=str, default=LOW_ENERGY)
    args = parser.parse_args()
    LEARNING_RATE = args.LEARNING_RATE
    LAYER_SIZE = args.LAYER_SIZE
    NUM_WORKERS = args.NUM_WORKERS
    MAX_EPISODE_LENGTH = args.MAX_EPISODE_LENGTH
    ENTROPY_BONUS = args.ENTROPY_BONUS
    CURIOSITY_BONUS = args.CURIOSITY_BONUS
    LEARNING_DECAY = args.LEARNING_DECAY
    ENTROPY_DECAY = args.ENTROPY_DECAY
    LOSS_CLIPPING = args.LOSS_CLIPPING
    GAMMA = args.GAMMA
    MINI_BATCH_SIZE = args.MINI_BATCH_SIZE
    TRAINING_BATCH = args.TRAINING_BATCH
    SHUFFLE = str(args.SHUFFLE) == "True"
    PARALLEL = str(args.PARALLEL) == "True"
    TESTING = str(args.TESTING) == "True"
    TASK = args.TASK
    ROBOT_TYPE = args.ROBOT_TYPE
    ROBUST = str(args.ROBUST) == "True"
    MIMIC = str(args.MIMIC) == "True"
    LOW_ENERGY = str(args.LOW_ENERGY) == "True"

    if REAL or REAL_TRAINING:
        env_name = 'Real'+"_"+ROBOT_TYPE+"_"+TASK
    else:
        env_name = 'PyBullet' + "_" + ROBOT_TYPE + "_" + TASK

    parameters = "LEARNING_RATE_"+str(LEARNING_RATE)+"_LAYER_SIZE_"+str(LAYER_SIZE)+"_MINI_BATCH_SIZE_"+str(MINI_BATCH_SIZE)
    parameters+= "_TRAINING_BATCH_"+str(TRAINING_BATCH)+"_NUM_WORKERS_"+str(NUM_WORKERS)+"_ENTROPY_BONUS_"+str(ENTROPY_BONUS)
    FOLDER_NAME = env_name+"/"+parameters
    create_folder_if_not_exists(FOLDER_NAME)

    create_folder_if_not_exists(env_name)
    if REAL_TRAINING:
        agent = PPOAgent(env_name, render=False, task=TASK, robot_type=ROBOT_TYPE, mimic=MIMIC, robust=ROBUST, low_energy=LOW_ENERGY)
        while True:
            # TODO: get the data from the raspberry pi
            # agent.run_real(states, next_states, actions, rewards, predictions, dones)
            pass
    elif REAL:
        agent = PPOAgent(env_name, render=False, task=TASK, robot_type=ROBOT_TYPE, mimic=MIMIC, robust=ROBUST, low_energy=LOW_ENERGY)
        state = np.random.uniform(-1, 1, size=78)
        start_time = time.time()
        action, prediction = agent.act(state)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")
        print(action)
        print(prediction)
    elif TESTING:
        agent = PPOAgent(env_name, render=True, task=TASK, robot_type=ROBOT_TYPE, mimic=MIMIC, robust=ROBUST, low_energy=LOW_ENERGY)
        average_score = agent.test()
        agent.env.close()
    elif PARALLEL:
        agent = PPOAgent(env_name, render=False, task=TASK, parallel=PARALLEL, robot_type=ROBOT_TYPE, mimic=MIMIC, robust=ROBUST, low_energy=LOW_ENERGY)
        agent.run_parallel(num_workers = NUM_WORKERS)
    else:
        agent = PPOAgent(env_name, render=False, task=TASK, robot_type=ROBOT_TYPE, mimic=MIMIC, robust=ROBUST, low_energy=LOW_ENERGY)
        agent.run_batch()