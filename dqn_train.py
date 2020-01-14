#!/usr/bin/env python

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import datetime
import random
import time
import gym
import sys

from dqn_common_np import DQNSolver
from logger import CSVLogger

"""
Deep Q-Learning attempt following this tutorial:
 - https://towardsdatascience.com/cartpole-introduction-to-reinforcement-learning-ed0eb5b58288
"""

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

ENV_NAME = "LunarLander-v2"

NUM_TRAIN_EPISODES = 1000

SAVE_EACH_N_EPS = 1

ep_history = {
    'score': [],
    'exploration_rate': []
}

REWARD_CLOSER_TO_GROUND = True
REWARD_CLOSER_TO_CENTER = True

# State:
#   x
#   y
#   vel.x
#   vel.y
#   angle
#   angularVelocity
#   1.0 if self.legs[0].ground_contact else 0.0
#   1.0 if self.legs[1].ground_contact else 0.0

def main():
    start_timestamp = datetime.datetime.now().isoformat()
    save_dir = os.path.join('trained_models', start_timestamp)
    os.makedirs(save_dir, exist_ok=True)
    print("Saving models to: {}".format(save_dir), file=sys.stderr)

    log_path = os.path.join(save_dir, '0000_log.csv')
    logger = CSVLogger(log_path, ['ep', 'step', 'score', 'aug_score', 'exploration_rate'])

    env = gym.make(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space)
    for ep in range(NUM_TRAIN_EPISODES):
        ep_start = time.time()
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        ep_score = 0
        ep_aug_score = 0
        ep_step = 0
        while True:
            ep_step += 1
            action = dqn_solver.act(state)
            state_next, reward, terminal, info = env.step(action)
            ep_score += reward
            if REWARD_CLOSER_TO_GROUND:
                # ground_reward = max(0, (1 - state_next[1]) * 1)
                # reward += ground_reward
                # reward += min(0, state_next[1] * (-10))
                reward += min(0, state_next[1] * (-5))
            if REWARD_CLOSER_TO_CENTER:
                # reward += min(0, np.abs(state_next[0]) * (-10))
                reward += min(0, np.abs(state_next[0]) * (-5))
            ep_aug_score += reward
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            if terminal:
                ep_duration = time.time() - ep_start
                logger.log(ep, ep_step, ep_score, ep_aug_score, dqn_solver.exploration_rate)
                print(f'Ep: {ep:5}; Length: {ep_step:3}; Duration: {ep_duration:.2f} s ({ep_duration/ep_step:.4f} per step); Score: {ep_score:.4f}; Exploration rate: {dqn_solver.exploration_rate:.4f}')
                ep_history['score'].append(ep_score)
                ep_history['exploration_rate'].append(dqn_solver.exploration_rate)
                break
            dqn_solver.experience_replay()
        print(f'{ep} of {NUM_TRAIN_EPISODES} episodes')
        if ep % SAVE_EACH_N_EPS == 0 and ep != 0:
            dqn_solver.model.save(os.path.join(save_dir, f'my_model_EP{ep}.h5'))

if __name__ == "__main__":
    main()
