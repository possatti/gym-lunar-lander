#!/usr/bin/env python

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense

import matplotlib.pyplot as plt
import numpy as np
import datetime
import random
import keras
import gym
import sys
import os

"""
Deep Q-Learning attempt following this tutorial:
 - https://towardsdatascience.com/cartpole-introduction-to-reinforcement-learning-ed0eb5b58288

I miss some dropout after the Dense layers. I wonder if it could improve it. Also, the exploration
rate goes down just too fast, it seems to me.

And does it need to feed one experience at a time really? Maybe I could fit the entire batch right
at once, so that it goes faster.
"""

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 128

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995
# EXPLORATION_DECAY = 0.99995

ARANGE_BATCH_SIZE = np.arange(BATCH_SIZE)

class DQNSolver:

    def __init__(self, observation_space, action_space, model_path=None, is_test=False):
        if is_test:
            self.exploration_rate = 0
        else:
            self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space

        self.mem_all_idxs   = np.arange(MEMORY_SIZE, dtype=int)
        self.mem_state      = np.empty((MEMORY_SIZE, observation_space), dtype=float)
        self.mem_action     = np.empty((MEMORY_SIZE), dtype=int)
        self.mem_reward     = np.empty((MEMORY_SIZE,), dtype=float)
        self.mem_state_next = np.empty((MEMORY_SIZE, observation_space), dtype=float)
        self.mem_terminal   = np.empty((MEMORY_SIZE,), dtype=bool)
        self.mem_i = 0
        self.mem_n_filled = 0

        if model_path is None:
            self.model = Sequential()
            self.model.add(Dense(64, input_shape=(observation_space,), activation="relu"))
            self.model.add(Dense(32, activation="relu"))
            self.model.add(Dense(self.action_space, activation="linear"))
            self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))
            self.model.summary()
        else:
            self.model = keras.models.load_model(model_path)

    def remember(self, state, action, reward, state_next, terminal):
        self.mem_state[self.mem_i] =  state
        self.mem_action[self.mem_i] = action
        self.mem_reward[self.mem_i] = reward
        self.mem_state_next[self.mem_i] = state_next
        self.mem_terminal[self.mem_i] = terminal
        self.mem_i += 1
        self.mem_n_filled = np.min([MEMORY_SIZE, self.mem_n_filled+1])

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if self.mem_n_filled < BATCH_SIZE:
            return
        exp_idxs = np.random.choice(self.mem_all_idxs[0:self.mem_n_filled], size=BATCH_SIZE, replace=False)
        preds_next = self.model.predict(self.mem_state_next[exp_idxs])
        rewards = self.mem_reward[exp_idxs]
        states = self.mem_state[exp_idxs]
        q_update = np.where(self.mem_terminal[exp_idxs], rewards, rewards + GAMMA * np.amax(preds_next, axis=1))
        q_values = self.model.predict(states)
        q_values[ARANGE_BATCH_SIZE,self.mem_action[exp_idxs]] = q_update
        self.model.fit(states, q_values, verbose=0)
        # for i in range(BATCH_SIZE):
        #     self.model.fit(states[[i],:], q_values[[i],:], verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)
