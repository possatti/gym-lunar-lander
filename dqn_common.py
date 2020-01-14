#!/usr/bin/env python

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from collections import deque

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
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

class DQNSolver:

    def __init__(self, observation_space, action_space, model_path=None, is_test=False):
        if is_test:
            self.exploration_rate = 0
        else:
            self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        if model_path is None:
            self.model = Sequential()
            self.model.add(Dense(64, input_shape=(observation_space,), activation="relu"))
            self.model.add(Dense(32, activation="relu"))
            self.model.add(Dense(self.action_space, activation="linear"))
            self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))
            self.model.summary()
        else:
            self.model = keras.models.load_model(model_path)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)
