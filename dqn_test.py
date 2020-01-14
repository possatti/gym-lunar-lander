#!/usr/bin/env python

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import gym
import sys
import os

from dqn_common import DQNSolver

ENV_NAME = "LunarLander-v2"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('-t', '--trials', type=int, default=0)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    env = gym.make(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space, model_path=args.model_path, is_test=True)

    trial_scores = np.empty(args.trials)
    ep = 0
    while True:
        if ep == args.trials:
            print(f'Average score of {args.trials} trials: {trial_scores.mean()}')
        done = False
        state = env.reset()
        ep_score = 0
        while not done:
            if ep >= args.trials:
                env.render()
            action = dqn_solver.act(state[np.newaxis,:])
            state_next, reward, done, info = env.step(action)
            # print("state: {}".format(state), file=sys.stderr)
            if np.abs(reward) > 5 and ep >= args.trials:
                print("reward: {}".format(reward), file=sys.stderr)
            ep_score += reward
            state = state_next
        if ep < args.trials:
            trial_scores[ep] += ep_score
        print(f"score (ep=={ep}):", ep_score)
        ep += 1

if __name__ == "__main__":
    main()
