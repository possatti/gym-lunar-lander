#!/usr/bin/env python

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import gym
import sys


env = gym.make('LunarLander-v2')

n_episodes = 1000

ep_history = {'total_reward': []}

for ep in range(n_episodes):
    done = False
    obs = env.reset()
    total_reward = 0
    while not done:
        if ep == n_episodes-1:
            env.render()
        action = env.action_space.sample()
        new_obs, reward, done, info = env.step(action)
        total_reward += reward
        obs = new_obs

    ep_history['total_reward'].append(total_reward)

    if ep % 100 == 0:
        print(f'{ep} of {n_episodes}')

env.close()

print('mean reward:', np.mean(ep_history['total_reward']))
print('median reward:', np.median(ep_history['total_reward']))

sns.set()
fig, axes = plt.subplots(1, 2, sharey=True)
sns.boxplot(y=ep_history['total_reward'], ax=axes[0])
sns.violinplot(y=ep_history['total_reward'], ax=axes[1])
# sns.swarmplot(y=ep_history['total_reward'], ax=axes[2])
plt.show()

