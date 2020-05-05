import gym
import numpy as np
import pandas as pd
from env.sdn_gym import SDN_Gym

env = gym.make('sdn-v0')

env.reset()
action = (0,0,0,0,0)
# print x



for i in range(3):
    observation, reward, done, info = env.step(action)
    print(observation)
    print(reward)

# for _ in range(1000):
#     #env.render()
#     env.step(env.action_space.sample()) # take a random action
# env.close()