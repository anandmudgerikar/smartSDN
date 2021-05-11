import sys
sys.path.append("/home/anand/gym")
sys.path.append('/home/anand/PycharmProjects/mininet_backend/env/')
import gym
import pickle

from PycharmProjects.mininet_backend.env.sdn_routing_gym import SDN_Routing_Gym

env = gym.make('sdn-routing-v0')

env.reset()
action = 0

# print x

for i in range(1200):
    observation, reward, done, info = env.step(action)
    print(observation)
    print(reward)
