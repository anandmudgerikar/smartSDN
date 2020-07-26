import sys
sys.path.append("/home/anand/gym")
import gym
import pickle

from .env.sdn_gym import SDN_Gym

env = gym.make('sdn-v0')

# from .env.attack_sig_gym import Attack_Sig_Gym
#
# env = gym.make('attack-sig-v0')

env.reset()
action = 0

# print x

for i in range(1200):
    observation, reward, done, info = env.step(action)
    print(observation)
    print(reward)
