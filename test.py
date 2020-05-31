import sys
sys.path.append("/home/anand/gym")
import gym
import pickle

from env.sdn_gym import SDN_Gym

env = gym.make('sdn-v0')

env.reset()
action = (0,0,0,0,0)

filename = 'partial_sig.sav'
dec_tree = pickle.load(open(filename, 'rb'))
# print x

for i in range(1200):
    observation, reward, done, info = env.step(action)
    sec_viol = dec_tree.predict([[env.previous_5_loadsum[0]]])

    if (observation[0] == 0):
        sec_viol[0] = 'NULL'

    print(observation,env.previous_5_loadsum[0],sec_viol[0])
    #print(reward)

# for _ in range(1000):
#     #env.render()
#     env.step(env.action_space.sample()) # take a random action
# env.close()