import sys
sys.path.append("/home/anand/gym")
import gym
import pickle

from env.sdn_gym import SDN_Gym

env = gym.make('sdn-v0')

env.reset()
action = 0

filename = 'partial_sig.sav'
dec_tree = pickle.load(open(filename, 'rb'))
# print x
step_counter = 0
prev_ob = []
prev_ob_counter = 0

for i in range(1200):
    observation, reward, done, info = env.step(action)

    prev_ob.append(observation[4])

    if(len(prev_ob) == 6):
        prev_ob.pop(0)

    #print(prev_ob)
    print(observation[4])
    #print(reward)

# for _ in range(1000):
#     #env.render()
#     env.step(env.action_space.sample()) # take a random action
# env.close()