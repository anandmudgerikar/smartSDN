import numpy as np
import random
import pickle
from replay_buffer import ReplayBuffer

import sys
sys.path.append("/home/anand/gym")
import gym
import pickle
import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from env.sdn_gym import SDN_Gym

# env = gym.make('sdn-v0')
# env = gym.make('attack-sig-v0')
data = pd.read_csv("/home/anand/PycharmProjects/mininet_backend/pcaps/all_attacks.csv", sep=',', header=0)

feature_cols = ['Interval','pckts_forward', 'bytes_forward', 'pckts_back', 'bytes_back', 'label']
data = data[feature_cols]  # Features

# def data_to_replay_buffer(states, actions, rewards,next_states, matrix_D):
#     # num_time_slots = len(state)
#     # state = np.array(state)
#     # state_norm = (state - np.min(state)) / (np.max(state) - np.min(state))
#     # state_norm[np.isnan(state_norm)] = 1
#     # reward_norm = np.tanh(1 / np.array(reward))
#     # for time in range(num_time_slots - 1):
#     #     if (time + 1) % num_time_slot > 0:
#     #         current_state = state_norm[time]
#     #         current_action = np.array(action[time])
#     #         current_reward = reward_norm[time]
#     #         next_state = state_norm[time + 1]
#     #         if (time + 2) % num_time_slot > 0:
#     #             dones = False
#     #         else:
#     #             dones = True
#     for (state,action,reward,next_state,done) in (states,actions,rewards,next_states,dones):
#             matrix_D.add(state, action, reward, next_state, dones)
#
#     return matrix_D

num_trajectory = 100
num_users = 10
state_size = 4
server_threshold = 5

data_size = 6000-60#len(data)-1 - 60
user_turns = [0]*num_users
user_dones = [False]*num_users
user_state = [[0]*state_size for _ in range(num_users)]
next_state = [[0]*state_size for _ in range(num_users)]
action_tab = {0:0, 1: 0.1, 2:0.2, 3:0.3,4:0.4, 5:0.5, 6:0.6, 7:0.7, 8:0.8, 9:0.9, 10:1 }

joint_state = []
joint_action = []
joint_next_state = []
joint_reward = []

states=[]
actions=[]
next_states=[]
rewards=[]
dones=[]

matrix_D = ReplayBuffer(num_trajectory *  60)

joint_done = False

for trajectory in range(num_trajectory):

    # getting initial states for all users
    for user in range(num_users):
        user_turns[user] = random.randint(0, data_size)
        next_state[user] = data.values[user_turns[user], 1:5]
        next_state[user] = np.divide(next_state[user], np.array([500, 20000, 500, 20000]))
        user_dones[user] = False


    joint_done = False


    while not joint_done:

        joint_action = []
        joint_state = []
        joint_next_state = []
        joint_reward = 0
        action_sum = 0
        interval = 0

        for user in range(num_users):

            if user_dones[user]:
                user_state[user] = np.zeros(state_size)
            else:
                user_state[user] = next_state[user]


            joint_state.append(user_state[user])

            #random action
            user_action = random.randint(0,10)
            action_sum += user_action
            joint_action.append(user_action)

            #getting rewards
            joint_reward += ((1 - action_tab[user_action]) * user_state[user][1])

            #next time stamp
            user_turns[user] += 1

            #getting if done
            if (data.values[user_turns[user], 0] - data.values[user_turns[user] - 1, 0]) != 1 :
                user_dones[user] = True

            #getting next state
            if not user_dones[user]:
                next_state[user] = data.values[user_turns[user], 1:5]
                next_state[user] = np.divide(next_state[user], np.array([500, 20000, 500, 20000]))
                next_state[user] -= (action_tab[user_action] * user_state[user])  # pckts blocked
            else:
                next_state[user] = np.zeros(state_size)

            #next state
            # print(user_state)
            # print(next_state)
            user_state[user] = next_state[user]

            joint_next_state.append(next_state[user])

        if(all(done == True for done in user_dones)):
             joint_done = True

        interval +=1

        #print(joint_reward)
        #server threshold
        if joint_reward >= server_threshold:
            joint_reward = -100 #(-1*(joint_reward - server_threshold)/num_users)
        else:
            joint_reward = (100*(joint_reward)/num_users)

        print(joint_reward/interval)

        matrix_D.add(joint_state, joint_action, joint_reward, joint_next_state, joint_done)

        #print(joint_action)

        states.append(joint_state)
        actions.append(joint_action)
        rewards.append(joint_reward)
        next_states.append(joint_next_state)
        dones.append(joint_done)
#

#
#joint_matrix_D = data_to_replay_buffer(states, actions, rewards,next_states, matrix_D)
with open("replay_buffer_mal.pickle", "wb") as fp:
    pickle.dump(matrix_D, fp)

#print(states)

print("done")