import numpy as np
import random
import pickle
from replay_buffer import ReplayBuffer

import sys
sys.path.append("/home/anand/gym")
import gym
import pickle
import keras

from env.sdn_gym import SDN_Gym

env = gym.make('attack-sig-v0')


num_trajectory = 10
num_users = 10
state_size = 4
server_threshold = 1

data_size = len(env.data)-1 - 60
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



def test(model,randaction= False):

    total_rewards = 0


    for trajectory in range(num_trajectory):

        prev_joint_state = []
        # getting initial states for all users
        for user in range(num_users):
            user_turns[user] = random.randint(0, data_size)
            next_state[user] = env.data.values[user_turns[user], 1:5]
            next_state[user] = np.divide(next_state[user], np.array([500, 20000, 500, 20000]))
            user_dones[user] = False
            prev_joint_state.append(next_state[user])

        joint_done = False
        sum_rewards = 0

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

                # random action
                if(randaction == True):
                    user_action = random.randint(0,10)
                else:
                    user_action = (model.policy_action(prev_joint_state)[user] + 10)//2
                    if(user_action <0):
                        user_action = 0
                #print(user_action)
                action_sum += user_action

                # getting rewards
                joint_reward += ((1 - action_tab[user_action]) * user_state[user][1])

                # next time stamp
                user_turns[user] += 1

                # getting if done
                if (env.data.values[user_turns[user], 0] - env.data.values[user_turns[user] - 1, 0]) != 1:
                    user_dones[user] = True

                # getting next state
                if not user_dones[user]:
                    next_state[user] = env.data.values[user_turns[user], 1:5]
                    next_state[user] = np.divide(next_state[user], np.array([500, 20000, 500, 20000]))
                    next_state[user] -= (action_tab[user_action] * user_state[user])  # pckts blocked
                else:
                    next_state[user] = np.zeros(state_size)

                # next state
                # print(user_state)
                # print(next_state)
                user_state[user] = next_state[user]

            prev_joint_state = joint_state

            if (all(done == True for done in user_dones)):
                joint_done = True

            # server threshold
            if joint_reward >= server_threshold:
                joint_reward = (-1* (joint_reward - server_threshold) / num_users)
            else:
                joint_reward = (10 * (joint_reward) / num_users)

            sum_rewards += joint_reward
            interval +=1

        total_rewards += (sum_rewards/interval) #/interval? not sure

    return (total_rewards/num_trajectory)