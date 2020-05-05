import gym
import env
import itertools
import matplotlib
import matplotlib.style
import numpy as np
import pandas as pd
import sys
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import matplotlib.pyplot as plt

from collections import defaultdict
# import matplotlib, plotting
import random
from env import sdn_gym

env = gym.make('sdn-v0')

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.safeactions = deque(maxlen=2000)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24,input_dim=self.state_size, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done, time):
        for i in range(self.action_size):
            if(action[i] == 1):
                int_action = np.zeros((self.action_size,),dtype=int)
                int_action[i] = 1
                int_next_state = np.reshape(state,[state_size,1])
                int_next_state = int_next_state[:11] ^ int_action
                int_next_state = np.append(int_next_state[0], time%48)
                # print(int_next_state)
                int_next_state = np.reshape(int_next_state, [1, state_size])
                int_reward = env._get_reward_state(state, int_action,time)
                self.memory.append((state, int_action, int_reward, int_next_state, done, time))
                state = int_next_state


    def act(self, state):
        if np.random.rand() <= self.epsilon:
            action = env.action_space.sample()
            return action

        action = np.zeros((self.action_size,), dtype=int)
        curr_state = state
        #print(state[0][11])
        #print(state.shape)
        time = state[0][11]

        for i in range(self.action_size):
            act_values = self.model.predict(curr_state)
            # if ((agent.sec_check_state_action(state, np.argmax(act_values[0])) == False)):
            #     print("security breach")

            # print(np.argmax(act_values[0]))
            # return np.argmax(act_values[0])  # returns action

            # Make sure action is secure
            # while(agent.sec_check(state, act_values) == False):
            #     act_values = np.delete(act_values, np.argmax(act_values[0]))
            #     if (act_values.size == 0):
            #         return action

            action[np.argmax(act_values[0])] = 1
            #max quality function

            int_state = curr_state[0]
            int_state = int_state[0:11] ^ action
            int_state = np.append(int_state,time)
            #print(int_state)
            #print(action)
            curr_state = np.reshape(int_state, [1,state_size])
            #print(curr_state)
            #print(np.argmax(act_values[0]))
        #pos/neg quality function
        # for i in range(self.action_size):
        #     if(act_values[0][i] > 0):
        #         action[i] = 1

        # print(action)

        # if (self.sec_check_state_action(state, action) == False):
        #     env.sec_violations += 1

        return action



    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states, targets_f = [], []
        for state, action, reward, next_state, done,time in minibatch:
            target = reward
            if not done:
                act_values = self.model.predict(next_state)
                # print(act_values)
                while(agent.sec_check(next_state, act_values) == False):
                    act_values = np.delete(act_values, np.argmax(act_values[0]))
                    if (act_values.size == 0):
                        break;
                action_curr = np.zeros((self.action_size,), dtype=int)

                #quality fun 1
                if (act_values.size > 0):
                    action_curr[np.argmax(act_values[0])] = 1
                    cum_reward = env._get_reward_state(next_state, action_curr, time)
                    # print(cum_reward)
                    target = (reward + self.gamma * cum_reward)
                #quality fun 2
                # for i in range(self.action_size):
                #     if (act_values[0][i] > 0):
                #         action_curr[i] = 1



            target_f = self.model.predict(state)

            for i in range(action_size):
                if (action[i] == 1):
                    target_f[0][i] = target

            #print(action_f)
            # target_f = self.model.predict(state)

            # Filtering out states and targets for training
            states.append(state[0])
            targets_f.append(target_f[0])
            # print(state[0])
            # print(target_f[0])
        history = self.model.fit(np.array(states), np.array(targets_f), epochs=1, verbose=0)
            # print(state)
        # print(states)
        # Keeping track of loss
        loss = history.history['loss'][0]
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss

    def replay_wosec(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states, targets_f = [], []
        for state, action, reward, next_state, done,time in minibatch:
            target = reward
            if not done:
                act_values = self.model.predict(next_state)
                # print(act_values)
                # while(agent.sec_check(next_state, act_values) == False):
                #     act_values = np.delete(act_values, np.argmax(act_values[0]))
                #     if (act_values.size == 0):
                #         break;
                action_curr = np.zeros((self.action_size,), dtype=int)

                #quality fun 1
                if (act_values.size > 0):
                    action_curr[np.argmax(act_values[0])] = 1
                    cum_reward = env._get_reward_state(next_state, action_curr, time)
                    # print(cum_reward)
                    target = (reward + self.gamma * cum_reward)
                #quality fun 2
                # for i in range(self.action_size):
                #     if (act_values[0][i] > 0):
                #         action_curr[i] = 1



            target_f = self.model.predict(state)

            for i in range(action_size):
                if (action[i] == 1):
                    target_f[0][i] = target

            #print(action_f)
            # target_f = self.model.predict(state)

            # Filtering out states and targets for training
            states.append(state[0])
            targets_f.append(target_f[0])
            # print(state[0])
            # print(target_f[0])
        history = self.model.fit(np.array(states), np.array(targets_f), epochs=1, verbose=0)
            # print(state)
        # print(states)
        # Keeping track of loss
        loss = history.history['loss'][0]
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss



    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def sec_check(self, state, act_values):

        #  #reading sec policies
        #  #1 fridge
        # next_state_t[3] = 1
        # if(np.argmax(act_values[0]) == 3):
        #     return False
        # # # 3 office timings
        # if (state[0][11] > 9 and ((state[0][11]) < 17) and (np.argmax(act_values[0]) == 5 or np.argmax(act_values[0]) == 9 )):
        #     return False
        # #     next_state_t[5] = 0
        # #     next_state_t[9] = 0
        # # #4 sleeping
        # if(state[0][11] > 0 and ((state[0][11]) < 7) and (np.argmax(act_values[0]) == 5 or np.argmax(act_values[0]) == 9 )):
        #    return False
            #     next_state_t[5] = 0
        #     next_state_t[9] = 0
        action_curr = np.zeros((self.action_size,), dtype=int)

        # quality fun


        action_curr[np.argmax(act_values[0])] = 1

        x = (state[0][0:11].tolist(),action_curr.tolist())
        # print(x)
        if(env.MLNagent.memory.__contains__(x) == False):
            return False

        return True

    def sec_check_state_action(self, state, action):

        #  #reading sec policies
        #  #1 fridge
        # next_state_t[3] = 1
        # if(np.argmax(act_values[0]) == 3):
        #     return False
        # # # 3 office timings
        # if (state[0][11] > 9 and ((state[0][11]) < 17) and (np.argmax(act_values[0]) == 5 or np.argmax(act_values[0]) == 9 )):
        #     return False
        # #     next_state_t[5] = 0
        # #     next_state_t[9] = 0
        # # #4 sleeping
        # if(state[0][11] > 0 and ((state[0][11]) < 7) and (np.argmax(act_values[0]) == 5 or np.argmax(act_values[0]) == 9 )):
        #    return False
            #     next_state_t[5] = 0
        #     next_state_t[9] = 0


        # quality fun 1
        for i in range(action_size):
            action_curr = np.zeros((self.action_size,), dtype=int)
            if action[i]==1:
                action_curr[i] = 1
                x = (state[0][0:11].tolist(),action_curr.tolist())
                # print(x)
                if(env.MLNagent.memory.__contains__(x) == False):
                    #print("random does not action goes through")
                    #print(action)
                    return False

        # print("random action goes through")
        # print(action)
        return True

# Q, stats = qLearning(env, 1000)

EPISODES = 1000

state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
# print(state_size)
# print(action_size)
agent = DQNAgent(state_size, action_size)
# agent.load("./save/cartpole-dqn.h5")
done = False
batch_size = 100
action = np.zeros((11,))
