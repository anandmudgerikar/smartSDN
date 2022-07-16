import gym
import random
from gym import error, spaces, utils
from gym.utils import seeding
import time
import pickle
import numpy as np
import pandas as pd
import keras

#discretizing actions : 1 = all pckts blocked, 0 = all allowed
action_tab = {0:0, 1: 0.1, 2:0.2, 3:0.3,4:0.4, 5:0.5, 6:0.6, 7:0.7, 8:0.8, 9:0.9, 10:1 }

class SimulatedSDNRateControlGym(gym.Env):

    def __init__(self):

        #Reading Training dataset
        #self.data = pd.read_csv("/home/anand/Dropbox/projects/thesis/smart_sdn/rateControl/state_based/test2_new_train.csv", sep=',', header=0)
        self.data = pd.read_csv("/home/anand/PycharmProjects/mininet_backend/pcaps/mal_fixed_interval.csv", sep=',', header=0)

        # Features
        feature_cols = ['Interval','pckts_forward', 'bytes_forward', 'pckts_back', 'bytes_back', 'label']
        self.data = self.data[feature_cols]

        #normalizing data
        self.state_max = np.array([self.data.max(axis=0).values[1],self.data.max(axis=0).values[2],self.data.max(axis=0).values[3],self.data.max(axis=0).values[4]])
        self.state_min = np.array([self.data.min(axis=0).values[1], self.data.min(axis=0).values[2],self.data.min(axis=0).values[3], self.data.min(axis=0).values[4]])

        # Printing the dataswet shape
        print("Dataset Length: ", len(self.data))
        print("Dataset Shape: ", self.data.shape)

        #setting state,actions and reward spaces
        self.action_space = spaces.Discrete(11)
        self.observation_space = spaces.Tuple((spaces.Discrete(1000),spaces.Discrete(10000),spaces.Discrete(1000),spaces.Discrete(10000)))
        self.ob = self._get_initial_state()
        self.episode_over = False
        self.turns = 0
        self.sum_rewards = 0.0
        self.state_size = len(self.observation_space.spaces)

        #security model
        self.reconstructed_model = keras.models.load_model("/home/anand/PycharmProjects/mininet_backend/dqn_agent_fixed_interval")
        self.dnn_model = keras.models.load_model("./rateControl/dnn_model")

    def _step(self, action):
        """
        Parameters
        ----------
        action_index :
        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """

        # ##with security constraints from dqn
        # state = np.reshape(self.ob, [1, self.state_size])
        # q_values = self.reconstructed_model.predict(state)
        # f1 = 0.1
        # self.reward = self.reward*(1-f1) + (f1)*q_values[0][action]

        ##with security constraints from dnn
        state = np.multiply(np.array(self.ob), self.state_max - self.state_min) + self.state_min

        q_values = self.dnn_model.predict([[state]])
        #print(q_values)
        f1 = 0.9
        self.reward = self.reward*(1-f1) - (((f1)*q_values[0][1])/1000)
        self.sum_rewards += self.reward

        if (self.data.values[self.turns, 0] - self.data.values[self.turns-1, 0]) != 1: #episode size
            self.episode_over = True

        if self.turns == len(self.data)-1:
            self.turns = 0
            self.episode_over = True

        # if((self.turns % 60) == 0):
        #     self.episode_over = True

        return self.ob, self.reward, self.episode_over, {}

    def _reset(self):
        """
        Reset the environment and supply a new state for initial state
        :return:
        """
        #self.turns = 0
        self.ob = self._get_initial_state()
        self.episode_over = False
        self.sum_rewards = 0.0

        return self.ob

    def _get_reward(self,action):
        """
        Get reward for the action taken in the current state
        :return:
        Fine tuning versions:
        v1 = fpr = 100 , tnr=100
        v2 : fpr=10*bytes_forward, tnr = 2*bytes_forward
        v3 : fpr=10*bytes_forward, tnr = 10*bytes_forward
        v4 : more exploration 0.2
        v5 : more exploration 0.2, fpr=bytes_forward, tnr = bytes_forward
        v6 : more exploration 0.2, fpr=bytes_forward, tnr = 10*bytes_forward
        v7: more exploration 0.2, fpr=bytes_forward, tnr = 10*bytes_forward, +ve reward = 500
         v8: more exploration 0.2, fpr=pckts_forward*5, tnr = pckts_forward
         v8: more exploration 0.2, fpr=pckts_forward*5, tnr = pckts_forward
         v9: more exploration 0.2, fpr=pckts_forward, tnr = pckts_forward
         v10: more exploration 0.2, fpr=pckts_forward, tnr = pckts_forward,  +ve reward = 500
         v11: more exploration 0.2, fpr=pckts_forward*2, tnr = pckts_forward,  +ve reward = 1000
         v12: more exploration 0.2, fpr=pckts_forward, tnr = pckts_forward,  +ve reward = 1000
         v13: more exploration 0.2, fpr=pckts_forward, tnr = pckts_forward*5,  +ve reward = 1000
         v14 : more exploration 0.2, fpr=pckts_forward, tnr = pckts_forward*5,  +ve reward = 500
         v15 : more exploration 0.2, fpr=pckts_forward, tnr = pckts_forward*5,  +ve reward = 500, ds=train_dataset(k attacks)
         v16 : more exploration 0.2, fpr=pckts_forward*10, tnr = pckts_forward,  +ve reward = 500, ds=train_dataset(k attacks)
         v17 : more exploration 0.2, fpr=pckts_forward, tnr = pckts_forward,  +ve reward = 300, ds=train_dataset(k attacks)
         v18 : more exploration 0.2, fpr=pckts_forward*40, tnr = pckts_forward*20,  +ve reward = 200, ds=train_dataset(k attacks) - best
         v19 : more exploration 0.2, fpr=pckts_forward*20, tnr = pckts_forward*20,  +ve reward = 200, ds=train_dataset(k attacks)
         v20 : more exploration 0.2, fpr=pckts_forward*20, tnr = pckts_forward*20,  +ve reward = 300, ds=train_dataset(k attacks)
         v21 : more exploration 0.2, fpr=pckts_forward*30, tnr = pckts_forward*20,  +ve reward = 200, ds=train_dataset(k attacks)
         v22 : more exploration 0.2, fpr=pckts_forward*40, tnr = pckts_forward*30,  +ve reward = 200, ds=train_dataset(k attacks)
         v23 : more exploration 0.2, fpr=pckts_forward*40, tnr = pckts_forward*40,  +ve reward = 200, ds=train_dataset(k attacks)
         v24 : more exploration 0.2, fpr=pckts_forward*40, tnr = pckts_forward*20,  +ve reward = 200, ds=train_dataset(k attacks), episode size = 10
         v25 : more exploration 0.2, fpr=-4, tnr = -2,  +ve reward = 2, ds=train_dataset(k attacks)
         v26 : more exploration 0.2, fpr=-3, tnr = -2,  +ve reward = 2, ds=train_dataset(k attacks)
         v27 : more exploration 0.2, fpr=-3, tnr = -2,  +ve reward = 2,4, ds=train_dataset(k attacks)
         v27 : more exploration 0.2, fpr=-3, tnr = -2,fn =3, tp=6  , ds=train_dataset(k attacks)
         v28 : more exploration 0.2, fpr=-3, tnr = -2,fn =1, tp=1  , ds=train_dataset(k attacks)
         v29 : more exploration 0.2, fpr=-1, tnr = -2,fn =1, tp=1  , ds=train_dataset(k attacks), less replay (todo)
         v30 : more exploration 0.2, fpr=-2, tnr = -1,fn =5, tp=5  , ds=train_dataset(k attacks), less replay
         v31 : more exploration 0.2, fpr=-2, tnr = -1,fn =5, tp=5  , ds=train_dataset(k attacks) (todo)
         v32: full episode training, no temporal (31)
        """
        reward = 0

        # # ##Security Reward
        # if (self.data.values[self.turns, 5] == "Malicious"):  # true neg
        #     reward -= ((1 - action_tab[action])*self.ob[1]*0.5)  #no of malicious bytes going through #*500
        # else:
        #     reward += ((1 - action_tab[action])*self.ob[1]) #false benign bytes going through #*500

        # f1 = 2
        # reward = reward*f1
        ##Load Balancing Reward
        mean, sigma = 30000, 5000
        server_capacity = np.random.normal(mean,sigma)
        #normalizing
        server_capacity = (server_capacity - self.state_min[1])/(self.state_max[1] - self.state_min[1])
        reward += ((1 - action_tab[action]) * self.ob[1])

        if reward >= server_capacity:
            reward = -0.01  # (-1*(joint_reward - server_threshold)/num_users)
            return reward
        # else:
        #     reward += ((1 - action_tab[action]) * self.ob[1])

        #user fairness reward
        mean, sigma = 500, 100
        user_sla = np.random.normal(mean, sigma)
        # normalizing
        user_sla = (user_sla - self.state_min[1]) / (self.state_max[1] - self.state_min[1])

        # reward += ((1 - action_tab[action]) * self.ob[1])

        if reward < user_sla and self.ob[1] > user_sla:
            reward = -0.001  # (-1*(joint_reward - server_threshold)/num_users)

        return reward

        # if(action == 0): #allow
        #     if(self.data.values[self.turns,4] == "Malicious"): #true neg
        #         reward -= 1 #no of malicious packets going through
        #     else:
        #         reward +=5 #false negative
        # else:
        #     if(self.data.values[self.turns,4] == "Benign"): #false positive
        #         reward -= 2 #no of benign packets getting dropped
        #     else:
        #         reward += 5 #true positive
        # return reward

    def _get_new_state(self,action):
        """
        Get the next state from current state
        :return:
        """
        next_state = self.data.values[self.turns,1:5]
        next_state = np.divide(next_state - self.state_min, self.state_max - self.state_min)
        next_state -= (action_tab[action]* self.ob) #pckts blocked

        return next_state


    def _get_initial_state(self, begin=0,stop = 0, full_episode=False):
        if stop == 0:
            stop = len(self.data)-1-60
        start = random.randint(begin,stop)
        if(full_episode == True):
            while(self.data.values[start,0] != 0):
                start +=1
        self.turns = start
        self.episode_over = False
        state = self.data.values[start,1:5]
        state = np.divide(state - self.state_min, self.state_max - self.state_min)
        return state

    def _seed(self):
        return



