import gym
import random
from gym import error, spaces, utils
from gym.utils import seeding
import time
import pickle
import numpy as np
import pandas as pd

#discretizing actions : 1 = all pckts blocked, 0 = all allowed
action_tab = {0:0, 1: 0.1, 2:0.2, 3:0.3,4:0.4, 5:0.5, 6:0.6, 7:0.7, 8:0.8, 9:0.9, 10:1 }

class Attack_Sig_Gym(gym.Env):

    def __init__(self):

        #self.data = pd.read_csv("/home/anand/Dropbox/projects/thesis/smart_sdn/sec_anal/state_based/test2_new_train.csv", sep=',', header=0)
        self.data = pd.read_csv("/home/anand/PycharmProjects/mininet_backend/pcaps/all_attacks.csv", sep=',', header=0)

        feature_cols = ['Interval','pckts_forward', 'bytes_forward', 'pckts_back', 'bytes_back', 'label']
        self.data = self.data[feature_cols]  # Features

        # Printing the dataswet shape
        print("Dataset Length: ", len(self.data))
        print("Dataset Shape: ", self.data.shape)

        self.action_space = spaces.Discrete(11) #action = allow:1, drop:0
        self.observation_space = spaces.Tuple((spaces.Discrete(1000),spaces.Discrete(10000),spaces.Discrete(1000),spaces.Discrete(10000)))
        self.ob = self._get_initial_state()
        self.episode_over = False
        self.turns = 0
        self.sum_rewards = 0.0



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

        self.turns += 1
        self.ob = self._get_new_state(action)

        #self.reward = self._get_reward(action) #for jarvis
        self.reward = self._get_reward(action)

        self.sum_rewards += self.reward

        # if self.turns == len(self.data)-1:
        #      self.episode_over = True

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

        if (self.data.values[self.turns, 5] == "Malicious"):  # true neg
            reward -= ((1 - action_tab[action])*self.ob[1]*500*0.5)  #no of malicious packets going through
        else:
            reward += ((1 - action_tab[action])*self.ob[1]*500) #false benign packets going through

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
        next_state = np.divide(next_state, np.array([500, 20000, 500, 20000]))
        next_state -= (action_tab[action]* self.ob) #pckts blocked

        return next_state


    def _get_initial_state(self, stop = 0):
        if stop == 0:
            stop = len(self.data)-1
        start = random.randint(0,stop)
        self.turns = start
        self.episode_over = False
        state = self.data.values[start,1:5]
        state = np.divide(state,np.array([500,20000,500,20000]))
        return state

    def _seed(self):
        return



