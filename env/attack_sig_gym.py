import gym
import random
from gym import error, spaces, utils
from gym.utils import seeding
import time
import pickle
import numpy as np
import pandas as pd


class Attack_Sig_Gym(gym.Env):

    def __init__(self):

        self.data = pd.read_csv("/home/anand/Dropbox/projects/thesis/smart_sdn/sec_anal/state_based/test1_2sec.csv", sep=',', header=0)

        feature_cols = ['pckts_forward', 'bytes_forward', 'pckts_back', 'bytes_back', 'label']
        self.data = self.data[feature_cols]  # Features

        # Printing the dataswet shape
        print("Dataset Length: ", len(self.data))
        print("Dataset Shape: ", self.data.shape)

        self.action_space = spaces.Discrete(2) #action = allow:1, drop:0
        self.observation_space = spaces.Tuple((spaces.Discrete(15000),spaces.Discrete(15000),spaces.Discrete(15000),spaces.Discrete(15000)))
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

        if self.turns == len(self.data)-1:
            self.turns = 0

        if((self.turns % 60) == 0):
            self.episode_over = True

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
        """
        reward = 10

        if(action == 0): #allow
            if(self.data.values[self.turns,4] == "Malicious"): #true negative
                reward -= self.ob[0] #no of malicious packets going through
        else:
            if(self.data.values[self.turns,4] == "Benign"): #false positive
                reward -= 100 #no of benign packets getting dropped
        return reward

    def _get_new_state(self,action):
        """
        Get the next state from current state
        :return:
        """
        next_state = self.data.values[self.turns,:4]

        return next_state


    def _get_initial_state(self):
        return self.data.values[0,:4]

    def _seed(self):
        return



