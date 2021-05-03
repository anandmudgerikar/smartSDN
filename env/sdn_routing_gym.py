import gym
import random
from gym import error, spaces, utils
from gym.utils import seeding
import time
import pickle
import numpy as np

import sys
#sys.path.append("/usr/lib/python2.7/dist-packages/")

class SDN_Routing_Gym(gym.Env):

    def __init__(self):

        #action space: discrete 1-10 (choose between 10 paths predefined for each node pair)
        self.action_space = spaces.Discrete(10)

        #state space: box tuple consisting of flow parameters and network environment
        # flow state parameters:
        # all edge weights:
        high = np.array([self.x_threshold * 2,
                         np.finfo(np.float32).max,
                         self.theta_threshold_radians * 2,
                         np.finfo(np.float32).max],
                        dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        #miscellaneous params
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

        return self.ob, self.reward, self.episode_over, {}

    def _reset(self):
        """
        Reset the environment and supply a new state for initial state
        :return:
        """

        return self.ob

    def _take_action(self, action):
        """
        Take an action correpsonding to action_index in the current state
        :param action_index:
        :return:
        """

    def _take_action_sec(self, action):
        """
        Take an action correpsonding to action_index in the current state
        :param action_index:
        :return:
        """

    def _get_reward(self,action):
        """
        Get reward for the action taken in the current state
        :return:
        """

        return reward

    # def _get_reward_sec(action):
    #
    #
    #     return reward

    def _get_new_state(self,action):
        """
        Get the next state from current state
        :return:
        """
        return next_state


    def _get_initial_state(self):
        return (0, 0, 0, 0, 0)

    def _seed(self):
        return



