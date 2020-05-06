import gym
import random
from gym import error, spaces, utils
from gym.utils import seeding
import time

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.util import dumpNodeConnections
from mininet.log import setLogLevel
from mininet.net import Mininet, CLI
from mininet.node import OVSKernelSwitch, Host
from mininet.link import TCLink, Link
from mininet.log import setLogLevel, info
from mininet_setup import Mininet_Backend

class SDN_Gym(gym.Env):

    def __init__(self):

        self.action_space = spaces.MultiDiscrete([3, 3, 3, 3, 3])
        self.observation_space = spaces.MultiDiscrete([100, 100, 100, 100, 100])
        self.ob = self._get_initial_state()
        self.episode_over = False
        self.turns = 0
        self.sum_rewards = 0.0

        #starting mininet setup
        setLogLevel('info')
        self.mn_backend = Mininet_Backend()
        self.curr_net = self.mn_backend.startTest()
        self.mn_backend.replay_flows(self.curr_net)

        #thresholds
        self.server_thresh = 10000
        self.sla = 200

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
        #each step is taken after 60 seconds
        time.sleep(10)
        #perform action
        self._take_action(action)

        self.turns += 1
        self.ob = self._get_new_state()
        self.reward = self._get_reward()

        if self.turns > 1000:
            self.episode_over = True

        return self.ob, self.reward, self.episode_over, {}

    def _reset(self):
        """
        Reset the environment and supply a new state for initial state
        :return:
        """
        self.turns = 0
        self.ob = self._get_initial_state()
        self.episode_over = False
        self.sum_rewards = 0.0
        return self.ob

    def _take_action(self, action):
        """
        Take an action correpsonding to action_index in the current state
        :param action_index:
        :return:
        """
        if(action[0] == 1): #queue flow
            self.curr_net.getNodeByName('s1').cmd('ovs-ofctl --protocols=OpenFlow13 mod-flows s1 idle_timeout=1000,priority=60000,nw_src=192.168.10.19,nw_dst=192.168.10.50,ip,tp_dst=21,actions=output:3')
        elif(action[0] == 2): #send flow to IDS
            self.curr_net.getNodeByName('s1').cmd('ovs-ofctl --protocols=OpenFlow13 mod-flows s1 idle_timeout=1000,priority=60000,nw_src=192.168.10.19,nw_dst=192.168.10.50,ip,tp_dst=21,actions=output:4')
        else: #normal forward to server
            self.curr_net.getNodeByName('s1').cmd('ovs-ofctl --protocols=OpenFlow13 mod-flows s1 idle_timeout=1000,priority=60000,nw_src=192.168.10.19,nw_dst=192.168.10.50,ip,tp_dst=21,actions=output:2')


    def _get_reward(self):
        """
        Get reward for the action taken in the current state
        :return:
        """
        reward = 10
        total_load = self.ob[0] + self.ob[1] + self.ob[2] + self.ob[3] + self.ob[4]

        #if server is overloaded
        if (total_load > self.server_thresh):
            reward = reward - 50

        #if sla is being breached,
        if(self.ob[0] < self.sla and self.ob[0] > 0):
            reward = reward - 10

        #do sla for other users

        #security policy (from partial signatures)

        return reward

    def _get_new_state(self):
        """
        Get the next state from current state
        :return:
        """
        curr_state = self.ob
        load = self.mn_backend.get_serverload(self.curr_net)

        next_state = (load[0] - curr_state[0],load[1] - curr_state[1],load[2] - curr_state[2],load[3] - curr_state[3],load[4] - curr_state[4])
        return next_state

    def _get_initial_state(self):
        return (0, 0, 0, 0, 0)

    def _seed(self):
        return



