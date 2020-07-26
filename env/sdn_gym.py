import gym
import random
from gym import error, spaces, utils
from gym.utils import seeding
import time
import pickle
import numpy as np

import sys
#sys.path.append("/usr/lib/python2.7/dist-packages/")
sys.path.append("/home/anand/mininet/mininet")

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

        self.action_space = spaces.Discrete(2) #for security only testing n=2, allow/drop, for jarvis n=3, allow/drop/queue
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
        self.previous_load = [0,0,0,0,0]
        self.queue_load = [0,0,0,0,0]

        self.previous_5_loadsum = [0,0,0,0,0]
        self.previous_5_counter = 0

        #loading partial signature dec tree model
        filename = 'partial_sig.sav'
        self.dec_tree = pickle.load(open(filename, 'rb'))

        #sleeping as flows start after 300 seconds in replays
        #time.sleep(300)

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
        time.sleep(2)
        #perform action
        #self._take_action(action) #for jarvis
        self._take_action_sec(action) #for sec analysis

        self.turns += 1
        self.ob = self._get_new_state(action)

        #self.reward = self._get_reward(action) #for jarvis
        self.reward = self._get_reward_sec(action)

        self.sum_rewards += self.reward

        if(self.previous_5_counter >=5):
            self.previous_5_counter = 0
            self.previous_5_loadsum = [0,0,0,0,0]

        if self.turns > 100:
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
        self.previous_5_loadsum = 0
        self.previous_5_counter = 0
        self.previous_load = [0, 0, 0, 0, 0]

        #stopping mininet
        self.mn_backend.stop_test(self.curr_net)

        # starting mininet setup
        self.curr_net = self.mn_backend.startTest()
        self.mn_backend.replay_flows(self.curr_net)

        return self.ob

    def _take_action(self, action):
        """
        Take an action correpsonding to action_index in the current state
        :param action_index:
        :return:
        """
        if(action == 1): #queue flow
            self.curr_net.getNodeByName('s1').cmd('ovs-ofctl --protocols=OpenFlow13 mod-flows s1 idle_timeout=1000,priority=60000,nw_src=192.168.10.19,nw_dst=192.168.10.50,ip,actions=output:3')
        elif(action == 2): #send flow to IDS
            self.curr_net.getNodeByName('s1').cmd('ovs-ofctl --protocols=OpenFlow13 mod-flows s1 idle_timeout=1000,priority=60000,nw_src=192.168.10.19,nw_dst=192.168.10.50,ip,actions=output:4')
        else: #normal forward to server
            self.curr_net.getNodeByName('s1').cmd('ovs-ofctl --protocols=OpenFlow13 mod-flows s1 idle_timeout=1000,priority=60000,nw_src=192.168.10.19,nw_dst=192.168.10.50,ip,actions=output:2')

    def _take_action_sec(self, action):
        """
        Take an action correpsonding to action_index in the current state
        :param action_index:
        :return:
        """
        if (action == 1):  # drop
            self.curr_net.getNodeByName('s1').cmd(
                'ovs-ofctl --protocols=OpenFlow13 mod-flows s1 idle_timeout=1000,priority=60000,nw_src=172.16.0.1,nw_dst=192.168.10.50,ip,actions=output:3')
        else:  # allow
            self.curr_net.getNodeByName('s1').cmd(
                'ovs-ofctl --protocols=OpenFlow13 mod-flows s1 idle_timeout=1000,priority=60000,nw_src=172.16.0.1,nw_dst=192.168.10.50,ip,actions=output:2')

    def _get_reward(self,action):
        """
        Get reward for the action taken in the current state
        :return:
        """
        #collecting parameters from flow
        reward = 10
        total_load = self.ob[0] + self.ob[1] + self.ob[2] + self.ob[3] + self.ob[4]
        flow_rate_user = (self.previous_5_loadsum[0]/10) #2 second intervals
        # sec_viol = self.dec_tree.predict(np.array([flow_rate_user,flow_rate_user]).reshape(1,-1))
        sec_viol = self.dec_tree.predict([[flow_rate_user]])

        #dealing with zero load case
        if(flow_rate_user == 0):
            sec_viol[0] = 'NULL'

        #print(sec_viol[0], flow_rate_user)

        if(action != 2): #not being sent to IDS.. not security violation detected
            #if server is overloaded
            if (total_load > self.server_thresh):
                if(action == 0):
                    reward = reward - 50 #not queued
                else:
                    reward = reward + 50 #correctly queued

            #if sla is being breached,
            if(self.ob[0] < self.sla and self.ob[0] > 0):
                if(action == 1):
                    reward = reward - 50 #incorrect queing causing low user rate
                else:
                    reward = reward + 50 #forwarding to improve user rate
            #do sla for other users

            #security policy violation but not being sent to IDS (true negative)
            if(sec_viol[0] == 'Malicious'): #Attack partial signature match
                reward = reward - 500
        else:
            if (sec_viol[0] == 'Malicious'):  # Attack partial signature match
                reward = reward + 500 #correctly sending to verify at IDS
            else:
                reward = reward - 500 #false positive

        # if(self.queue_load[0] > 0): #existing queue load
        #     if(self.ob[0] == 0): #no current load on server
        #         if(action == 0): #unque
        #             reward += 50
        #         elif(action == 1):
        #             reward -= 50
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
        curr_state = self.ob
        load = self.mn_backend.get_serverload(self.curr_net)
        #print(load)

        #getting next state
        next_state = (load[0] - self.previous_load[0],load[1] - self.previous_load[1],load[2] - self.previous_load[2],load[3] - self.previous_load[3],load[4] - self.previous_load[4])
        self.previous_load = load

        #for calculating rate
        self.previous_5_loadsum = np.add(self.previous_5_loadsum,next_state)
        self.previous_5_counter +=1

        #for queue calculating
        # if(action == 1): #queing
        #     self.queue_load[0] += next_state[0]
        # elif(action == 0): #normal forwarding
        #     if(next_state[0] == 0): #no current load
        #         self.queue_load[0] -= self.server_thresh
        #         self.queue_load[0] = max(self.queue_load[0],0) #removing negative load
        #     else:
        #         leftover = max(next_state[0]-self.server_thresh,0)
        #         self.queue_load += leftover

        return next_state


    def _get_initial_state(self):
        return (0, 0, 0, 0, 0)

    def _seed(self):
        return



