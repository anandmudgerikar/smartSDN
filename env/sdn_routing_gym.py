import gym
import random
from gym import error, spaces, utils
from gym.utils import seeding
import time
import pickle
import numpy as np
import pandas as pd
from routing.inet_topology_parser import inet_topology_parser
import random

import sys
#sys.path.append("/usr/lib/python2.7/dist-packages/")

class SDN_Routing_Gym(gym.Env):

    def __init__(self):

        #action space: discrete 1-10 (choose between 10 paths predefined for each node pair)
        self.action_space = spaces.Discrete(10)

        #state space: box tuple consisting of flow parameters and network environment
        # flow state parameters: Dst Port, Protocol, Flow Duration,	Tot Fwd Pkts, Tot Bwd Pkts, TotLen Fwd Pkts, TotLen Bwd Pkts,
        # Fwd Pkt Len Max, Fwd Pkt Len Min,	Fwd Pkt Len Mean, Fwd Pkt Len Std, Bwd Pkt Len Max,	Bwd Pkt Len, Min Bwd Pkt Len ,
        # Mean	Bwd Pkt Len, Std Flow Byts/s, Flow Pkts/s, Flow IAT Mean, Flow IAT Std, Flow IAT Max, Flow IAT Min, Fwd IAT Tot,
        # Fwd IAT Mean,	Fwd IAT Std, Fwd IAT Max, Fwd IAT Min, Bwd IAT Tot, Bwd IAT Mean, Bwd IAT Std, Bwd IAT Max, Bwd IAT Min,
        # Fwd PSH Flags, Bwd PSH Flags,	Fwd URG Flags, Bwd URG Flags, Fwd Header Len, Bwd Header Len, Fwd Pkts/s, Bwd Pkts/s, Pkt Len Min
        # Pkt Len Max, Pkt Len Mean, Pkt Len Std, Pkt Len Var, FIN Flag Cnt, SYN Flag Cnt, RST Flag Cnt, PSH Flag Cnt, ACK Flag Cnt,
        # URG Flag Cnt,	CWE Flag Count,	ECE Flag Cnt, Down/Up Ratio, Pkt Size Avg, Fwd Seg Size, Avg Bwd Seg Size, Avg Fwd Byts/b, Avg Fwd Pkts/b,
        # Avg Fwd Blk Rate, Avg	Bwd Byts/b, Avg	Bwd Pkts/b, Avg	Bwd Blk Rate, Avg Subflow Fwd Pkts,	Subflow Fwd Byts, Subflow Bwd Pkts,
        # Subflow Bwd Byts, Init Fwd Win Byts, Init Bwd Win Byts, Fwd Act Data Pkts, Fwd Seg Size Min, Active Mean,	Active Std, Active Max,
        # Active Min, Idle Mean, Idle Std, Idle Max, Idle Min, Label
        high = np.array([np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max,
                         np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max,
                         np.finfo(np.float32).max, np.finfo(np.float32).max,
                         np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max,
                         np.finfo(np.float32).max, np.finfo(np.float32).max,np.finfo(np.float32).max, np.finfo(np.float32).max,np.finfo(np.float32).max,
                         np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max,
                         np.finfo(np.float32).max, np.finfo(np.float32).max,
                         np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max,
                         np.finfo(np.float32).max, np.finfo(np.float32).max,
                         np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max,
                         np.finfo(np.float32).max, np.finfo(np.float32).max,
                         np.finfo(np.float32).max, np.finfo(np.float32).max,
                         np.finfo(np.float32).max, np.finfo(np.float32).max,np.finfo(np.float32).max,
                         1,1,1,1,1,1,1,1,
                         np.finfo(np.float32).max,
                         np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max,
                         np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max,
                         np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max,
                         np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max,np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max,
                         np.finfo(np.float32).max, np.finfo(np.float32).max,#Nodes
                         5, #label
                         # end of flow based params
                         np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max,np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max,
                         np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max,np.finfo(np.float32).max,np.finfo(np.float32).max,np.finfo(np.float32).max], #path params
                        dtype=np.float32)

        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        #not well defined observation space, if required use above for more nuanced state/obs space
        #self.observation_space = spaces.Tuple((spaces.Discrete(self.data.shape[0]),))

        #miscellaneous params
        self.episode_over = False
        self.turns = 0
        self.sum_rewards = 0.0
        self.risk_param = 0.5
        self.sum_ddos_bytes = 0.0
        self.sum_latency_reward = 0.0

        self.data = pd.read_csv("/home/anand/PycharmProjects/mininet_backend/pcaps/csv/Friday_botnet.csv", sep=',',header=0)
        feature_cols = ["Dst Port", "Protocol","Flow Duration",	"Tot Fwd Pkts","Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts",
        "Fwd Pkt Len Max", "Fwd Pkt Len Min","Fwd Pkt Len Mean","Fwd Pkt Len Std","Bwd Pkt Len Max","Bwd Pkt Len Min", "Bwd Pkt Len Mean",
        "Bwd Pkt Len Std","Flow Byts/s", "Flow Pkts/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Tot",
        "Fwd IAT Mean",	"Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Tot", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
        "Fwd PSH Flags", "Bwd PSH Flags","Fwd URG Flags", "Bwd URG Flags", "Fwd Header Len", "Bwd Header Len", "Fwd Pkts/s", "Bwd Pkts/s", "Pkt Len Min",
        "Pkt Len Max", "Pkt Len Mean", "Pkt Len Std", "Pkt Len Var", "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt", "ACK Flag Cnt",
        "URG Flag Cnt",	"CWE Flag Count","ECE Flag Cnt", "Down/Up Ratio", "Pkt Size Avg", "Fwd Seg Size Avg", "Bwd Seg Size Avg", "Fwd Byts/b Avg", "Fwd Pkts/b Avg",
        "Fwd Blk Rate Avg", "Bwd Byts/b Avg", "Bwd Pkts/b Avg", "Bwd Blk Rate Avg", "Subflow Fwd Pkts",	"Subflow Fwd Byts", "Subflow Bwd Pkts",
        "Subflow Bwd Byts", "Init Fwd Win Byts", "Init Bwd Win Byts", "Fwd Act Data Pkts", "Fwd Seg Size Min", "Active Mean","Active Std", "Active Max",
        "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min", "Label"]
        self.data = self.data[feature_cols]  # Features

        #Dealing with categorical label column: 0-Benign, 1-Botnet
        self.data["Label"] = self.data["Label"].astype('category')
        self.data["Label"] = self.data["Label"].cat.codes
        self.flow_params_len = self.data.shape[1]
        # print(self.data)
        # print(self.flow_params_len)

        self.inet_parser = inet_topology_parser()
        self.ob = self._get_initial_state()


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

        # self.reward = self._get_reward(action)
        # for jarvis with security
        self.reward = self._get_reward(action)
        self.sum_rewards += self.reward


        if self.turns > 10:
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
        self.sum_ddos_bytes = 0.0
        self.sum_latency_reward = 0.0
        return self.ob

    def _take_action(self, action):
        """
        Take an action correpsonding to action_index in the current state
        :param action_index:
        :return:
        """
        #random policy
        #return random.randint(0,9)

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
        func_reward = self.ob[self.flow_params_len + action] * -1
        risk_reward = self.ob[self.flow_params_len - 1] * -10000  # 0 if flow is benign, 1 if flow is malicious
        # print("flow is malicious : ",self.ob[self.flow_params_len-1],risk_reward)


        if "DDoS" in self.sec_services[action]:
            risk_reward = 0

        # maintaining individual rewards for plotting and testing
        self.sum_latency_reward += func_reward
        self.sum_ddos_bytes += risk_reward

        return func_reward #negative weight for path weights

    def _get_reward_sec(self,action):

        func_reward = self.ob[self.flow_params_len+action]*-1
        risk_reward = self.ob[self.flow_params_len-1]*-10000 # 0 if flow is benign, 1 if flow is malicious

        #print("flow is malicious : ",self.ob[self.flow_params_len-1],risk_reward)

        #print(self.sec_services)
        #print(action)
        if "DDoS" in self.sec_services[action]:
            risk_reward = 0

        #maintaining individual rewards for plotting and testing
        self.sum_latency_reward += func_reward
        self.sum_ddos_bytes += risk_reward

        return func_reward*(1-self.risk_param) + risk_reward*self.risk_param

    def _get_new_state(self,action):
        """
        Get the next state from current state
        :return:
        """
        flow_params = self.data.sample(1)
        #print(flow_params)
        node1 = 0
        node2 = 0

        while node1 == node2:
            node1 = random.randint(0,self.inet_parser.no_of_nodes)
            node2 = random.randint(0,self.inet_parser.no_of_nodes)

        paths,sec_services = self.inet_parser.find_paths(node1, node2)
        next_state = np.concatenate((flow_params.values[0],[node1,node2],np.array(paths)),axis=0)
        self.sec_services = sec_services
        #print(next_state)
        return next_state

    def _get_initial_state(self):

        flow_params = self.data.sample(1)
        #print(flow_params)
        node1 = 0
        node2 = 0

        while node1 == node2:
            node1 = random.randint(0,self.inet_parser.no_of_nodes)
            node2 = random.randint(0,self.inet_parser.no_of_nodes)

        paths,sec_services = self.inet_parser.find_paths(node1, node2)
        state = np.concatenate((flow_params.values[0],[node1,node2],np.array(paths)),axis=0)
        self.sec_services = sec_services
        #print(state)
        return state

    def _seed(self):
        return



