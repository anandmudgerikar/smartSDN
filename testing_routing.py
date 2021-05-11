import sys
sys.path.append("/home/anand/gym")
sys.path.append('/home/anand')
import gym
import pickle
import numpy as np

from env.sdn_routing_gym import SDN_Routing_Gym
from training_routing import DQNAgent

env = gym.make('sdn-routing-v0')

env.reset()
action = 0
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
print("state and action sizes are:"+str(state_size)+","+str(action_size))
agent = DQNAgent(state_size, action_size)
# print x

for i in range(100):

    state = env.reset()
    state = np.reshape(state, [1, state_size])
    for time in range(1000):

        action = agent.act(state)
        # print(action)

        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        state = next_state

        if done:
            print("episode over")
            print("DDoS Bytes Loss:",env.sum_ddos_bytes)
            print("Latency Loss:",env.sum_latency_reward)

import pandas as pd
import numpy as np
from keras import preprocessing
from scipy.interpolate import interpolate
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from tensorflow import keras
from numpy.random import randn
import matplotlib.pyplot as plt
#
# data = pd.read_pickle("joint_3_domains_matrix_D.pickle")
# print("Dataset Length: ", len(data))
# print(data.sample(1))
#
# data = pd.read_pickle("SD_3_domains.pickle")
# print("Dataset Length: ", len(data))
# print(data.sample(1))

# X = data.sample(5000)
#
# for j in range(4):
#     for i in range(5000):
#         X_temp = np.concatenate((X[0][i], X[1][i], X[2][i], X[3][i], X[4][i]), axis=None)
#         if len(X[j][i]) != 18:
#             print("error",j,X[j][i])

#
#
# #convert to
# #
# td = pd.read_pickle("aug_data2.pickle")
#
# sd = pd.read_pickle("SD.pickle")
#
# X = td.sample(1)
# # sc0 = preprocessing.StandardScaler().fit(X[0][:])
# # sc1 = preprocessing.StandardScaler().fit(X[1][:])
# # sc2 = preprocessing.StandardScaler().fit(X[2][:])
# # sc3 = preprocessing.StandardScaler().fit(X[3][:])
#
# n = 10
# X = td.sample(n)
# Y = sd.sample(n)
# Z = data.sample(n)
# #
# print(X)
# # print(Y)
#print(Z)

# model = keras.models.load_model('./models/generator_model_1001')
#
# x_input = randn(49 * n)
# # reshape into a batch of inputs for the network
# x_input = x_input.reshape(n, 49)
#
# X_gan = model.predict(x_input)
# states = []
# rewards = []
# actions = []
# states_ = []
# dones = []
#
# for i in range(n):
#     states.append(sc0.inverse_transform(X_gan[i][0:12]))
#     actions.append(sc1.inverse_transform(X_gan[i][12:24]))
#     rewards.append(sc2.inverse_transform(X_gan[i][24:36]))
#     states_.append(sc3.inverse_transform(X_gan[i][36:48]))
#     dones.append(True if X_gan[i][48] >= 0 else False)
#
# #print(X_gan)
# print(states,actions,rewards,states_,dones)
# print(X)
#



#
# print(X_gan)
# print(X)
# print(Y)

#X = np.concatenate((X[0],X[1],X[2],X[3],X[4]),axis=0)

# test= []
# test_norm = []
# #print(X[0][:])
# X0 = preprocessing.scale(X[0][:])
# X1 = preprocessing.scale(X[1][:])
# X2 = preprocessing.scale(X[2][:])
# X3 = preprocessing.scale(X[3][:])
#
# #X = td.sample(2)
# #print(X_scaled)
#
# #normalizing features
#
# for i in range(2):
#     X_test = np.concatenate((X[0][i],X[1][i],X[2][i],X[3][i],X[4][i]),axis=None)
#     test = np.concatenate((test,X_test),axis=0)
#
# # print(X[1][:])
# # print(X1)
#
# for i in range(2):
#     X_test = np.concatenate((X0[i],X1[i],X2[i],X3[i],X[4][i]),axis=None)
#     test_norm = np.concatenate((test_norm,X_test),axis=0)
#
# test = np.reshape(test,(-1,49))
# test_norm = np.reshape(test_norm,(-1,49))
#
# print(test)
# print(test_norm)
#
#
#
# #
# # #X = np.reshape(X,(-1,49))
# #
# # print(X)
# # print(X.shape)
#
#
#
# print(data.sample(1))
#
# x,y_aug = pd.read_pickle('ally_augmented_explorations.pickle')
# _,y_real = pd.read_pickle('ally_real_explorations.pickle')
# _,y_one = pd.read_pickle('ally_ensemble_explorations.pickle')
# _,y_sametime = pd.read_pickle('one_explorations_3domain_same_time.pickle')

# x_new = np.linspace(1, 130, 500)
# a_BSpline = interpolate.make_interp_spline(x, y_real)
# y_real_new = a_BSpline(x_new)
# a_BSpline = interpolate.make_interp_spline(x, y_aug)
# y_aug_new = a_BSpline(x_new)

# plt.plot(x[20:130], y_real[20:130], label="Real Explorations (10000 time slots)")
# plt.plot(x[20:130], y_aug[20:130], label="Augmented Explorations (One by one scenario )(100 time slots)")
# plt.plot(x[20:130], y_one[20:130], label="Limited Real Explorations (100 time slots)")
# #plt.plot(x[0:200], y_sametime[0:200], label="Augmented Explorations (Same Time scenario) (100 time slots)")
# plt.xlabel('Episode')
# plt.ylabel('Total Reward')
#
# plt.legend()
# plt.show()
