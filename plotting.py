import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# data = pd.read_pickle("replay_buffer_wposreward.pickle")
#
# print(data.sample(100)[2])

#only security
# actions_benign_dqn = pd.read_pickle('actions_benign_dqn_1000')
# actions_mal_dqn = pd.read_pickle('actions_mal_dqn_1000')
# actions_mal_dqn = pd.read_pickle('actions_mal_dqn.pickle')
# actions_benign_dnn = pd.read_pickle('actions_mal_dnn.pickle')
#
# #plt.plot(range(61),actions_mal_dnn, label = "mal dnn")
# plt.plot(range(61),actions_mal_dqn, label = "Malicious flows")
# #plt.plot(range(61),actions_benign_dnn, label = "benign dnn")
# plt.plot(range(61),actions_benign_dqn, label = "Benign flows")

#malicious flows
# rewards_mal_dqn = pd.read_pickle('rewards_mal_dqn_1000.pickle')
# total_pckts_mal_dqn = pd.read_pickle('totpckts_mal_dqn_1000.pickle')
#
# rewards_mal_dqn = np.multiply(np.array(rewards_mal_dqn),-1)
# # print(rewards_mal_dqn)
#
# plt.bar(range(50),rewards_mal_dqn[500:550],label='Malicious bytes allowed',width=0.5,align='edge')
# plt.bar(range(50),total_pckts_mal_dqn[500:550],label='Total Malicious Bytes',color='y',width=0.5)
# plt.ylim(top=100000)

# #benign flows
# rewards_ben_dqn = pd.read_pickle('rewards_benign_dqn_1000.pickle')
# total_pckts_ben_dqn = pd.read_pickle('totpckts_benign_dqn_1000.pickle')
#
# # print(rewards_mal_dqn)
#
# plt.bar(range(50),rewards_ben_dqn[:50],label='Benign bytes allowed',width=0.5,align='edge')
# plt.bar(range(50),total_pckts_ben_dqn[:50],label='Total Benign Bytes',width=0.5,color='y')
#
# plt.legend()
# plt.xlabel('Episode')
# plt.ylabel('Bytes per Interval')
# plt.show()


# #load balancing
#actions_lb_wosec_mal = pd.read_pickle('actions_mal_dqn_lb_wosec.pickle')
actions_lb_wosec_ben = pd.read_pickle('actions_benign_dqn_lb_wosec.pickle')
#actions_lb_wsec_mal_upper = pd.read_pickle('actions_mal_dqn_lb_wsec0.9.pickle')
actions_lb_wsec_ben_upper = pd.read_pickle('actions_ben_dqn_lb_wsec0.9.pickle')
#
#plt.plot(range(61),actions_lb_wosec_mal, label = "Malicious flows (without security) d=0",color = 'r')
plt.plot(range(61),actions_lb_wosec_ben, label = "Benign flows (without security) d=0",color = 'r')
#plt.plot(range(61),actions_lb_wsec_mal_upper, label = "Malicious flows (with security) d=0.9",color='b')
plt.plot(range(61),actions_lb_wsec_ben_upper, label = "Benign flows (with security) d=0.9",color = 'b')
#
#plt.fill_between(range(61),actions_lb_wosec_mal,actions_lb_wsec_mal_upper,color='g')
plt.fill_between(range(61),actions_lb_wosec_ben,actions_lb_wsec_ben_upper,color='g')
#
plt.legend()
plt.xlabel('Time (seconds)')
plt.ylabel('Action %(pckts blocked)')
plt.show()

