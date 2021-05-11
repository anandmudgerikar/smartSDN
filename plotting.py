import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# data = pd.read_pickle("replay_buffer_wposreward.pickle")
#
# print(data.sample(100)[2])

#only security
# actions_benign_dqn = pd.read_pickle('actions_benign_dqn_1000')
# actions_mal_dnn = pd.read_pickle('actions_mal_dnn.pickle')
# actions_mal_dqn = pd.read_pickle('actions_mal_dqn_1000')
# actions_benign_dnn = pd.read_pickle('actions_benign_dnn.pickle')
#
# plt.plot(range(60),actions_mal_dnn, label = "Malicious flows: DNN Softmax")
# plt.plot(range(60),actions_mal_dqn[:60], label = "Malicious flows: DQN")
# plt.plot(range(60),actions_benign_dnn, label = "Benign flows: DNN Softmax")
# plt.plot(range(60),actions_benign_dqn[:60], label = "Benign flows: DQN")

#malicious flows
# rewards_mal_dqn = pd.read_pickle('rewards_mal_dqn_1000.pickle')
# total_pckts_mal_dqn = pd.read_pickle('totpckts_mal_dqn_1000.pickle')
#
x_sec, y_sec,y_latency_loss_sec,y_ddos_bytes_sec = pd.read_pickle("rewards_rl_with_sec.pickle")
x_wosec, y_wosec,y_latency_loss_wosec,y_ddos_bytes_wosec = pd.read_pickle("rewards_rl_without_sec.pickle")

# plt.plot(x_sec,y_ddos_bytes_sec/np.min(y_ddos_bytes_wosec), label = "DDoS detection Loss with Jarvis")
# plt.plot(x_sec,y_ddos_bytes_wosec/np.min(y_ddos_bytes_wosec), label = "DDoS detection Loss without Jarvis")

y_latency_loss_wosec = np.array(y_latency_loss_wosec)
y_latency_loss_sec = np.array(y_latency_loss_sec)

y_latency_loss_wosec[y_latency_loss_wosec == -np.inf] = -40000
y_latency_loss_sec[y_latency_loss_sec == -np.inf] = -40000

print(y_latency_loss_wosec)
print(y_latency_loss_sec)

plt.plot(x_sec,y_latency_loss_sec/np.min(y_latency_loss_sec), label = "Latency Loss with Jarvis")
plt.plot(x_sec,y_latency_loss_wosec/np.min(y_latency_loss_sec), label = "Latency Loss without Jarvis")

plt.legend()
plt.xlabel('Episode')
plt.ylabel('Latency Loss')
plt.show()

# plt.plot(range(60),actions_mal_dqn[:60], label = "Malicious flows: DQN")

# rewards_mal_dqn = np.multiply(np.array(rewards_mal_dqn),-1)
# # print(rewards_mal_dqn)
#
# plt.bar(range(50),rewards_mal_dqn[500:550]/np.max(total_pckts_mal_dqn[500:550]),label='Malicious bytes allowed',width=0.5,align='edge')
# #plt.bar(range(50),total_pckts_mal_dqn[500:550]/(total_pckts_mal_dqn[500:550]),label='Total Malicious Bytes',color='y',width=0.5)
#plt.ylim(top=100000)

# #benign flows
# rewards_ben_dqn = pd.read_pickle('rewards_benign_dqn_1000.pickle')
# total_pckts_ben_dqn = pd.read_pickle('totpckts_benign_dqn_1000.pickle')
#
# # print(rewards_mal_dqn)
#
# plt.bar(range(50),rewards_ben_dqn[:50]/np.max(total_pckts_ben_dqn[:50]),label='Benign bytes allowed',width=0.5,align='edge')
# plt.bar(range(50),total_pckts_ben_dqn[:50]/np.max(total_pckts_ben_dqn[:50]),label='Total Benign Bytes',width=0.5,color='y')

# plt.legend()
# plt.xlabel('Episode')
# plt.ylabel('Bytes per Interval (scaled to 1)')
# plt.show()


# #load balancing
# #actions_lb_wosec_mal = pd.read_pickle('actions_mal_dqn_lb_wosec.pickle')
# actions_lb_wosec_ben = pd.read_pickle('actions_benign_dqn_lb_wosec.pickle')
# actions_lb_ben_dnn = pd.read_pickle('actions_ben_dnn_lb_wsec0.9')
# #actions_lb_mal_dnn = pd.read_pickle('actions_mal_dnn_lb_wsec0.9')
# #actions_lb_wsec_mal_upper = pd.read_pickle('actions_mal_dqn_lb_wsec0.9.pickle')
# actions_lb_wsec_ben_upper = pd.read_pickle('actions_ben_dqn_lb_wsec0.9.pickle')
# #
# #plt.plot(range(61),actions_lb_wosec_mal, label = "Malicious flows: DQN based Signatures, d=0",color = 'r')
# plt.plot(range(61),actions_lb_wosec_ben, label = "Benign flows: DQN based Signatures (d=0)",color = 'r')
# #plt.plot(range(61),actions_lb_wsec_mal_upper, label = "Malicious flows: DQN based Signatures, d=0.9",color='b')
# plt.plot(range(61),actions_lb_wsec_ben_upper, label = "Benign flows: DQN based Signatures (d=0.9)",color = 'b')
# plt.plot(range(61),actions_lb_ben_dnn, label = "Malicious flows: DNN softmax based Signatures, d=0.9",color = 'y')
# #
# plt.fill_between(range(61),10,actions_lb_wsec_ben_upper,color='b',alpha=0.2,label = "Safe Action Space")
# plt.fill_between(range(61),actions_lb_wosec_ben,actions_lb_wsec_ben_upper,color='g',alpha=0.2,label = "Region of Interest")
# plt.fill_between(range(61),actions_lb_wosec_ben,0,color='r',alpha=0.2,label = "Unsafe Action Space")
# #plt.fill_between(range(61),actions_lb_wosec_ben,actions_lb_wsec_ben_upper,color='g',alpha=0.2)
# #
# plt.rcParams.update({'font.size': 12})
# plt.rc('axes', titlesize=12)
# plt.legend()
# plt.xlabel('Time (seconds)')
# plt.ylabel('Action %(pckts blocked)')
# plt.show()

