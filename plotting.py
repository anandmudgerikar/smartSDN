import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_pickle("replay_buffer.pickle")

print(data.sample(1))

# actions_benign_dnn = pd.read_pickle('actions_benign_dnn.pickle')
# actions_benign_dqn = pd.read_pickle('actions_benign_dqn.pickle')
# actions_mal_dqn = pd.read_pickle('actions_mal_dqn.pickle')
# actions_mal_dnn = pd.read_pickle('actions_mal_dnn.pickle')
#
# plt.plot(range(60),actions_mal_dnn, label = "mal dnn")
# plt.plot(range(60),actions_mal_dqn, label = "mal dqn")
# plt.plot(range(60),actions_benign_dnn, label = "benign dnn")
# plt.plot(range(60),actions_benign_dqn, label = "benign dqn")
# plt.legend()
# plt.xlabel('Time (seconds)')
# plt.ylabel('Action %(pckts blocked)')
# plt.show()