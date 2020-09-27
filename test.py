import sys
sys.path.append("/home/anand/gym")
import gym
import pickle
import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from env.sdn_gym import SDN_Gym

# env = gym.make('sdn-v0')
env = gym.make('attack-sig-v0')

env.reset()
action = env.action_space.sample()
state_size = len(env.observation_space.spaces)
#print(action)

filename = './sec_anal/rforest.sav'
dec_tree = pickle.load(open(filename, 'rb'))
# print x
done = False
sum_rewards = 0
reconstructed_model = keras.models.load_model("./dqn_agent4")

dnn_model = keras.models.load_model("./sec_anal/dnn_model")

#action = dec_tree.predict()
actions = [0]*60
rewards = [0]*1000
#50 episodes

#starting with only malicious episodes
env.turns = 0

for i in range(1000):
    env.reset()
    sum_rewards = 0
    done = False
    observation = env._get_initial_state(19000)
    env.ob = observation
    interval = 0

    while not done:
        #action = env.action_space.sample()

        #action = 1 - np.argmax(dec_tree.predict([env.ob]),axis=1)[0]
        #print(action)

        state = np.reshape(observation, [1, state_size])
        #state = np.divide(state, np.array([500, 20000, 500, 20000]))


        # q_values = reconstructed_model.predict(state)
        # # #print(q_values)
        # action = np.argmax(q_values)



        state = np.multiply(state, np.array([500, 20000, 500, 20000]))
        q_values = dnn_model.predict(state)

        # action = np.argmax(q_values)
        action = np.ceil((q_values[0][1] * 10))



        observation, reward, done, info = env.step(action)
        # if(reward < 0):
        #     print(q_values, action, interval, reward, env.ob)
        #print(action, interval,q_values)
        actions[interval] += action

        # sec_viol = dec_tree.predict([[env.previous_5_loadsum[0]]])
        #
        # if (observation[0] == 0):
        #     sec_viol[0] = 'NULL'

        #print(observation,env.previous_5_loadsum[0],sec_viol[0])
        #print(reward)
        #print(observation,action, reward, sum_rewards)
        #reward *=500
        sum_rewards +=reward
        interval +=1


    rewards[i] = (sum_rewards/interval)
    print("Episode ", i, " : ", rewards[i])
    #filling up rest of the intervals with average
    for j in range(interval,60):
        actions[j] += 10



actions = np.divide(np.array(actions),1000)

with open("actions_mal_dnn.pickle", "wb") as fp:
    pickle.dump((actions), fp)

with open("rewards_mal_dnn.pickle", "wb") as fp:
    pickle.dump((rewards), fp)

actions_benign_dnn = pd.read_pickle('actions_benign_dnn.pickle')
actions_benign_dqn = pd.read_pickle('actions_benign_dqn.pickle')
actions_mal_dqn = pd.read_pickle('actions_mal_dqn.pickle')

plt.plot(range(60),actions, label = "mal dnn")
plt.plot(range(60),actions_mal_dqn, label = "mal dqn")
plt.plot(range(60),actions_benign_dnn, label = "benign dnn")
plt.plot(range(60),actions_benign_dqn, label = "benign dqn")
plt.legend()
plt.xlabel('Time (seconds)')
plt.ylabel('Action %(pckts blocked)')
#plt.plot(range(100),rewards, label = "rewards")
#plt.plot(x,yq_values[:,1], label = "mal")
#plt.plot(x,y_test[:]*100)

plt.show()
# for _ in range(1000):
#     #env.render()
#     env.step(env.action_space.sample()) # take a random action
# env.close()