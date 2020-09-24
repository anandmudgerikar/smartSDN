import sys
sys.path.append("/home/anand/gym")
import gym
import pickle
import keras
import numpy as np

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
reconstructed_model = keras.models.load_model("./dqn_agent2")

#action = dec_tree.predict()

#50 episodes

#starting with only malicious episodes
env.turns = 0

for i in range(100):
    sum_rewards = 0
    done = False
    observation = env._get_initial_state(500)


    while not done:
        #action = env.action_space.sample()

        #action = 1 - np.argmax(dec_tree.predict([env.ob]),axis=1)[0]
        #print(action)

        state = np.reshape(observation, [1, state_size])



        q_values = reconstructed_model.predict(state)
        # #print(q_values)
        action = np.argmax(q_values)
        observation, reward, done, info = env.step(action)
        # sec_viol = dec_tree.predict([[env.previous_5_loadsum[0]]])
        #
        # if (observation[0] == 0):
        #     sec_viol[0] = 'NULL'

        #print(observation,env.previous_5_loadsum[0],sec_viol[0])
        #print(reward)
        #print(observation,action, reward, sum_rewards)
        sum_rewards +=reward

    print("Episode ",i," : ",sum_rewards)

# for _ in range(1000):
#     #env.render()
#     env.step(env.action_space.sample()) # take a random action
# env.close()