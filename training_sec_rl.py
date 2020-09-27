import sys
sys.path.append("/home/anand/gym")
import gym
import pickle
import random
import numpy as np

from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from env.attack_sig_gym import Attack_Sig_Gym

env = gym.make('attack-sig-v0')
env.reset()
# action = 0
#
# # print x
# done = False
#
# while done == False:
#     observation, reward, done, info = env.step(action)
#     action = env.action_space.sample()
#     print(observation, action, reward)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(64,input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def act(self, state):
        if np.random.rand() <= self.epsilon:
           return env.action_space.sample()

        act_values = self.model.predict(state)
        return np.argmax(act_values[0])


    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states, targets_f = [], []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            # Filtering out states and targets for training
            states.append(state[0])
            targets_f.append(target_f[0])
        history = self.model.fit(np.array(states), np.array(targets_f), epochs=1, verbose=0)
        # Keeping track of loss
        #loss = history.history['loss'][0]
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        #return loss

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        #self.model.save_weights(name)
        self.model.save(name)

# Q, stats = qLearning(env, 1000)

EPISODES = 1000
print()
state_size = len(env.observation_space.spaces)
action_size = env.action_space.n
print("state and action sizes are:"+str(state_size)+","+str(action_size))
agent = DQNAgent(state_size, action_size)
# agent.load("./save/cartpole-dqn.h5")
done = False
batch_size =200
avg = 0

for e in range(EPISODES):
    state = env.reset()
    # state = env._get_initial_state()
    # #env.turns = 0
    # env.sum_rewards = 0.0
    done = False
    # env.episode_done = False

    state = np.reshape(state, [1, state_size])
    while not done:
        # env.render()
        action = agent.act(state)
        # print(action)
        next_state, reward, done, info = env.step(action)

        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print("episode: {}/{}, score: {}, e: {:.2f}"
                  .format(e, EPISODES, env.sum_rewards, agent.epsilon))
            avg += env.sum_rewards
            # agent.save("rl_model_v31")

            if len(agent.memory) > batch_size:
                #loss =
                agent.replay(batch_size)

            if(e % 500) == 0:
                agent.save("dqn_agent4")

            #best - 0.5 negative reward, state clipping, no reward clipping, epsilon_decay=0.995, mem=2000, batchsize = 60
            #dqn3 - 0.1 negative reward
            #dqn4 - 0.5 negative reward
            #dqn5 - batchsize = 200
            break



print("average score = ",(avg//EPISODES))
