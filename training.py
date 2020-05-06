import sys
sys.path.append("/home/anand/gym")
# sys.path.append("/home/anand/anaconda3/envs/mininet_backend/lib/python3.7/site-packages")
# sys.path.append("/home/anand/anaconda3/envs/mininet_backend/bin/python")
import gym
import numpy as np
import random

from env.sdn_gym import SDN_Gym

env = gym.make('sdn-v0')

from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

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
        model.add(Dense(5,input_dim=self.state_size, activation='relu'))
        model.add(Dense(10, activation='relu'))
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
        loss = history.history['loss'][0]
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss



    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

# Q, stats = qLearning(env, 1000)

EPISODES = 70

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
print("state and action sizes are:"+str(state_size)+","+str(action_size))
agent = DQNAgent(state_size, action_size)
# agent.load("./save/cartpole-dqn.h5")
done = False
batch_size =100

for e in range(EPISODES):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    for time in range(1000):
        # env.render()
        action = agent.act(state)
        # print(action)
        next_state, reward, done, info = env.step(action)

        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print("episode: {}/{}, score: {}, e: {:.2}"
                  .format(e, EPISODES, env.sum_rewards, agent.epsilon))
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
