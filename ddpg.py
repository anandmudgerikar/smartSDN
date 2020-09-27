import pickle
import matplotlib.pyplot as plt
import numpy as np
from actor import Actor
from critic import Critic
import pandas as pd
from replay_buffer import ReplayBuffer
import run_policy as rp

import sys
sys.path.append("/home/anand/gym")


EPISODES = 1000
num_users = 10
no_updates = 100
#regularization loss weights
w_loss1 = 0.7
w_grads = 0.3




class DDPG:
    """ Deep Deterministic Policy Gradient (DDPG) Helper Class
    """

    def __init__(self, act_dim, env_dim, act_range, gamma = 0.99, lr = 0.00005, tau = 0.001, batchsize = 100):
        """ Initialization
        """
        # Environment parameters
        self.act_dim = act_dim
        self.act_range = act_range
        self.env_dim =  env_dim
        self.gamma = gamma
        self.lr = lr
        # Create actor and critic networks
        self.actor = Actor(self.env_dim, act_dim, act_range, 0.1 * lr, tau)
        self.critic = Critic(self.env_dim, act_dim, lr, tau)
        self.batch_size = batchsize

        #data samples:
        self.data = pd.read_pickle("replay_buffer.pickle")
        #s, a, r, _, _ = self.data.pull_by_index(0,batchsize)

        # #buffer to store results
        # self.x = []
        # self.y = []

    # def plot_graphs(self):
    #     with open("aug_exploration3.pickle", "wb") as fp:
    #         pickle.dump((self.x,self.y), fp)
    #     plt.plot(self.x, self.y, label="Limited Exploration (100 timeslots)")
    #     plt.show()

    def policy_action(self, s):
        """ Use the actor to predict value
        """
        s = np.reshape(s,(40,))
        return self.actor.predict(s)[0]

    def bellman(self, rewards, q_values, dones):
        """ Use the Bellman Equation to compute the critic target
        """
        critic_target = np.asarray(q_values)
        for i in range(q_values.shape[0]):
            if dones[i]:
                critic_target[i] = rewards[i]
            else:
                critic_target[i] = rewards[i] + self.gamma * q_values[i]
        return critic_target

    #since we are training completely offline in current testing, we ignore this
    # def memorize(self, state, action, reward, done, new_state):
    #     """ Store experience in memory buffer
    #     """
    #     self.buffer.memorize(state, action, reward, done, new_state)

    def sample_batch(self, batch_size,e):
        #for dynamic exploration
        #if(self.explored_data.can_sample(batch_size)):
        #if e<(num_trajectory-1):
        # else:

        #dynamic exploration sample
        #return self.explored_data.sample(batch_size)

        #augmented data sample
        return self.data.sample(batch_size)

        #normal exploration sample
        #return self.data.sample(batch_size)

    def update_models(self, states, actions, critic_target):
        """ Update actor and critic networks from sampled experience
        """
        # Train critic
        self.critic.train_on_batch(states, actions, critic_target)
        # Q-Value Gradients under Current Policy
        actions = self.actor.model.predict(states)
        grads = self.critic.gradients(states, actions)

        #regularization loss for invalid actions
        #loss1: total assignments for one service greater than one
        loss1 = []
        #print(actions)
        for action in actions:
            if (sum(action) > 10):
                temploss = -1#(sum((action[service*actions_per_service:(service+1)*actions_per_service] + 1) / 2) - 1)/actions_per_service
            else:
                temploss = 0

            loss1.append([temploss] * num_users)

        #logarithmic loss function, ignore now, important for approach 2
        #loss1 = -1*np.exp(loss1)

        #generating new gradients
        newgrads = []
        grads = self.critic.gradients(states, actions)

        #normalizing for combination
        grads_norm = (grads - np.min(grads)) / (np.max(grads) - np.min(grads))
        #print(grads_norm)

        loss1 = np.array(loss1)
        grads_norm = np.array(grads_norm)

        #adding regularization losses to gradients
        new_grads = np.add(grads_norm*w_grads,loss1*w_loss1)
        grads = new_grads

        # Train actor
        self.actor.train(states, actions, np.array(grads).reshape((-1, self.act_dim)))
        # Transfer weights to target networks at rate Tau
        self.actor.transfer_weights()
        self.critic.transfer_weights()

    def train(self, batch_size):
        results = []
        e = 0

        # First, gather experience
        #since training is offline, dont need this

        while e < EPISODES:

            # # Reset episode
            # time, cumul_reward, done = 0, 0, False
            # old_state = env.reset()
            # actions, states, rewards = [], [], []
            # noise = OrnsteinUhlenbeckProcess(size=self.act_dim)

            #for updating model more than once per batch sample, ignore required for dynamic exploration testing
            updates = 0

            while updates < no_updates:
                time = 0
                updates +=1
                time = 0

                # while time <= num_trajectories:
                    #ignore, required for online training
                    # if args.render: env.render()
                    # # Actor picks an action (following the deterministic policy)
                    # a = self.policy_action(old_state)
                    # # Clip continuous values to be valid w.r.t. environment
                    # a = np.clip(a+noise.generate(time), -self.act_range, self.act_range)
                    # # Retrieve new state, reward, and whether the state is terminal
                    # new_state, r, done, _ = env.step(a)
                    # # Add outputs to memory buffer
                    # self.memorize(old_state, a, r, done, new_state)

                    # Sample experience from buffer
                if self.data.can_sample(batch_size):
                    states, actions, rewards, new_states, dones = self.sample_batch(batch_size,e)
                    states = np.reshape(states,(batch_size,40))
                    new_states = np.reshape(new_states, (batch_size, 40))

                    # Predict target q-values using target networks
                    q_values = self.critic.target_predict([new_states, self.actor.target_predict(new_states)])
                    # Compute critic target
                    critic_target = self.bellman(rewards, q_values, dones)
                    # Train both networks on sampled batch, update target networks
                    self.update_models(states, actions, critic_target)

                    #ignore, for online training
                    # Update current state
                    #old_state = new_state
                    #cumul_reward += r

                    #results per epoch
                    #print("epoch", (e*num_time_slot)+time, " : ", rp.test(self))
                    # x.append((e*num_time_slot)+time)
                    # y.append(rp.test(self))


            #testing
            total_reward = rp.test(self)
            print("episode number",e," : ",total_reward )
            # self.x.append(e)
            # self.y.append(total_reward)

            # e +=1
            # # #adding more exploration for dynamic exploration testing, ignore
            # if e< 2:#(num_trajectory-1):
            #     s, a, r, _, _ = self.data.pull_by_index(0, batch_size*(e+1))
            #     #increase buffer size
            #     self.explored_data = ReplayBuffer(batch_size*(e+1))
            #     self.explored_data = data_to_replay_buffer(s, a, r, self.explored_data, num_time_slot)

        return results

    def save_weights(self, path):
        path += '_LR_{}'.format(self.lr)
        self.actor.save(path)
        self.critic.save(path)

    def load_weights(self, path_actor, path_critic):
        self.critic.load_weights(path_critic)
        self.actor.load_weights(path_actor)
