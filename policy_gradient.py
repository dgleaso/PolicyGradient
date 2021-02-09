import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.distributions import Categorical
import gym
import numpy as np

import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')

class Policy(nn.Module):
    def __init__(self, state_space, layer_size, action_space):
        super(Policy, self).__init__()
        self.layer1 = nn.Sequential(
                nn.Linear(state_space, layer_size),
                nn.ReLU()
                )
        self.layerFinal = nn.Sequential(
                nn.Linear(layer_size, action_space),
                # Softmax in last layer to treat output as probability of discrete random variable
                # All values between 0 and 1, sums to 1
                nn.Softmax(dim=-1)
                )
        
    def forward(self, x):
        output = self.layer1(x)
        output = self.layerFinal(output)
        return output

class PolicyGradient():
    def __init__(self, lr, gamma, state_space, layer_size, action_space):
        self.policy = Policy(state_space, layer_size, action_space)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-3)
        self.gamma = gamma

    # Runs the current policy for one episode and returns training data
    def get_trajectory(self, env, episode_length, variance):

        rewards = []
        # To store log probability of actions
        log_actions = []

        obs = env.reset()

        reward_sum = 0
        for i in range(episode_length):

            # Need to convert from numpy array to torch tensor
            obs = torch.from_numpy(obs).float()

            action_probabilities = self.policy(obs)

            # Using categorical distribution for discrete action space
            distribution = Categorical(action_probabilities)
            action = distribution.sample()

            obs, reward, done, _ = env.step(action.numpy())

            # Storing trajectory data for training
            rewards.append(reward)
            reward_sum += reward
            log_actions.append(distribution.log_prob(action))

            if done:
                break
        return rewards, log_actions, reward_sum

    # updates the policy using data gathered by get_trajectory()
    def train_policy(self, rewards, log_actions):
        # Calculate returns using rewards
        returns = []
        # Initializes returns from after episode to 0
        R = 0
        # Calculates returns with discount factor gamma
        for r in reversed(rewards):
            R = r + self.gamma * R
            # Inserts return into beginning of list to put back in chronological order
            returns.insert(0, R)

        # Returns need to be converted to tensor for gradient calculation
        returns = torch.tensor(returns)

        # To store policy loss
        loss = []
        for R, log_actions in zip(returns, log_actions):
            # Appends negative returns * log probability because we want to maximize, however pytorch optim minimizes
            # This is the loss function
            loss.append(-R*log_actions)

        loss = torch.stack(loss)
        # resets gradient
        self.optimizer.zero_grad()
        
        # Sums to account for each timestep
        # Pytorch autograd will calculate this gradient automatically
        loss = loss.sum()

        # Pytorch performs backprop and policy parameter update
        loss.backward()
        self.optimizer.step()



model = PolicyGradient(lr=0.003, gamma=0.99, state_space=4, layer_size=10, action_space=2)

epochs = 1000
running_reward = 0
running_reward_list = [running_reward]
max_reward = -10**5
for e in range(epochs):
    rewards, actions, reward_sum = model.get_trajectory(env=env, episode_length=500, variance=0.1)
    if max_reward < reward_sum:
        max_reward = reward_sum
    running_reward = 0.05 * reward_sum + (1 - 0.05) * running_reward
    running_reward_list.append(running_reward)

    # Early stop if policy is good enough
    if running_reward > 195:
        break
    if e % 10 == 0:
        print("Running Reward: {0}, Max Reward: {1}".format(running_reward, max_reward))
    model.train_policy(rewards, actions)


policy = model.policy
# Saves the model after training
torch.save(policy.state_dict(), "saved-model.pth.tar")

# Runs the trained model
num_episodes = 5
for n in range(num_episodes):
    obs = env.reset()
    reward_sum = 0
    for i in range(500):
        obs = torch.from_numpy(obs).float()
    
        probabilities = policy(obs) 

        distribution = Categorical(probabilities)
        action = distribution.sample()
        obs, reward, done, _ = env.step(action.numpy())
        reward_sum += reward
        env.render()
        if done:
            print("Episode {0} reward: {1}".format(n, reward_sum))
            break

running_reward_array = np.array(running_reward_list)

plt.plot(running_reward_array)
plt.xlabel("Episode")
plt.ylabel("Running reward")
plt.show()

env.close()
