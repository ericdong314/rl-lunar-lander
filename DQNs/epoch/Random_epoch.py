# Double DQN + Dueling Network + Annealing Epsilon + Target Soft Updating
import math
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import time
import pickle

prioritized = True
double = True
dueling = True


# define the replay buffer
class ReplayBuffer:
    def __init__(self, capacity, buffer_min):
        self.buffer = collections.deque(maxlen=capacity)
        self.prioritize = collections.deque(maxlen=capacity)
        self.buffer_min = buffer_min

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        self.prioritize.append(1)  # the latest experience likely to be chosen

    def sample(self, batch_size):
        priorities = np.array(self.prioritize)
        probabilities = priorities / sum(priorities)

        index = random.choices(range(len(self.buffer)), k=batch_size, weights=probabilities)
        exp = np.array(self.buffer, dtype=object)[index]

        # calculate weight for selected samples
        weight = 1 / probabilities[index]
        weight = weight / max(weight)

        # exp = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*exp)
        return np.array(state), action, reward, np.array(next_state), done, index, weight

    def start_training(self):
        if len(self.buffer) > self.buffer_min:
            return True
        else:
            return False

    def update_probability(self, index, error, min_prioritize=0.1):
        # pass
        for i, e in zip(index, error):
            self.prioritize[i] = e + min_prioritize
            if self.prioritize[i] > 1:
                self.prioritize[i] = 1


class DuelingNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingNetwork, self).__init__()
        self.layer1 = nn.Linear(state_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 64)

        self.state_values = nn.Linear(64, 1)
        self.advantages = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))

        state_values = self.state_values(x)
        action_advantages = self.advantages(x)

        return state_values + (action_advantages - action_advantages.mean(dim=-1, keepdim=True))


class ImprovedDQN:
    def __init__(self, state_dim, action_dim, learning_rate, gm, eps, target_update, device):
        self.TAU = 0.005
        self.eps_end = 0.01
        if dueling:
            self.q = DuelingNetwork(state_dim, action_dim).to(device)
            self.target_q = DuelingNetwork(state_dim, action_dim).to(device)
        else:
            hidden_dim = 128
            self.q = DQNnet(state_dim, action_dim, hidden_dim).to(device)
            self.target_q = DQNnet(state_dim, action_dim, hidden_dim).to(device)

        self.target_q.load_state_dict(self.q.state_dict())
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=learning_rate)  # adam optimizer
        self.gamma = gm
        self.epsilon = eps  # parameter ε-greedy
        self.target_update = target_update
        self.step_counter = 0
        self.episode_counter = 0
        self.device = device

    def take_action(self, state):  # take action with ε-greedy
        return np.random.randint(action_dim)
        # eps_threshold = max(self.eps_end, (0.99 ** self.episode_counter))
        # if np.random.random() < eps_threshold:
        #     action = np.random.randint(action_dim)
        # else:
        #     # the state is a tuple, we need a tensor
        #     state = torch.tensor(np.array(state), dtype=torch.float).to(self.device)
        #     with torch.no_grad():
        #         action = self.q(state).argmax().item()
        # return action

    def learn(self, exp_data, weight):

        # sample buffer
        state = torch.tensor(exp_data["state"], dtype=torch.float).to(self.device)
        action = torch.tensor(exp_data["action"]).view(-1, 1).to(self.device)
        reward = torch.tensor(exp_data["reward"], dtype=torch.float).view(-1, 1).to(self.device)
        next_state = torch.tensor(exp_data["next_state"], dtype=torch.float).to(self.device)
        done = torch.tensor(exp_data["done"], dtype=torch.float).view(-1, 1).to(self.device)
        weight = torch.tensor(weight, dtype=torch.float).to(self.device)

        # calculate loss
        q_v = self.q(state).gather(1, action)  # Q value
        with torch.no_grad():
            if double:
                best_next_actions = self.q(state).max(1)[1].unsqueeze(1)
                max_next_values = self.target_q(next_state).gather(1, best_next_actions)
            else:
                max_next_values = self.target_q(next_state).max(1)[0].unsqueeze(1)

        # max_next_values = self.target_q(next_state).max(1)[0].view(-1, 1)
        q_t = reward + self.gamma * max_next_values * (1 - done)  # target Q value
        error = abs(q_v - q_t).view(-1)
        # loss = F.mse_loss(q_v, q_t))  # loss
        loss = F.huber_loss(weight * q_v, weight * q_t)  # huber loss

        # train
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # soft update target model
        q_state_dict = self.q.state_dict()
        target_state_dict = self.target_q.state_dict()
        for k in q_state_dict:
            target_state_dict[k] = self.TAU * q_state_dict[k] + (1 - self.TAU) * target_state_dict[k]
            self.target_q.load_state_dict(target_state_dict)
        self.step_counter += 1  # step + 1
        return self.step_counter, error.cpu().detach().numpy()


def plot_rewards(episode_rewards):
    plt.figure(1)
    rewards = torch.tensor(episode_rewards)

    plt.clf()
    plt.xlabel('epoch')
    plt.ylabel('reward')
    plt.plot(rewards.numpy())
    if len(episode_rewards) > 50:
        moving_average = rewards.unfold(0, 50, 1).mean(1).view(-1)
        moving_average = torch.cat([torch.zeros(49), moving_average])
        plt.plot(moving_average.numpy())
    plt.pause(0.001)


# hyperparameter setting
learning_rate = 1e-4
steps = 205000
buffer_size = 20000
batch_size = 64
gamma = 0.99
epsilon = 0.01
target_update = 10
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using device: {device}')
buffer_min_size = 500

# initialize environment
env = gym.make("LunarLander-v2")

# seed
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

buffers = ReplayBuffer(buffer_size, buffer_min_size)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = ImprovedDQN(state_dim, action_dim, learning_rate, gamma, epsilon, target_update, device)

env.reset(seed=0)

reward_lists = []
for i in range(4):
    epoch_reward = []
    reward_list = []
    eps_count = 0
    count = 0

    agent = ImprovedDQN(state_dim, action_dim, learning_rate, gamma, epsilon, target_update, device)
    while count < steps:
        # for n_eps in range(episodes):
        eps_count += 1
        episode_reward = 0
        s = env.reset()[0]
        done = False
        truncated = False
        while not (done or truncated):
            a = agent.take_action(s)
            next_s, r, done, truncated, __ = env.step(a)
            # buffers.add(s, a, r, next_s, done)
            s = next_s
            episode_reward += r

            # if buffers.start_training():
            #     bs, ba, br, bns, bd, index, weight = buffers.sample(batch_size)
            #     exp_data = {"state": bs, "action": ba, "reward": br, "next_state": bns, "done": bd}
            #     count, error = agent.learn(exp_data, weight)
            #     buffers.update_probability(index, error)
            count+=1

            if count % 4096 == 0 and count != 0:  # set epoche size here
                if len(epoch_reward) > 0:
                    reward_list.append(sum(epoch_reward) / len(epoch_reward))
                else:
                    reward_list.append(reward_list[-1])
                print(f'episode: {eps_count}', f'step_count: {count}')
                print(f'epoch average reward: {reward_list[-1]}')
                epoch_reward.clear()
        epoch_reward.append(episode_reward)
        print(f'episode{eps_count}reward: {episode_reward}')

        agent.episode_counter += 1

    reward_lists.append(reward_list)

torch.save(reward_lists, './data/epoch/epoch_reward_lists_random')
