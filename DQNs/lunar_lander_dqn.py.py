# Double DQN + Dueling Network + Annealing Epsilon + Target Soft Updating
import itertools
import math
import pathlib
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='ImprovedDQN')
parser.add_argument('--num_episodes', default=500, type=int, help='number of episodes to train in a training loop')
parser.add_argument('--num_agents', default=4, type=int, help='number of agents to train for each category')

args = parser.parse_args()
num_episodes = args.num_episodes
times = args.num_agents


#  The implementation of Prioritized Experience Replay (PER)
#  borrowed the idea from https://m.youtube.com/watch?v=MqZmwQoOXw4
class ReplayBuffer:
    def __init__(self, capacity, buffer_min):
        self.buffer = collections.deque(maxlen=capacity)
        self.prioritize = collections.deque(maxlen=capacity)
        self.buffer_min = buffer_min

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        self.prioritize.append(1)  # the latest experience likely to be chosen

    def sample(self, batch_size):
        exp = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*exp)
        return np.array(state), action, reward, np.array(next_state), done

    def sample_per(self, batch_size):
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
        for i, e in zip(index, error):
            self.prioritize[i] = e + min_prioritize
            if self.prioritize[i] > 1:
                self.prioritize[i] = 1


# define nn
class DQNnet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNnet, self).__init__()
        self.layer1 = nn.Linear(state_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 64)

        self.layer4 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)


#  The Dueling architecture is influenced by https://nn.labml.ai/rl/dqn/model.html
#  but conv layers are substituted by linear layers
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
    def __init__(self, state_dim, action_dim, learning_rate, gm, eps, target_update, device, prioritized, double,
                 dueling, name):
        self.name = name
        self.prioritized = prioritized
        self.double = double
        self.dueling = dueling
        self.TAU = 0.005
        self.eps_end = 0.01
        if self.dueling:
            print('Using Dueling Network')
            self.q = DuelingNetwork(state_dim, action_dim).to(device)
            self.target_q = DuelingNetwork(state_dim, action_dim).to(device)
        else:
            print('Using Deep-Q Network')
            self.q = DQNnet(state_dim, action_dim).to(device)
            self.target_q = DQNnet(state_dim, action_dim).to(device)

        self.target_q.load_state_dict(self.q.state_dict())
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=learning_rate)  # adam optimizer
        self.gamma = gm
        self.epsilon = eps  # parameter ε-greedy
        self.target_update = target_update
        self.training_counter = 0
        self.step_counter = 0
        self.episode_counter = 0
        self.device = device
        # self.steps_done = 0

    #  Decaying epsilon inspired by
    #  https://davidrpugh.github.io/stochastic-expatriate-descent/pytorch/deep-reinforcement-learning/deep-q-networks/2020/04/11/double-dqn.html
    def take_action(self, state):  # take action with ε-greedy
        eps_threshold = max(self.eps_end, (0.99 ** self.episode_counter))
        if np.random.random() < eps_threshold:
            action = np.random.randint(action_dim)
        else:
            # the state is a tuple, we need a tensor
            state = torch.tensor(np.array(state), dtype=torch.float).to(self.device)
            with torch.no_grad():
                action = self.q(state).argmax().item()
        return action

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
            if self.double:
                best_next_actions = self.q(state).max(1)[1].unsqueeze(1)
                max_next_values = self.target_q(next_state).gather(1, best_next_actions)
            else:
                max_next_values = self.target_q(next_state).max(1)[0].unsqueeze(1)

        q_t = reward + self.gamma * max_next_values * (1 - done)  # target Q value
        error = abs(q_v - q_t).view(-1)
        if self.prioritized:
            loss = F.huber_loss(weight * q_v, weight * q_t)  # huber loss
        else:
            loss = F.huber_loss(q_v, q_t)

        # train
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update target model inspired by https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        q_state_dict = self.q.state_dict()
        target_state_dict = self.target_q.state_dict()
        for k in q_state_dict:
            target_state_dict[k] = self.TAU * q_state_dict[k] + (1 - self.TAU) * target_state_dict[k]
            self.target_q.load_state_dict(target_state_dict)
        self.training_counter += 1  # step + 1
        return self.training_counter, error.cpu().detach().numpy(), loss.item()


# hyperparameter setting
learning_rate = 1e-4
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

buffers = ReplayBuffer(buffer_size, buffer_min_size)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Instantiate different types of agents here

# prioritized_dueling_DDQN = ImprovedDQN(state_dim, action_dim, learning_rate, gamma, epsilon, target_update, device,
#                                        prioritized=True, double=True, dueling=True, name='Prioritized Dueling DDQN')
# prioritized_DDQN = ImprovedDQN(state_dim, action_dim, learning_rate, gamma, epsilon, target_update, device,
#                                prioritized=True, double=True, dueling=False, name='Prioritized DDQN')
# prioritized_DQN = ImprovedDQN(state_dim, action_dim, learning_rate, gamma, epsilon, target_update, device,
#                               prioritized=True, double=False, dueling=False, name='Prioritized DQN')
DQN = ImprovedDQN(state_dim, action_dim, learning_rate, gamma, epsilon, target_update, device,
                  prioritized=False, double=False, dueling=False, name='DQN')
DDQN = ImprovedDQN(state_dim, action_dim, learning_rate, gamma, epsilon, target_update, device,
                   prioritized=False, double=True, dueling=False, name='DDQN')
dueling_DDQN = ImprovedDQN(state_dim, action_dim, learning_rate, gamma, epsilon, target_update, device,
                           prioritized=False, double=True, dueling=True, name='Dueling DDQN')

# put the agents you want to train
# agents = [DQN, prioritized_DQN, prioritized_DDQN, prioritized_dueling_DDQN]
agents = [DQN, DDQN, dueling_DDQN]


def train_n_times(agent, n, num_episodes):  # return list of length n
    pathlib.Path('./models/').mkdir(parents=True, exist_ok=True)
    torch.save(agent, f'./models/{agent.name} (before training)')

    data_for_each_training = {'episode_rewards': [],
                              'episode_lengths': [],
                              'episode_losses': []}
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    env.reset(seed=0)
    for i in range(n):
        episode_reward_list = []
        episode_length_list = []
        episode_loss_list = []
        step_reward_list = []
        agent = torch.load(f'./models/{agent.name} (before training)')
        for _ in range(num_episodes):
            episode_reward = 0
            step_loss_list = []
            s = env.reset()[0]
            for t in itertools.count():
                a = agent.take_action(s)
                next_s, r, done, truncated, __ = env.step(a)
                agent.step_counter += 1
                buffers.add(s, a, r, next_s, done)
                s = next_s
                step_reward_list.append(r)
                episode_reward += r

                if buffers.start_training():
                    if agent.prioritized:
                        bs, ba, br, bns, bd, index, weight = buffers.sample_per(batch_size)
                        exp_data = {"state": bs, "action": ba, "reward": br, "next_state": bns, "done": bd}
                        _, error, loss = agent.learn(exp_data, weight)
                        buffers.update_probability(index, error)
                    else:
                        bs, ba, br, bns, bd = buffers.sample(batch_size)
                        exp_data = {"state": bs, "action": ba, "reward": br, "next_state": bns, "done": bd}
                        _, _, loss = agent.learn(exp_data, 0)
                    step_loss_list.append(loss)

                if done or truncated:
                    episode_length_list.append(t + 1)
                    if len(step_loss_list) == 0:
                        episode_loss_list.append(0)
                    else:
                        episode_loss_list.append(sum(step_loss_list) / len(step_loss_list))
                    break

            agent.episode_counter += 1
            episode_reward_list.append(episode_reward)
            print(f'agent: {agent.name} episode{agent.episode_counter} reward: {episode_reward} '
                  f'agent.step_counter: {agent.step_counter}')
        data_for_each_training['episode_rewards'].append(episode_reward_list)
        data_for_each_training['episode_lengths'].append(episode_length_list)
        data_for_each_training['episode_losses'].append(episode_loss_list)
    pathlib.Path('./models/').mkdir(parents=True, exist_ok=True)
    torch.save(agent, f'./models/{agent.name} (after training)')  # save the model of the last trained agent
    return data_for_each_training


def average_nested_list(nested_list):
    rows = len(nested_list)
    cols = len(nested_list[0])
    average_across_rows = []
    for j in range(0, cols):
        reward = 0
        for i in range(0, rows):
            reward += nested_list[i][j]
        average_across_rows.append(reward / rows)
    return average_across_rows


def train_n_times_averaged(agent):
    data_for_each_training = train_n_times(agent, times, num_episodes)
    pathlib.Path('./data/').mkdir(parents=True, exist_ok=True)
    torch.save(data_for_each_training, f'./data/data_for_each_training-{agent.name}')
    for k in data_for_each_training:
        print(f'{k}: {data_for_each_training[k]}')
    averaged_data = {k: average_nested_list(data_for_each_training[k])
                     for k in data_for_each_training}
    return averaged_data


agent_to_averaged_data = {agent.name: train_n_times_averaged(agent) for agent in agents}
pathlib.Path('./data/').mkdir(parents=True, exist_ok=True)
torch.save(agent_to_averaged_data, f'./data/agent_to_averaged_data')

print('Models saved to ./models/')
print('Data saved to ./data/')
