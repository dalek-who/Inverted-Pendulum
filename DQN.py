"""
参数化方法：深度Q网络(DQN)
"""
import matplotlib
# matplotlib.use('TkAgg')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from collections import deque
from tqdm import tqdm

from BaseAgent import BaseAgent


class DQN_Net(nn.Module):
    def __init__(self, n_features, n_actions):
        super().__init__()
        hidden_size = 100
        self.net = nn.Sequential(nn.Linear(n_features, hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, n_actions))

    def forward(self, x):
        return self.net(x)



class DQN_Agent(BaseAgent):
    def __init__(self, n_theta=200, n_d_theta=201, n_actions=3, gamma=0.98):
        super(DQN_Agent, self).__init__(n_theta, n_d_theta, n_actions, gamma)
        self.q_table = None

    # 通过epsilon贪心策略返回一个action
    def epsilon_greedy_action(self, feature, epsilon, return_index=False):
        if np.random.rand() > 1-epsilon:  # 以1-epsilon的概率返回q最大的action
            with torch.no_grad():
                index_action = self.target_net(torch.tensor(feature).float())
                index_action = index_action.argmax()
            action = self.actions[index_action.item()]
        else:  # 以epsilon的概率随机返回一个action
            index_action = np.random.randint(0, self.n_actions)
            action = np.random.choice(self.actions)
        action = action if isinstance(action, np.ndarray) else np.array([action])
        return index_action if return_index else action

    def do_action(self, feature, return_index=False):
        with torch.no_grad():
            index_action = self.target_net(torch.tensor(feature).float())
            index_action = index_action.argmax()
        if return_index:
            return index_action
        else:
            action = self.actions[index_action.item()]
            action = action if isinstance(action, np.ndarray) else np.array([action])
            return action

    def train_step(self):
        pass

    def train(self):
        queue_size = 2000
        num_epoch = 100
        episode_len = 200
        batch_size = 32
        max_epsilon = 1
        min_epsilon = 0.01
        epsilon_decay = 0.999
        max_grad_norm = 0.5
        lr = 1e-3
        n_features = 4

        self.policy_net: nn.Module = DQN_Net(n_actions=self.n_actions, n_features=n_features)
        self.target_net: nn.Module = DQN_Net(n_actions=self.n_actions, n_features=n_features)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        examples_queue = deque(maxlen=queue_size)
        global_step = -1
        reward_list = []
        epsilon = max_epsilon
        running_reward, running_q = -1000, 0
        self.optimizer: optim.Optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        for epoch in tqdm(range(num_epoch), desc="Epoch"):
            score = 0
            state = self.env.reset()
            for episode_step in range(episode_len):  # 采样新的episode
                feature = (state[0], state[1], np.sin(state[0]), np.cos(state[0]))  # theta, d_theta, sin(theta), cos(theta)
                index_action = self.epsilon_greedy_action(feature, epsilon=epsilon, return_index=True)  # epsilon-greedy采用action
                next_state, reward, done, info = self.env.step(np.array([self.actions[index_action]]))
                theta, d_theta = state
                next_theta, next_d_theta = next_state
                # example = (theta, d_theta, index_action, reward, next_theta, next_d_theta)
                example = (theta, d_theta, np.sin(theta), np.cos(theta),  index_action, reward, next_theta, next_d_theta, np.sin(next_theta), np.cos(next_theta))
                examples_queue.append(example)
                score += reward
                self.env.render()
                if done:
                    # theta, d_theta = self.env.reset()
                    state = self.env.reset()
                else:
                    # theta, d_theta = next_theta, next_d_theta
                    state = next_state
            if len(examples_queue)<examples_queue.maxlen:
                continue
            dataloader = DataLoader(TensorDataset(torch.tensor(examples_queue)), batch_size=batch_size)
            for data in dataloader:
                global_step += 1
                data = data[0]
                # tensor_state = data[:, [0,1]].float()  # 当前state，[theta, d_theta]
                # tensor_index_action = data[:, [2]].long()  # 当前state依据epsilon-greedy采样的策略index
                # tensor_reward = data[:, [3]].float()  # 当前(state, action)的reward
                # tensor_next_state = data[:, [4,5]].float()  # 下一个state
                tensor_state = data[:, [0, 1, 2, 3]].float()  # 当前state，[theta, d_theta]
                tensor_index_action = data[:, [4]].long()  # 当前state依据epsilon-greedy采样的策略index
                tensor_reward = data[:, [5]].float()  # 当前(state, action)的reward
                tensor_next_state = data[:, [6, 7, 8, 9]].float()  # 下一个state

                tensor_state_q_values = self.policy_net(tensor_state)  # 当前state下，神经网络拟合出的每个action的q_value
                tensor_state_action_q_value = tensor_state_q_values.gather(dim=1, index=tensor_index_action)  # 当前state对应的epsilon-greedy action的，神经网络拟合的q_value
                with torch.no_grad():
                    tensor_next_state_q_values = self.target_net(tensor_next_state)  # 下个state下，神经网络拟合出的每个action的q_value
                tensor_expected_state_action_q_value = (tensor_next_state_q_values.max(dim=1, keepdim=True)[0] * self.gamma) + tensor_reward  # 依据greedy策略，当前state期望的q_value
                loss = F.smooth_l1_loss(tensor_state_action_q_value, tensor_expected_state_action_q_value)

                self.optimizer.zero_grad()
                loss.backward()
                for param in self.policy_net.parameters():
                    param.grad.data.clamp_(-max_grad_norm, max_grad_norm)
                self.optimizer.step()
                epsilon = max(epsilon * epsilon_decay, min_epsilon)
                if global_step % 200 == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                with torch.no_grad():
                    q = tensor_state_action_q_value.mean().item()
            running_reward = running_reward * 0.9 + score * 0.1
            reward_list.append(score)
            if running_reward>-200:
                break
        self.show_convergence_curve(reward_list)




agent = DQN_Agent()
agent.train()
agent.demo()



