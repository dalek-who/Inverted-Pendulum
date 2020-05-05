"""
参数化方法：深度Q网络(DQN)
"""
import matplotlib
matplotlib.use('TkAgg')

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
        hidden_size = 8
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
        self.policy_net: nn.Module = DQN_Net(n_actions=self.n_actions, n_features=2)
        self.target_net: nn.Module = DQN_Net(n_actions=self.n_actions, n_features=2)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer: optim.Optimizer = optim.RMSprop(self.policy_net.parameters())

    # 通过epsilon贪心策略返回一个action
    def epsilon_greedy_action(self, theta, d_theta, epsilon, return_index=False):
        if np.random.rand() > 1-epsilon:  # 以1-epsilon的概率返回q最大的action
            with torch.no_grad():
                index_action = self.target_net(torch.tensor([theta, d_theta]).float())
                index_action = index_action.argmax()
            action = self.actions[index_action.item()]
        else:  # 以epsilon的概率随机返回一个action
            index_action = np.random.randint(0, self.n_actions)
            action = np.random.choice(self.actions)
        action = action if isinstance(action, np.ndarray) else np.array([action])
        return index_action if return_index else action

    def do_action(self, theta, d_theta, return_index=False):
        with torch.no_grad():
            index_action = self.target_net(torch.tensor([theta, d_theta]).float())
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
        queue_size= 5000
        num_epoch = 500
        episode_len = 500
        batch_size = 32
        epsilon = 0.2

        examples_queue = deque(maxlen=queue_size)
        global_step = -1
        loss_list = []
        for epoch in tqdm(range(num_epoch), desc="Epoch"):
            theta, d_theta = self.env.reset()
            for episode_step in range(episode_len):  # 采样新的episode
                index_action = self.epsilon_greedy_action(theta, d_theta, epsilon=epsilon, return_index=True)  # epsilon-greedy采用action
                (next_theta, next_d_theta), reward, done, info = self.env.step(np.array([self.actions[index_action]]))
                example = (theta, d_theta, index_action, reward, next_theta, next_d_theta)
                examples_queue.append(example)
                if done:
                    theta, d_theta = self.env.reset()
                else:
                    theta, d_theta = next_theta, next_d_theta
            dataloader = DataLoader(TensorDataset(torch.tensor(examples_queue)), batch_size=batch_size)
            for data in dataloader:
                global_step += 1
                data = data[0]
                tensor_state = data[:, [0,1]].float()  # 当前state，[theta, d_theta]
                tensor_index_action = data[:, [2]].long()  # 当前state依据epsilon-greedy采样的策略index
                tensor_reward = data[:, [3]].float()  # 当前(state, action)的reward
                tensor_next_state = data[:, [4,5]].float()  # 下一个state

                tensor_state_q_values = self.policy_net(tensor_state)  # 当前state下，神经网络拟合出的每个action的q_value
                tensor_state_action_q_value = tensor_state_q_values.gather(dim=1, index=tensor_index_action)  # 当前state对应的epsilon-greedy action的，神经网络拟合的q_value
                tensor_next_state_q_values = self.policy_net(tensor_next_state)  # 下个state下，神经网络拟合出的每个action的q_value
                tensor_expected_state_action_q_value = (tensor_next_state_q_values.max(dim=1, keepdim=True)[0] * self.gamma) + tensor_reward  # 依据greedy策略，当前state期望的q_value
                loss = F.smooth_l1_loss(tensor_state_action_q_value, tensor_expected_state_action_q_value)

                self.optimizer.zero_grad()
                loss.backward()
                for param in self.policy_net.parameters():
                    param.grad.data.clamp_(-1, 1)
                self.optimizer.step()
                loss_list.append(loss.detach().item())
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.show_convergence_curve(loss_list)




agent = DQN_Agent()
agent.train()
agent.demo()



