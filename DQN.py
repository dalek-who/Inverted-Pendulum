"""
参数化方法：深度Q网络(DQN)，用gym内置的Pendulum-v0倒立摆环境可以成功,但是我自己定义的倒立摆超参数没调出来
"""
import matplotlib
# matplotlib.use('TkAgg')

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, RandomSampler
from collections import deque
from tqdm import tqdm
from collections import namedtuple
from time import sleep
import matplotlib.pyplot as plt
from gym import wrappers

from BaseAgent import BaseAgent
from game.my_Pendulum import MyPendulumEnv

seed = 2
torch.manual_seed(seed)
np.random.seed(seed)

Example = namedtuple('Example', ['state', 'feature', 'index_action', 'reward', 'next_state', 'next_feature'])  # 方便dataloader读取，并且可以自由调整feature的维度而不用改变索引

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


class DQN_Dataset(Dataset):
    def __init__(self, example_list):
        super(DQN_Dataset, self).__init__()
        self.example_list = example_list

    def __len__(self):
        return len(self.example_list)

    def __getitem__(self, item):
        return self.example_list[item]


def state_to_feature(state):
    return state  # Pendulum-v0 返回的state(准确说是observation)是 (sin(theta), cos(theta), d_theta)
    # return np.array([np.sin(state[0]), np.cos(state[0]), state[1]])  # 我自己的倒立摆的state是(theta, d_theta)，转换成feature


class DQN_Agent(BaseAgent):
    def __init__(self, n_theta=200, n_d_theta=201, n_actions=3, gamma=0.98):
        super(DQN_Agent, self).__init__(n_theta, n_d_theta, n_actions, gamma)
        self.q_table = None
        self.env = gym.make('Pendulum-v0')

    # 通过epsilon贪心策略返回一个action
    def epsilon_greedy_action(self, feature, epsilon, return_index=False):  # return_index：返回的是action的数值还是index
        if np.random.rand() > 1-epsilon:  # 以1-epsilon的概率返回q最大的action
            return self.do_action(feature, return_index)
        else:  # 以epsilon的概率随机返回一个action
            index_action = np.random.randint(0, self.n_actions)
            action = np.random.choice(self.actions)
            action = action if isinstance(action, np.ndarray) else np.array([action])
            return index_action if return_index else action

    # 贪心策略选择action
    def do_action(self, feature, return_index=False):
        feature = torch.from_numpy(feature).float().unsqueeze(0)
        with torch.no_grad():
            index_action = self.target_net(torch.tensor(feature).float())
            index_action = index_action.argmax()
        if return_index:
            return index_action
        else:
            action = self.actions[index_action.item()]
            action = action if isinstance(action, np.ndarray) else np.array([action])
            return action

    def train(self):
        queue_size = 2000  # 样本池（FIFO队列）的大小。每个epoch采样一个episode，当队列满了时用新数据代替最老的旧数据
        num_epoch = 1000  # epoch数量，每个epoch采样一个episode
        episode_len = 200  # 一个episode的长度
        batch_size = 32
        max_epsilon = 1  # epsilon-greedy采样动作时epsilon是衰减的。这是起始epsilon值
        min_epsilon = 0.01  # epsilon衰减时不能低于这个值
        epsilon_decay = 0.999  # epsilon的衰减率
        max_grad_norm = 0.5  # 梯度剪裁
        lr = 1e-3  # 学习率
        render_epoch = 50  # 每多少个epoch渲染一次，展示效果
        n_features = len(state_to_feature(agent.env.reset()))  # 网络输入的特征数。（sin(theta), cos(theta), d_theta）三个特征

        self.policy_net: nn.Module = DQN_Net(n_actions=self.n_actions, n_features=n_features)  # 梯度更新的网络
        self.target_net: nn.Module = DQN_Net(n_actions=self.n_actions, n_features=n_features)  # 用于采样动作的网络
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        examples_queue = deque(maxlen=queue_size)  # 样本池（FIFO队列）
        global_step = -1  # 第几个梯度更新step
        loss_list = []  # 记录每次梯度更新的train_loss
        epsilon = max_epsilon  # epsilon-greedy的epsilon
        running_reward, running_q = -1000, 0
        self.optimizer: optim.Optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)  # 网络优化器

        for epoch in tqdm(range(num_epoch), desc="Epoch"):
            score = 0  # 一个epoch内采样episode的reward总值
            state = self.env.reset()
            epoch_loss = 0

            # 采样一个episode
            for episode_step in range(episode_len):  # 采样新的episode
                feature = state_to_feature(state)  # (sin(theta), cos(theta), d_theta)
                index_action = self.epsilon_greedy_action(feature, epsilon=epsilon, return_index=True)  # epsilon-greedy采用action
                next_state, reward, done, info = self.env.step(np.array([self.actions[index_action]]))
                next_fearure = state_to_feature(next_state)
                example = Example(state, feature, np.array([index_action]), np.array([reward]), next_state, next_fearure)  # 一个训练样本点
                examples_queue.append(example)
                score += reward
                if done:
                    state = self.env.reset()
                else:
                    state = next_state
                if epoch % render_epoch == 0:  # 用来展示当前的网络效果。（但因为是epsilon-greedy采样，看上去不是特别稳定）
                    self.env.render()

            # 将采样的样品转换为torch的DataLoader，方便训练时batch随机采样
            dataset = DQN_Dataset(example_list=examples_queue)
            dataloader = DataLoader(dataset, batch_size=batch_size, sampler=RandomSampler(dataset))

            # 网络训练过程。（和课上讲的不太一样，课上是每采样一个样本训练一次，batch为1。我这里是采样一个episode后训练。反正最终效果也还行）
            for data in dataloader:
                global_step += 1

                tensor_state = data.state.float()  # 当前state，[sin(theta), cos(theta), d_theta]
                tensor_feature = data.feature.float()  # 当前state转成网络输入feature
                tensor_index_action = data.index_action.long()  # 当前state依据epsilon-greedy采样的策略index
                tensor_reward = data.reward.float()  # 当前(state, action)的reward
                tensor_next_state = data.next_state.float()  # 下一个state
                tensor_next_feature = data.next_feature.float()  # 下一个state的feature

                tensor_state_q_values = self.policy_net(tensor_feature)  # 当前state下，神经网络拟合出的每个action的q_value
                tensor_state_action_q_value = tensor_state_q_values.gather(dim=1, index=tensor_index_action)  # 当前state对应的epsilon-greedy action的，神经网络拟合的q_value
                with torch.no_grad():
                    tensor_next_state_q_values = self.target_net(tensor_next_feature)  # 下个state下，神经网络拟合出的每个action的q_value
                tensor_expected_state_action_q_value = (tensor_next_state_q_values.max(dim=1, keepdim=True)[0] * self.gamma) + tensor_reward  # 依据greedy策略，当前state期望的q_value
                # 注意强化学习和监督学习的loss区别：监督学习的loss是预测值和ground_turth的loss，强化学习loss是预测值域期望值的loss，期望值起了ground_turth的作用
                loss = F.smooth_l1_loss(tensor_state_action_q_value, tensor_expected_state_action_q_value)

                # 梯度更新
                self.optimizer.zero_grad()
                loss.backward()
                for param in self.policy_net.parameters():
                    param.grad.data.clamp_(-max_grad_norm, max_grad_norm)
                self.optimizer.step()
                epsilon = max(epsilon * epsilon_decay, min_epsilon)  # epsilon-greedy的epsilon衰减
                if global_step % 200 == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

                # 打印部分结果
                with torch.no_grad():
                    q = tensor_state_action_q_value.mean().item()
                    running_q = 0.99 * running_q + 0.01 * q
                epoch_loss += loss.item()

            loss_list.append(epoch_loss / len(examples_queue))
            running_reward = running_reward * 0.9 + score * 0.1
            if epoch % 10 == 0:
                print('Ep {}\tAverage score: {:.2f}\tAverage Q: {:.2f}'.format(
                    epoch, running_reward, running_q))
            if running_reward>-200:
                break
        self.show_convergence_curve(loss_list)

    # gym展示模型效果
    def demo(self, max_step=1000, save_video=False):
        original_env = None
        if save_video:
            original_env = self.env
            self.env = wrappers.Monitor(self.env, self.data_dir + "demo_video", force=True)
        observation = self.env.reset()
        self.env.render()
        sleep(1)
        step = 0
        while True:
            step += 1
            self.env.render()  # 渲染动画
            feature = state_to_feature(observation)
            action = self.do_action(feature)
            observation, reward, done, info = self.env.step(action)
            if done or step>max_step:
                print("Episode finished after {} timesteps, done={}".format(step + 1, done))
                break
        sleep(2)
        self.env.close()

    # 画收敛曲线
    def show_convergence_curve(self, diff_list):
        fig, ax = plt.subplots()
        plot = ax.plot(np.arange(1, len(diff_list)+1), diff_list)

        # 子图ax的标题
        ax.set_title("Train loss of each epoch\n action=%s" % (self.n_actions))  # y可以调整标题的位置

        # 设置坐标轴格式
        # x轴（step）
        ax.set_xlabel('step')  # x轴title

        # y轴（二范数）
        ax.set_ylabel('loss')  # y轴title
        # 保存与显示
        fig.tight_layout()  # 适应窗口大小，否则可能有东西在窗口里画不下
        fig.savefig(self.data_dir + "convergence.png")
        fig.canvas.set_window_title(self.model_name + ": convergence")  # 窗口fig的title
        fig.show()


if __name__=="_main__":
    agent = DQN_Agent()
    agent.train()
    for i in range(3):
        agent.demo()
    agent.demo(save_video=True)
    agent.env_close()



