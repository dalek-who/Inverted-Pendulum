"""
连续的动作、状态空间离散化，用SARSA算法对Q-table估值
"""
import matplotlib
matplotlib.use('TkAgg')

import numpy as np

from BaseAgent import BaseAgent

class SARSA(BaseAgent):

    def train(self, episode_num=2000, episode_length=300):  # episode_num: episode数目，episode_length: 一个episode的长度

        decay = 0.9995  # epsilon和学习率的衰减率
        epsilon = 0.3  # 初始的epsilon贪心的探索率
        epsilon_low = 0.01  # epsilon的下界
        gamma = self.gamma

        diff_list = []
        global_step = 0  # q_table进行的多少次更新
        episode_count = 0  # 采样了多少个episode
        while episode_count < episode_num:
            episode_count += 1
            old_q_table = self.q_table.copy()
            theta, d_theta = self.env.reset()
            action = self.epsilon_greedy_action(theta, d_theta, epsilon)
            index_theta = self.value_to_index(theta, -self.max_theta, self.max_theta, self.n_theta)
            index_d_theta = self.value_to_index(d_theta, -self.max_d_theta, self.max_d_theta, self.n_d_theta)
            index_action = self.value_to_index(action, -self.max_action, self.max_action, self.n_actions)
            episode_step = 0  # 当前episode采样了多少个点
            while episode_step < episode_length:
                episode_step += 1
                global_step += 1
                lr = 1 / global_step  # 每回合的"学习率"
                (new_theta, new_d_theta), reward, done, info = self.env.step(action)
                new_action = self.epsilon_greedy_action(new_theta, new_d_theta, epsilon)
                index_new_theta = self.value_to_index(new_theta, -self.max_theta, self.max_theta, self.n_theta)
                index_new_d_theta = self.value_to_index(new_d_theta, -self.max_d_theta, self.max_d_theta, self.n_d_theta)
                index_new_action = self.value_to_index(new_action, -self.max_action, self.max_action, self.n_actions)
                self.q_table[index_theta, index_d_theta, index_action] += lr * (reward + gamma*self.q_table[index_new_theta, index_new_d_theta, index_new_action] - self.q_table[index_theta, index_d_theta, index_action])
                if done:
                    break
            diff = np.linalg.norm(self.q_table - old_q_table)
            diff_list.append(diff)
            print("episode %s:  diff= %s" % (episode_count, diff))
            if global_step % 100 == 0:
                self.save()
        self.show_convergence_curve(diff_list)

    # 通过epsilon贪心策略返回一个action
    def epsilon_greedy_action(self, theta, d_theta, epsilon):
        index_theta = self.value_to_index(theta, -self.max_theta, self.max_theta, self.n_theta)
        index_d_theta = self.value_to_index(d_theta, -self.max_d_theta, self.max_d_theta, self.n_d_theta)
        if np.random.rand() > 1-epsilon:  # 以1-epsilon的概率返回q最大的action
            index_action =  np.argmax(self.q_table[index_theta, index_d_theta])
            action = self.actions[index_action]
        else:  # 以1-epsilon的概率随机返回一个action
            action = np.random.choice(self.actions)
        action = action if isinstance(action, np.ndarray) else np.array([action])
        return action

    # 给定状态下，返回最优action
    def do_action(self, theta, d_theta):
        index_theta = self.value_to_index(theta, -self.max_theta, self.max_theta, self.n_theta)
        index_d_theta = self.value_to_index(d_theta, -self.max_d_theta, self.max_d_theta, self.n_d_theta)
        action_index = np.argmax(self.q_table[index_theta, index_d_theta])
        return np.array([self.actions[action_index]])


agent = SARSA(n_theta=500, n_d_theta=500, n_actions=3, gamma=0.98)
# agent.train(episode_num=2000, episode_length=300)
agent.load()
agent.show_histogram()
agent.show_contour()
# agent.show_q_table_3d()
# for i in range(5):
#     agent.demo()