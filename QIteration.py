"""
离散化Q迭代
"""

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from time import sleep
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import gym

from game.my_Pendulum import MyPendulumEnv, angle_normalize
from BaseAgent import BaseAgent



class DiscreteQIteration(BaseAgent):

    def train(self, epsilon=0.01):
        # 注:以 m_ 开头的变量都是np.ndarray类型，可以利用numpy并行计算，速度飞升
        # 存储每个离散状态的index的两个meshgrid，便于后面并行计算
        m_index_d_theta, m_index_theta = np.meshgrid(np.arange(self.n_d_theta), np.arange(self.n_theta))
        # 每个离散区间的代表值
        m_theta = self.index_to_value(m_index_theta, -self.max_theta, self.max_theta, self.n_theta)
        m_d_theta = self.index_to_value(m_index_d_theta, -self.max_d_theta, self.max_d_theta, self.n_d_theta)
        step = 0
        diff_list = []
        while True:
            step += 1
            new_q_table = np.zeros_like(self.q_table)
            m_max_q = np.max(self.q_table, axis=2)
            for i, u in enumerate(self.actions):
                # 新状态的角度、角速度
                m_next_theta, m_next_d_theta = self.env.new_state(m_theta, m_d_theta, u)
                # 新状态的角度、角速度转换成离散区间id
                m_index_next_theta = self.value_to_index(m_next_theta, -self.max_theta, self.max_theta, self.n_theta)
                m_index_next_d_theta = self.value_to_index(m_next_d_theta, -self.max_d_theta, self.max_d_theta, self.n_d_theta)
                m_reward = self.env.reward(m_theta, m_d_theta, u)  # reward
                new_q_table[:, :, i] = m_reward + self.gamma * m_max_q[m_index_next_theta, m_index_next_d_theta]  # Q值迭代
            diff = np.linalg.norm(new_q_table - self.q_table)
            diff_list.append(diff)
            self.q_table = new_q_table
            print("step %s:  diff= %s" % (step, diff))
            if step % 100 == 0:
                self.save()
            if diff < epsilon:
                print("finish")
                break
        self.show_convergence_curve(diff_list)

    # 给定状态下，返回最优action
    def do_action(self, theta, d_theta):
        index_theta = self.value_to_index(theta, -self.max_theta, self.max_theta, self.n_theta)
        index_d_theta = self.value_to_index(d_theta, -self.max_d_theta, self.max_d_theta, self.n_d_theta)
        action_index = np.argmax(self.q_table[index_theta, index_d_theta])
        return np.array([self.actions[action_index]])


if __name__ == "__main__":

    agent = DiscreteQIteration(n_theta=200, n_d_theta=200, n_actions=3, gamma=0.98)
    # agent.train(epsilon=0.1)  # 训练模型
    agent.load()  # load参数
    # agent.show_histogram()  # 画策略热力图
    # agent.show_contour()  # 画action的Q值等高线
    # agent.show_q_table_3d()  # 画一个action的三维Q值曲面
    # agent.demo(max_step=2000)  # 演示
    agent.demo(save_video=True)  # 保存视频
    agent.env_close()
