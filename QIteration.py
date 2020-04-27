"""
重写的离散化Q迭代
"""

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from time import sleep
import matplotlib.pyplot as plt
import os

from game.my_Pendulum import MyPendulumEnv, angle_normalize



class DiscreteQIteration(object):
    def __init__(self, n_theta=200, n_d_theta=201, n_actions=3, gamma=0.98):
        self.model_name = self.__class__.__name__  # 模型名称

        self.gamma = gamma  # 衰减因子
        self.n_theta = n_theta  # 角度划分为多少个区间
        self.n_d_theta = n_d_theta  # 角速度划分为多少个区间
        self.n_actions = n_actions  # action(电压)划分为多少个区间

        self.max_theta = np.pi  # 最大角度
        self.max_d_theta = 15 * np.pi  # 最大角速度
        self.max_action = 3.  # 最大动作（电压）

        self.actions = np.linspace(-self.max_action, self.max_action, self.n_actions)  # 动作空间
        self.env = MyPendulumEnv()  # 交互环境
        self.q_table = np.zeros((self.n_theta, self.n_d_theta, len(self.actions)))  # Q(state, action)表，其中state=(theta, d_theta)

        self.data_dir = "./result/%s_th-%s_dth-%s_a-%s/" % (self.model_name, self.n_theta, self.n_d_theta, self.n_actions)  # 数据保存路径
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)



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

    # 把离散区间id映射成区间的值（以中点的值为代表）参数：区间id，区间起始值，区间终止值，区间总数
    def index_to_value(self, index, start, end, numbers):
        one_length = (end-start) / numbers  # 区间的单位长度
        return start + one_length /2 + index * one_length

    # 把连续值映射成离散区间id。参数：离散值，区间起始值，区间终止值，区间总数
    def value_to_index(self, value, start, end, numbers):
        value = value if isinstance(value, np.ndarray) else np.array([value])
        index = ((value-start) / (end-start) * numbers).astype(int)
        index[index==numbers] -= 1
        return index

    # 给定状态下，返回最优action
    def do_action(self, theta, d_theta):
        index_theta = self.value_to_index(theta, -self.max_theta, self.max_theta, self.n_theta)
        index_d_theta = self.value_to_index(d_theta, -self.max_d_theta, self.max_d_theta, self.n_d_theta)
        action_index = np.argmax(self.q_table[index_theta, index_d_theta])
        return np.array([self.actions[action_index]])

    # 通过gym将agent与环境的交互过程可视化
    def demo(self, max_step=1000):
        observation = self.env.reset()
        self.env.render()
        sleep(1)
        step = 0
        while True:
            step += 1
            self.env.render()  # 渲染动画
            theta, thetadot = observation
            action = self.do_action(theta, thetadot)
            observation, reward, done, info = self.env.step(action)
            if done or step>max_step:
                print("Episode finished after {} timesteps".format(step + 1))
                break
        sleep(2)
        self.env.close()

    def save(self):
        np.save(self.data_dir + "checkpoint.npy", self.q_table)

    # 加载已训练好的q_table
    def load(self, file=""):
        file = file if file else self.data_dir + "checkpoint.npy"
        self.q_table = np.load(file)

    # 画收敛曲线
    def show_convergence_curve(self, diff_list):
        fig, ax = plt.subplots()
        plot = ax.plot(np.arange(1, len(diff_list)+1), diff_list)

        # 子图ax的标题
        ax.set_title("2-Norm of Q_tables between each two steps\n theta=%s, dot_theta=%s, action=%s" % (self.n_theta, self.n_d_theta, self.n_actions))  # y可以调整标题的位置

        # 设置坐标轴格式
        # x轴（step）
        ax.set_xlabel('step')  # x轴title

        # y轴（二范数）
        ax.set_ylabel('2-Norm')  # y轴title
        # 保存与显示
        fig.tight_layout()  # 适应窗口大小，否则可能有东西在窗口里画不下
        fig.savefig(self.data_dir + "convergence.png")
        fig.canvas.set_window_title(self.model_name + ": convergence")  # 窗口fig的title
        fig.show()


    # 用热力图(histogram)可视化每个状态对应的动作
    def show_histogram(self):
        m_index_actions = np.argmax(self.q_table, axis=2)  # 每个(速度，角速度)状态对应的最优action的index
        m_action_of_state = self.actions[m_index_actions]  # 把action index转换成action值
        m_action_of_state = (m_action_of_state-np.min(self.actions)) / (np.max(self.actions) - np.min(self.actions))  # 归一化，方便画热力图

        # 画图过程（顺便练习matplotlib画图的方法）
        # 创建窗口fig和子图ax
        fig, ax = plt.subplots()

        # 绘制热力图
        cmap = plt.get_cmap('jet', self.n_actions)  # 创建颜色映射，有self.n_actions个离散值
        histogram = ax.matshow(m_action_of_state, cmap=cmap, vmin=0, vmax=1)  # 子图ax绘制热力图

        # 添加颜色条
        cbar = fig.colorbar(histogram, ax=ax, label="Action: voltage (V)", drawedges=True)  # 添加颜色条
        cbar.set_ticks((np.arange(self.n_actions)+0.5) / self.n_actions)  # 在颜色条哪些位置设置刻度线。颜色条的每个刻度线的相对位置在[0,1]之间。超过[0,1]的刻度线画不上去
        cbar.set_ticklabels(np.round(self.actions, decimals=3))  # 颜色条的每个刻度上应该写什么（这里是保留两位小数的action值）

        # 子图ax的标题
        ax.set_title("Action of each State\n theta=%s, dot_theta=%s, action=%s" % (self.n_theta, self.n_d_theta, self.n_actions))
        # 设置坐标轴格式
        # x轴（角速度）
        ax.set_xlabel('State: dot_theta (rad/s)')  # x轴title
        ax.xaxis.tick_bottom()  # x轴的刻度放在下面
        # 设置x轴刻度
        x_start, x_end = ax.get_xlim()  # 刻度的起止点。和颜色条的刻度不同，histogram上的x、y轴是有起止点的，刻度线相对位置要在起止点以内
        ax.xaxis.set_ticks(np.linspace(x_start, x_end, 7))  # 设置刻度线位置
        ax.xaxis.set_ticklabels(np.round(np.linspace(-self.max_d_theta, self.max_d_theta, 7), decimals=2))  # 设置刻度线上写什么（保留两位小数的角速度）

        # y轴（角度）
        ax.set_ylabel('State: theta (rad)')  # y轴title
        y_start, y_end = ax.get_ylim()  # 刻度的起止点。和颜色条的刻度不同，histogram上的x、y轴是有起止点的，刻度线相对位置要在起止点以内
        ax.yaxis.set_ticks(np.linspace(y_start, y_end, 7))  # 设置刻度线位置
        ax.yaxis.set_ticklabels(np.round(np.linspace(-self.max_theta, self.max_theta, 7), decimals=2))  # 设置刻度线上写什么（保留两位小数的角速度）

        # 保存与显示
        fig.tight_layout()  # 适应窗口大小，否则可能有东西在窗口里画不下
        fig.savefig(self.data_dir + "histogram.png")
        fig.canvas.set_window_title(self.model_name + ": histogram")  # 窗口fig的title
        fig.show()  # 展示窗口fig里的所有子图（此处只有一个子图ax）





agent = DiscreteQIteration(n_theta=200, n_d_theta=200, n_actions=11, gamma=0.98)
agent.train(epsilon=0.1)
agent.load()
agent.show_histogram()
# for i in range(5):
#     agent.demo()