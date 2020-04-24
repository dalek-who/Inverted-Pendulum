"""
按实验任务书的参数，定义的倒立摆环境
"""
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import time

class MyPendulumEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):
        self.m = 0.055
        self.g = 9.81
        self.l = 0.042
        self.J = 1.91 * 10 **(-4)
        self.b = 3. * 10 ** (-6)
        self.K = 0.0536
        self.R = 9.5

        self.max_theta = np.pi
        self.max_speed = 15 * np.pi
        self.max_u = 3.
        self.dt = 0.005  # 采样时间间隔

        self.viewer = None

        high = np.array([1., 1., self.max_speed], dtype=np.float32)
        self.action_space = spaces.Discrete(3) # -3,0,3三种随机动作。但是采样只能采样出0,1,2，需要进一步映射成-3,0,3
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        th, thdot = self.state # th := theta, 角度；thdot：角速度

        m = self.m
        g = self.g
        l = self.l
        J = self.J
        b = self.b
        K = self.K
        R = self.R
        dt = self.dt

        u = np.clip(u, -self.max_u, self.max_u)[0]
        self.last_u = u # for rendering
        costs = 5 * angle_normalize(th)**2 + .1 * thdot**2 + u**2  # cost是reward的相反数，是正的。reward是负的

        thdotdot = (1/J) * (m * g * l * np.sin(th) - b*thdot - (K**2/R)*thdot + (K/R)*u )  # 角加速度
        newthdot = thdot + dt * thdotdot  # 角速度更新
        newth = th + newthdot*dt  # 角度更新
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111

        self.state = np.array([newth, newthdot])  # 状态更新
        done = np.equal(self.state, np.array([0., 0.])).all()  # 是否达到终止状态
        return self._get_obs(), -costs, done, {}

    def reset(self):
        high = np.array([np.pi, 1])
        self.state = np.array([np.pi, 0])
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([theta, thetadot])

    def render(self, mode='human'):  # 渲染图像

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0,0,0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi/2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def action_sample(self):  # 把0，1，2策略映射成-3,0,3。只是简单地均匀采样。如果想按照策略采样需要自己额外定义
        action_map = {0: np.array([-3.]), 1: np.array([0.]), 2: np.array([3.])}
        action = env.action_space.sample()
        action = action_map[action]
        return action


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)


if __name__=="__main__":
    env = MyPendulumEnv()
    for i_episode in range(3):
        observation = env.reset()  # 重置游戏
        env.render()
        time.sleep(1)
        for t in range(1000):
            env.render()  # 渲染动画
            print(observation)
            action = env.action_sample()  # 采样动作（实际需要按学习到的策略采样。这里只是个demo），只接受-3，0，3
            observation, reward, done, info = env.step(action)   # 执行策略。observation=np.array([角度, 角速度])
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()