"""
File: inv_pendulum.py
Author: Nathaniel Hamilton modified code from 
        https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py

Description: An update to the inverted pendulum environment that uses the same 
             initial values as our safe_pendulum.py implementation. The initial conditions and system constraints are the same
             used in https://rcheng805.github.io/files/aaai2019.pdf.
"""
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path


class InvPendulumEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, g=10.0, unsafe_angle=1.0, reward_style=1):
        """
        TODO: define input variables
        :reward_style: (int) value determining the reward style for training in the 
                             environment; 0 is minimizing cost, 1 is maximizing reward
        """
        self.max_speed = 60.0
        self.max_torque = 15.0
        self.dt = .05
        self.g = g
        self.m = 1.
        self.l = 1.
        self.viewer = None
        self.unsafe_angle = unsafe_angle
        self.reward_style = reward_style

        high = np.array([1., 1., self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque, shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

        self.seed()

    def calculate_reward(self, theta, thetadot, action):
        """
        TODO define inputs
        """
        costs = angle_normalize(theta) ** 2 + .1 * thetadot ** 2 + .001 * (action ** 2)

        # adjust the reward value depending on the reward style
        if self.reward_style == 1:
            reward = 5 - costs
        else:
            reward = -costs
        
        return reward

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u, evaluate=False):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt
        done = False

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering

        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])

        # terminate the episode if the safety condition is violated
        if abs(newth) > self.unsafe_angle:
            done = True

        reward = self.calculate_reward(newth, newthdot, u)

        return self._get_obs(), reward, done, {'theta': newth}

    def reset(self):
        high = np.array([0.8, 1.0])  # Pendulum starts +-46 degrees and +-1.0 rad/s
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)