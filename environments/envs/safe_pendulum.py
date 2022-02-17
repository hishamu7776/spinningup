"""
File: safe_pendulum.py
Author: Nathaniel Hamilton modified code from 
        https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py

Description: An update to the inverted pendulum environment that adds a safety constraint and a runtime assurance (RTA)
             strategy to ensure it. The initial conditions and system constraints are the same used in
             https://rcheng805.github.io/files/aaai2019.pdf. The RTA in this environment uses a simplex architecture
             (explained well in this paper http://www.taylortjohnson.com/research/johnson2016tecs.pdf).
             TODO: explain how the RTA works
"""
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path


class SafePendulumEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, g=10.0, unsafe_angle=1.0, reward_style=1):
        """
        TODO: define input variables
        :reward_style: (int) value determining the reward style for training in the 
                             environment; 0 is minimizing cost, 1 is maximizing reward,
                             2 is maximize reward and punish SC usage
        """
        self.max_speed = 60.0
        self.max_torque = 15.0
        self.dt = .05
        self.g = g
        self.m = 1.
        self.l = 1.
        self.viewer = None
        self.initial_angle_bound = 0.8  # Pendulum starts +-46 degrees and +-1.0 rad/s
        self.initial_vel_bound = 1.0
        self.unsafe_angle = unsafe_angle
        self.reward_style = reward_style
        self.look_ahead = 100

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
    
    def calculate_reward(self, theta, thetadot, action, intervening=False):
        """
        TODO define inputs
        """
        costs = angle_normalize(theta) ** 2 + .1 * thetadot ** 2 + .001 * (action ** 2)

        # adjust the reward value depending on the reward style
        if self.reward_style == 1:
            reward = 5 - costs
        elif self.reward_style == 2:
            if intervening:
                costs += 1  # an additional cost is applied if the safety controller is used
            reward = 5 - costs
        else:
            reward = -costs
        
        return reward

    def get_safe_action(self, theta, thetadot):
        """
        TODO: define inputs
        """
        # TODO: Maybe make this better because it is very basic
        # if theta < 0.0:
        #     safe_action = np.array([2])
        # else:
        #     safe_action = np.array([-2])
        
        u = np.array([((-32.0 / np.pi) * theta)])  # + ((-1.0 / np.pi) * thetadot)])
        safe_action = np.clip(u, -self.max_torque, self.max_torque)[0]
        # print(safe_action)

        return safe_action

    def runtime_assurance(self, action):
        """
        TODO: define variables
        """
        intervening = False
        safe_action = action
        simulated_done = False

        # Simulate one time-step forward with the desired action
        th, thdot = self.state
        next_theta, next_thetadot = self.simulate_one_step(th, thdot, action)
        simulated_next_observation = np.array([np.cos(next_theta), np.sin(next_theta), next_thetadot])
        simulated_reward = self.calculate_reward(next_theta, next_thetadot, action)
        if abs(next_theta) > self.unsafe_angle:
            intervening = True
            safe_action = self.get_safe_action(th, thdot)
            simulated_done = True
        else:
            # Simulate a few time-steps forward with the safety controller to ensure safety property won't be violated
            # in the future
            for i in range(self.look_ahead - 1):
                next_theta, next_thetadot = self.simulate_one_step(next_theta, next_thetadot, 
                                                                   self.get_safe_action(next_theta, next_thetadot))
                if abs(next_theta) > self.unsafe_angle:
                    intervening = True
                    safe_action = self.get_safe_action(th, thdot)
                    break

                # The look-ahead can also terminate as safe if the pendulum re-enters the initial conditions
                if abs(next_theta) <= self.initial_angle_bound and abs(next_thetadot) <= self.initial_vel_bound:
                    break

        return intervening, safe_action, simulated_next_observation, simulated_reward, simulated_done

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def simulate_one_step(self, th, thdot, u):
        """
        TODO: define inputs
        """
        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        return newth, newthdot

    def step(self, u):
        """
        TODO: define variables
        """
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt
        done = False

        u = np.clip(u, -self.max_torque, self.max_torque)[0]

        # determine the safety of the desired action
        intervening_bool, safe_u, sim_o2, sim_r, sim_d = self.runtime_assurance(u)

        self.last_u = safe_u  # for rendering

        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * safe_u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])

        # terminate the episode if the safety condition is violated
        if abs(newth) > self.unsafe_angle:
            # print('Safety violated')
            done = True

        reward = self.calculate_reward(newth, newthdot, safe_u, intervening_bool)
        
        info = {'intervening': intervening_bool, 'action taken': safe_u, 
                'sim r': sim_r, 'sim o2': sim_o2, 'sim d': sim_d, 'theta': newth}

        return self._get_obs(), reward, done, info

    def reset(self):
        high = np.array([self.initial_angle_bound, self.initial_vel_bound])
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
    return ((x + np.pi) % (2 * np.pi)) - np.pi
