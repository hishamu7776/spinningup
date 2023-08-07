# general libraries
import gym
from matplotlib.patches import Rectangle
import torch
import os
from numpy.lib.function_base import append
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import colors as mcolors
import seaborn as sns

# spinup-specific
from spinup import ppo_pytorch as ppo
from spinup.utils.test_policy import *
from spinup.utils.plot import *
from spinup.utils.tables import *

#from rmlgym import RMLGym
from environments import *


# Function for plotting evaluation traces
def do_rollouts(num_rollouts, policy1, policy2, policy3):
    """
    TODO
    """
    env = gym.make('Pendulum-v1')
    trajectories1 = []
    trajectories2 = []
    trajectories3 = []
    for i in range(num_rollouts):
        # Rollout policy1
        done = False
        history1 = []
        o = env.reset()
        init_theta, init_theta_dot = env.state
        theta = init_theta
        while not done:
            history1.append(theta)
            o, r, done, info = env.step(policy1(o))
            theta = info['theta']
        trajectories1.append(history1)

        # Rollout policy2
        done = False
        history2 = []
        env.reset()
        env.state = np.array([init_theta, init_theta_dot])
        o = np.array([np.cos(init_theta), np.sin(init_theta), init_theta_dot])
        theta = init_theta
        while not done:
            history2.append(theta)
            o, r, done, info = env.step(policy2(o))
            theta = info['theta']
        trajectories2.append(history2)

        # Rollout policy3
        done = False
        history3 = []
        env.reset()
        env.state = np.array([init_theta, init_theta_dot])
        o = np.array([np.cos(init_theta), np.sin(init_theta), init_theta_dot])
        theta = init_theta
        while not done:
            history3.append(theta)
            o, r, done, info = env.step(policy3(o))
            theta = info['theta']
        trajectories3.append(history3)

    return trajectories1, trajectories2, trajectories3

def plot_trajectories(trajectories1, trajectories2, trajectories3, save_name):
    """
    TODO: trajectories1 is baseline
    trajectories2 is retrained
    """
    csfont = {'fontname': 'Times New Roman', 'fontsize': 20}
    fig, axis = plt.subplots(1, 1)
    # axis = axis.flatten()

    """ Plot individual trajectory lines """
    for i in range(len(trajectories1)-1):
        theta1 = trajectories1[i]
        axis.plot(theta1, np.linspace(0, len(theta1)-1, len(theta1)), 'b')
        theta2 = trajectories2[i]
        axis.plot(theta2, np.linspace(0, len(theta2)-1, len(theta2)), 'g--')
        theta3 = trajectories3[i]
        axis.plot(theta3, np.linspace(0, len(theta3)-1, len(theta3)), 'r-')
    theta1 = trajectories1[-1]
    axis.plot(theta1, np.linspace(0, len(theta1)-1, len(theta1)), 'b', label='baseline')
    theta2 = trajectories2[-1]
    axis.plot(theta2, np.linspace(0, len(theta2)-1, len(theta2)), 'g--', label='sparse')
    theta3 = trajectories3[-1]
    axis.plot(theta3, np.linspace(0, len(theta3)-1, len(theta3)), 'r-', label='dense')
    """"""

    """ Add boundaries to plots """
    # Angle
    # axis.axvspan(-2.4, -0.5, color='red', alpha=0.2)
    # axis.axvspan(0.5, 2.4, color='red', alpha=0.2)
    # axis.add_patch(Rectangle((-1.5, 199), 3, 6, facecolor=mcolors.cnames['lime'], alpha=0.5, fill=True))
    # axis.add_patch(Rectangle((-1.5, 499), 3, 6, facecolor=mcolors.cnames['lime'], alpha=0.5, fill=True))
    axis.set_title("Angle Trajectories", **csfont)
    axis.set_ylabel("Time", **csfont)
    axis.set_xlabel("Angle (radians)", **csfont)
    axis.set_xlim([-np.pi, np.pi])
    axis.tick_params(axis='x', labelsize=14)
    axis.tick_params(axis='y', labelsize=14)
    axis.legend(loc='lower right', fontsize=12)

    plt.tight_layout()
    fig.savefig(save_name, bbox_inches='tight', dpi=200)
    # plt.show()

    return
