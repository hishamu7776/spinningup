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
def do_rollouts(num_rollouts, policy1, policy2, policy3, policy4):
    """
    TODO
    """
    env = gym.make('Pendulum-v1')
    trajectories1 = []
    trajectories2 = []
    trajectories3 = []
    trajectories4 = []
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

        # Rollout policy4
        done = False
        history4 = []
        env.reset()
        env.state = np.array([init_theta, init_theta_dot])
        o = np.array([np.cos(init_theta), np.sin(init_theta), init_theta_dot])
        theta = init_theta
        while not done:
            history4.append(theta)
            o, r, done, info = env.step(policy4(o))
            theta = info['theta']
        trajectories4.append(history4)

    return trajectories1, trajectories2, trajectories3,trajectories4

def plot_trajectories(trajectories1, trajectories2, trajectories3, trajectories4, save_name):
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
        theta4 = trajectories4[i]
        axis.plot(theta4, np.linspace(0, len(theta4)-1, len(theta4)), 'c')
    theta1 = trajectories1[-1]
    axis.plot(theta1, np.linspace(0, len(theta1)-1, len(theta1)), 'b', label='baseline')
    theta2 = trajectories2[-1]
    axis.plot(theta2, np.linspace(0, len(theta2)-1, len(theta2)), 'g--', label='sparse')
    theta3 = trajectories3[-1]
    axis.plot(theta3, np.linspace(0, len(theta3)-1, len(theta3)), 'r-', label='dense')
    theta4 = trajectories4[-1]
    axis.plot(theta4, np.linspace(0, len(theta4)-1, len(theta4)), 'c', label='rml')
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


def do_rollout_cartpole(num_rollouts, policy1, policy2, policy3, policy4):
    """
    TODO
    """
    env = gym.make('CartPole-v0')
    trajectories1 = []
    trajectories2 = []
    trajectories3 = []
    trajectories4 = []
    for i in range(num_rollouts):
        # Rollout policy1
        done = False
        history1 = []
        o = env.reset()
        init_x, init_x_dot, init_theta, init_theta_dot = env.state
        while not done:
            history1.append(o)
            o, r, done, _ = env.step(policy1(o))
        trajectories1.append(history1)

        # Rollout policy2
        done = False
        history2 = []
        env.reset()
        env.state = (init_x, init_x_dot, init_theta, init_theta_dot)
        o = np.array([init_x, init_x_dot, init_theta, init_theta_dot])
        while not done:
            history2.append(o)
            o, r, done, _ = env.step(policy2(o))
        trajectories2.append(history2)

        # Rollout policy3
        done = False
        history3 = []
        env.reset()
        env.state = (init_x, init_x_dot, init_theta, init_theta_dot)
        o = np.array([init_x, init_x_dot, init_theta, init_theta_dot])
        while not done:
            history3.append(o)
            o, r, done, _ = env.step(policy3(o))
        trajectories3.append(history3)

        # Rollout policy4
        done = False
        history4 = []
        env.reset()
        env.state = (init_x, init_x_dot, init_theta, init_theta_dot)
        o = np.array([init_x, init_x_dot, init_theta, init_theta_dot])
        while not done:
            history4.append(o)
            o, r, done, _ = env.step(policy4(o))
        trajectories4.append(history4)

    return trajectories1, trajectories2, trajectories3,trajectories4

def plot_trajectories_cartpole(trajectories1, trajectories2, trajectories3, trajectories4, save_name):
    """
    TODO: trajectories1 is baseline
    trajectories2 is retrained
    """
    csfont = {'fontname': 'Times New Roman', 'fontsize': 20}
    fig, axis = plt.subplots(1, 2)
    axis = axis.flatten()

    """ Plot individual trajectory lines """
    for i in range(len(trajectories1)-1):
        x1, _, theta1, _ = zip(*trajectories1[i])
        axis[0].plot(x1, np.linspace(0, len(x1)-1, len(x1)), 'b')
        axis[1].plot(theta1, np.linspace(0, len(theta1)-1, len(theta1)), 'b')
        x2, _, theta2, _ = zip(*trajectories2[i])
        axis[0].plot(x2, np.linspace(0, len(x2)-1, len(x2)), 'g--')
        axis[1].plot(theta2, np.linspace(0, len(theta2)-1, len(theta2)), 'g--')
        x3, _, theta3, _ = zip(*trajectories3[i])
        axis[0].plot(x3, np.linspace(0, len(x3)-1, len(x3)), 'r-')
        axis[1].plot(theta3, np.linspace(0, len(theta3)-1, len(theta3)), 'r-')
        x4, _, theta4, _ = zip(*trajectories4[i])
        axis[0].plot(x4, np.linspace(0, len(x4)-1, len(x4)), 'c')
        axis[1].plot(theta4, np.linspace(0, len(theta4)-1, len(theta4)), 'c')
    x1, _, theta1, _ = zip(*trajectories1[-1])
    axis[0].plot(x1, np.linspace(0, len(x1)-1, len(x1)), 'b', label='baseline')
    axis[1].plot(theta1, np.linspace(0, len(theta1)-1, len(theta1)), 'b', label='baseline')
    x2, _, theta2, _ = zip(*trajectories2[-1])
    axis[0].plot(x2, np.linspace(0, len(x2)-1, len(x2)), 'g--', label='sparse')
    axis[1].plot(theta2, np.linspace(0, len(theta2)-1, len(theta2)), 'g--', label='sparse')
    x3, _, theta3, _ = zip(*trajectories3[-1])
    axis[0].plot(x3, np.linspace(0, len(x3)-1, len(x3)), 'r-', label='dense')
    axis[1].plot(theta3, np.linspace(0, len(theta3)-1, len(theta3)), 'r-', label='dense')
    x4, _, theta4, _ = zip(*trajectories4[-1])
    axis[0].plot(x4, np.linspace(0, len(x4)-1, len(x4)), 'c', label='rml')
    axis[1].plot(theta4, np.linspace(0, len(theta4)-1, len(theta4)), 'c', label='rml')
    """"""

    """ Add boundaries to plots """
    # Angle
    # axis.axvspan(-2.4, -0.5, color='red', alpha=0.2)
    # axis.axvspan(0.5, 2.4, color='red', alpha=0.2)
    # axis.add_patch(Rectangle((-1.5, 199), 3, 6, facecolor=mcolors.cnames['lime'], alpha=0.5, fill=True))
    # axis.add_patch(Rectangle((-1.5, 499), 3, 6, facecolor=mcolors.cnames['lime'], alpha=0.5, fill=True))
    # Position
    # Position
    axis[0].axvspan(-2.4, -0.5, color='red', alpha=0.2)
    axis[0].axvspan(0.5, 2.4, color='red', alpha=0.2)
    axis[0].add_patch(Rectangle((-1.5, 199), 3, 6, facecolor=mcolors.cnames['lime'], alpha=0.5, fill=True))
    # axis[0].add_patch(Rectangle((-1.5, 499), 3, 6, facecolor=mcolors.cnames['lime'], alpha=0.5, fill=True))
    axis[0].set_title("Position Trajectories", **csfont)
    axis[0].set_ylabel("Time", **csfont)
    axis[0].set_xlabel("Horizontal Position", **csfont)
    axis[0].set_xlim([-2.4, 2.4])
    axis[0].tick_params(axis='x', labelsize=14)
    axis[0].tick_params(axis='y', labelsize=14)
    axis[0].legend(loc='lower right', fontsize=12)

    # Angle
    axis[1].axvspan(-0.48, -0.20944, color='red', alpha=0.2)
    axis[1].axvspan(0.20944, 0.48, color='red', alpha=0.2)
    axis[1].add_patch(Rectangle((-0.48, 199), 0.94, 6, facecolor=mcolors.cnames['lime'], alpha=0.5, fill=True))
    # axis[1].add_patch(Rectangle((-0.48, 499), 0.94, 6, facecolor=mcolors.cnames['lime'], alpha=0.5, fill=True))
    axis[1].set_title("Angle Trajectories", **csfont)
    axis[1].set_ylabel("Time", **csfont)
    axis[1].set_xlabel("Angle (radians)", **csfont)
    axis[1].set_xlim([-0.48, 0.48])
    axis[1].tick_params(axis='x', labelsize=14)
    axis[1].tick_params(axis='y', labelsize=14)
    # axis[1].legend(loc='lower right', fontsize=12)

    plt.tight_layout()
    fig.savefig(save_name, bbox_inches='tight', dpi=200)
    # plt.show()

    return