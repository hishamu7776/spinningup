"""
file: example_bad_specs.py
author: Nathaniel Hamilton
email: nathaniel_hamilton@outlook.com

description:
    TODO

"""
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

# environment libraries
import stlgym
from environments import *


# Function for plotting evaluation traces
def do_rollouts_cartpole(num_rollouts, policy1):
    """
    TODO
    """
    env = gym.make('CartPole-v1')
    trajectories1 = []
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

    return trajectories1

def plot_trajectories_cartpole(trajectories1, save_name):
    csfont = {'fontname': 'Times New Roman', 'fontsize': 20}
    fig, axis = plt.subplots(1, 2)
    axis = axis.flatten()

    """ Plot individual trajectory lines """
    for i in range(len(trajectories1)-1):
        x1, _, theta1, _ = zip(*trajectories1[i])
        axis[0].plot(x1, np.linspace(0, len(x1)-1, len(x1)), 'b')
        axis[1].plot(theta1, np.linspace(0, len(theta1)-1, len(theta1)), 'b')
    x1, _, theta1, _ = zip(*trajectories1[-1])
    axis[0].plot(x1, np.linspace(0, len(x1)-1, len(x1)), 'b')
    axis[1].plot(theta1, np.linspace(0, len(theta1)-1, len(theta1)), 'b')
    """"""

    """ Add boundaries to plots """
    # Position
    axis[0].axvspan(-2.4, -0.5, color='red', alpha=0.2)
    axis[0].axvspan(0.5, 2.4, color='red', alpha=0.2)
    axis[0].add_patch(Rectangle((-1.5, 199), 3, 6, facecolor=mcolors.cnames['lime'], alpha=0.5, fill=True))
    axis[0].add_patch(Rectangle((-1.5, 499), 3, 6, facecolor=mcolors.cnames['lime'], alpha=0.5, fill=True))
    axis[0].set_title("Position Trajectories", **csfont)
    axis[0].set_ylabel("Time", **csfont)
    axis[0].set_xlabel("Horizontal Position", **csfont)
    axis[0].set_xlim([-2.4, 2.4])
    axis[0].tick_params(axis='x', labelsize=14)
    axis[0].tick_params(axis='y', labelsize=14)
    axis[0].legend(loc='lower right', fontsize=12)

    # Angle
    axis[1].axvspan(-0.48, -0.0872665, color='red', alpha=0.2)
    axis[1].axvspan(0.0872665, 0.48, color='red', alpha=0.2)
    axis[1].add_patch(Rectangle((-0.48, 199), 0.94, 6, facecolor=mcolors.cnames['lime'], alpha=0.5, fill=True))
    axis[1].add_patch(Rectangle((-0.48, 499), 0.94, 6, facecolor=mcolors.cnames['lime'], alpha=0.5, fill=True))
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

def do_rollouts_pendulum(num_rollouts, policy1):
    """
    TODO
    """
    env = gym.make('Pendulum-v1')
    trajectories1 = []
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

    return trajectories1

def plot_trajectories_pendulum(trajectories1, save_name):
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
        axis.plot(theta1, np.linspace(0, len(theta1)-1, len(theta1))) #, 'b')
    theta1 = trajectories1[-1]
    axis.plot(theta1, np.linspace(0, len(theta1)-1, len(theta1)), 'b')
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

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-all', help="Train all experimental cases.", action="store_true")
    parser.add_argument('--pendulum', help="TODO", action="store_true")
    parser.add_argument('--cartpole',
                        help="TODO",
                        action="store_true")
    parser.add_argument('--config-path', help='Path to STLGym config folder with specification.', type=str, default="./configs/")
    parser.add_argument('--plot-all', help="Generate plots for all experimental cases.", action="store_true")
    parser.add_argument('--plot-traces', help="TODO", action="store_true")
    parser.add_argument('--table', help="Generate text for latex table comparing final performance of all experimental cases.",
                        action="store_true")
    parser.add_argument('--show-seeds', help="Split the table results to show individual random seed results.",
                        action="store_true")
    parser.add_argument('--num-evals', help="Set the number of evaluations to occur after training. Default: 100",
                        type=int, default=50)
    args = vars(parser.parse_args())

    # Setting up the log directory and config file paths for STLGym environments TODO: finish adding steps here
    cwd = os.getcwd()
    if "spinup" in cwd:
        if "examples" in cwd:
            if "sefm2022" in cwd:
                log_directory = "./logs/ex_bad_specs/"
                fig_directory = "./logs/figures/ex_bad_specs/"
            else:
                log_directory = "./sefm2022/logs/ex_bad_specs/"
                fig_directory = "./sefm2022/logs/figures/ex_bad_specs/"
        else:
            log_directory = "./examples/sefm2022/logs/ex_bad_specs/"
            fig_directory = "./examples/sefm2022/logs/figures/ex_bad_specs/"
    else:
        log_directory = "/tmp/logs/ppo/cartpole/"
        fig_directory = "/tmp/logs/ppo/cartpole/figures/ex_bad_specs/"
    stl_env_config_pendulum = args['config_path'] + "ex_bad_specs_pendulum.yaml"
    stl_env_config_pendulum_eval = args['config_path'] + "ex_bad_specs_pendulum_eval.yaml"
    stl_env_config_cartpole = args['config_path'] + "ex_bad_specs_cartpole.yaml"
    stl_env_config_cartpole_eval = args['config_path'] + "ex_bad_specs_cartpole_eval.yaml"

    # Hyperparameters
    random_seeds = [1630, 2241, 2320] # , 2990, 3281, 4930, 5640, 8005, 9348, 9462]
    ac_kwargs = dict(hidden_sizes=(64, 64,))
    steps_per_epoch = 4000
    epochs = 100
    gamma = 0.99  
    clip_ratio = 0.2
    pi_lr = 3e-4
    vf_lr = 1e-3
    train_pi_iters = 80
    train_v_iters = 80
    lam = 0.97
    num_test_episodes = 10
    max_ep_len = 200
    target_kl = 0.01
    save_freq = 1
    num_evals = args['num_evals']

    # Experiment names for plotting
    exp1_name = "pendulum"
    exp2_name = "cartpole"
    plot_legend = [exp1_name] #, exp2_name]

    if args['train_all']:
        # Overwrite the default false values to train all the experiments
        args['pendulum'] = True
        args['cartpole'] = False # Specification is actually good

    # pendulum performance
    if args['pendulum']:
        for i in range(len(random_seeds)):
            env_fn = partial(stlgym.make, stl_env_config_pendulum)
            test_env_fn = partial(gym.make, 'Pendulum-v1')
            alt_test_env_fn = partial(stlgym.make, stl_env_config_pendulum_eval)
            log_dest = log_directory + "pendulum/rand_seed_" + str(random_seeds[i])
            logger_kwargs = dict(output_dir=log_dest, exp_name=exp1_name)
            print(f"Training PPO pendulum, random seed: {random_seeds[i]}...")
            ppo(env_fn, test_env_fn=test_env_fn, alt_test_env_fn=alt_test_env_fn, ac_kwargs=ac_kwargs, seed=random_seeds[i], 
                steps_per_epoch=steps_per_epoch, epochs=epochs, gamma=gamma, clip_ratio=clip_ratio, pi_lr=pi_lr,
                vf_lr=vf_lr, train_pi_iters=train_pi_iters, train_v_iters=train_v_iters, lam=lam, num_test_episodes=num_test_episodes, 
                max_ep_len=max_ep_len, target_kl=target_kl, logger_kwargs=logger_kwargs, save_freq=save_freq)

            # After the policy is trained, evaluate it for a given number of steps
            env, get_action = load_policy_and_env(fpath=log_dest, itr='last', deterministic=True)
            original_env = gym.make('Pendulum-v1')
            stl_env = stlgym.make(stl_env_config_pendulum_eval)
            evaluate_policy_in_2_environments(env1=original_env, env2=stl_env, get_action=get_action, log_dest=log_dest, max_ep_len=200, num_episodes=num_evals)

    # Retraining with STLGym reward function
    if args['cartpole']:
        for i in range(len(random_seeds)):
            env_fn = partial(stlgym.make, stl_env_config_cartpole)
            test_env_fn = partial(gym.make, 'CartPole-v0')
            alt_test_env_fn = partial(stlgym.make, stl_env_config_cartpole_eval)
            log_dest = log_directory + "cartpole/rand_seed_" + str(random_seeds[i])
            logger_kwargs = dict(output_dir=log_dest, exp_name=exp2_name)
            print(f"Training PPO cartpole, random seed: {random_seeds[i]}...")
            ppo(env_fn, test_env_fn=test_env_fn, alt_test_env_fn=alt_test_env_fn, ac_kwargs=ac_kwargs, seed=random_seeds[i], 
                steps_per_epoch=steps_per_epoch, epochs=epochs, gamma=gamma, clip_ratio=clip_ratio, pi_lr=pi_lr,
                vf_lr=vf_lr, train_pi_iters=train_pi_iters, train_v_iters=train_v_iters, lam=lam, num_test_episodes=num_test_episodes, 
                max_ep_len=max_ep_len, target_kl=target_kl, logger_kwargs=logger_kwargs, save_freq=save_freq)

            # After the policy is trained, evaluate it for a given number of steps
            env, get_action = load_policy_and_env(fpath=log_dest, itr='last', deterministic=True)
            original_env = gym.make('CartPole-v0')
            stl_env = stlgym.make(stl_env_config_cartpole_eval)
            evaluate_policy_in_2_environments(env1=original_env, env2=stl_env, get_action=get_action, log_dest=log_dest, max_ep_len=200, num_episodes=num_evals)
    
    if args['plot_all']:
        log_dirs = []
        for i in range(len(plot_legend)):
            log_dirs.append(log_directory + plot_legend[i])
        save_name = fig_directory + "sample_complexity_baseline.png"
        make_plots(log_dirs, legend=plot_legend, xaxis='TotalEnvInteracts', values=['AverageTestEpRet'],
                #    ylim=(0, 1100), 
                   count=False, smooth=1, select=None, exclude=None, estimator='mean', save_name=save_name)

        save_name = fig_directory + "sample_complexity_stl.png"
        make_plots(log_dirs, legend=plot_legend, xaxis='TotalEnvInteracts', values=['AverageAltTestEpRet'],
                #    ylim=(0, 1100), 
                   count=False, smooth=1, select=None, exclude=None, estimator='mean', save_name=save_name)

        save_name = fig_directory + "episode_length_baseline.png"
        make_plots(log_dirs, legend=plot_legend, xaxis='TotalEnvInteracts', values=['TestEpLen'],
                #    ylim=(0, 240), 
                   count=False, smooth=1, select=None, exclude=None, estimator='mean', save_name=save_name)

        save_name = fig_directory + "episode_length_stl.png"
        make_plots(log_dirs, legend=plot_legend, xaxis='TotalEnvInteracts', values=['AltTestEpLen'],
                #    ylim=(0, 240), 
                   count=False, smooth=1, select=None, exclude=None, estimator='mean', save_name=save_name)
    
    if args['plot_traces']:
        # Ensure the figure directory exists
        if not os.path.exists(fig_directory):
            os.mkdir(fig_directory)
        
        # Generate the trace figures for all random seeds with cartpole
        # for i in random_seeds:
        #     log_dest_cartpole = log_directory + "cartpole/rand_seed_" + str(i)
        #     save_name = fig_directory + "bad_cartpole_rand_seed_" + str(i) + "_traces.png"
        #     _, get_action = load_policy_and_env(fpath=log_dest_cartpole, itr='last', deterministic=True)
        #     trajectories1 = do_rollouts_cartpole(num_evals, get_action)
        #     plot_trajectories_cartpole(trajectories1, save_name)
        
        # Generate the trace figures for all random seeds with pendulum
        for i in random_seeds:
            log_dest_pendulum = log_directory + "pendulum/rand_seed_" + str(i)
            save_name = fig_directory + "bad_pendulum_rand_seed_" + str(i) + "_traces.png"
            _, get_action = load_policy_and_env(fpath=log_dest_pendulum, itr='last', deterministic=True)
            trajectories1 = do_rollouts_pendulum(2, get_action)
            plot_trajectories_pendulum(trajectories1, save_name)

    if args['table']:
        log_dirs = []
        for i in range(len(plot_legend)):
            log_dirs.append(log_directory + plot_legend[i])
        make_table(log_dirs, legend=plot_legend, separate=args['show_seeds'], select=None, exclude=None)

