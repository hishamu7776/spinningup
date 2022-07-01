"""
file: example_split_spec_cartpole.py
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
def do_rollouts(num_rollouts, policy1, policy2, policy3):
    """
    TODO
    """
    env = gym.make('CartPole-v1')
    trajectories1 = []
    trajectories2 = []
    trajectories3 = []
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

    return trajectories1, trajectories2, trajectories3

def plot_trajectories(trajectories1, trajectories2, trajectories3, save_name):
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
        axis[0].plot(x1, np.linspace(0, len(x1)-1, len(x1)), 'b--')
        axis[1].plot(theta1, np.linspace(0, len(theta1)-1, len(theta1)), 'b--')
        x2, _, theta2, _ = zip(*trajectories2[i])
        axis[0].plot(x2, np.linspace(0, len(x2)-1, len(x2)), 'g--')
        axis[1].plot(theta2, np.linspace(0, len(theta2)-1, len(theta2)), 'g--')
        x3, _, theta3, _ = zip(*trajectories3[i])
        axis[0].plot(x3, np.linspace(0, len(x3)-1, len(x3)), 'r--')
        axis[1].plot(theta3, np.linspace(0, len(theta3)-1, len(theta3)), 'r--')
    x1, _, theta1, _ = zip(*trajectories1[-1])
    axis[0].plot(x1, np.linspace(0, len(x1)-1, len(x1)), 'b', label='single')
    axis[1].plot(theta1, np.linspace(0, len(theta1)-1, len(theta1)), 'b', label='sinlge')
    x2, _, theta2, _ = zip(*trajectories2[-1])
    axis[0].plot(x2, np.linspace(0, len(x2)-1, len(x2)), 'g--', label='split')
    axis[1].plot(theta2, np.linspace(0, len(theta2)-1, len(theta2)), 'g--', label='split')
    x3, _, theta3, _ = zip(*trajectories3[-1])
    axis[0].plot(x3, np.linspace(0, len(x3)-1, len(x3)), 'r--', label='stlgym')
    axis[1].plot(theta3, np.linspace(0, len(theta3)-1, len(theta3)), 'r--', label='stlgym')
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-all', help="Train all experimental cases.", action="store_true")
    parser.add_argument('--single', help="TODO", action="store_true")
    parser.add_argument('--split',
                        help="TODO",
                        action="store_true")
    parser.add_argument('--stlgym',
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
                log_directory = "./logs/ex_split_spec_cartpole/"
                fig_directory = "./logs/figures/ex_split_spec_cartpole/"
            else:
                log_directory = "./sefm2022/logs/ex_split_spec_cartpole/"
                fig_directory = "./sefm2022/logs/figures/ex_split_spec_cartpole/"
        else:
            log_directory = "./examples/sefm2022/logs/ex_split_spec_cartpole/"
            fig_directory = "./examples/sefm2022/logs/figures/ex_split_spec_cartpole/"
    else:
        log_directory = "/tmp/logs/ppo/cartpole/"
        fig_directory = "/tmp/logs/ppo/cartpole/figures/ex_split_spec_cartpole/"
    single_env_config = args['config_path'] + "ex_split_spec_cartpole_single.yaml"
    split_env_config = args['config_path'] + "ex_split_spec_cartpole_split.yaml"
    stlgym_env_config = args['config_path'] + "ex_split_spec_cartpole_stlgym.yaml"
    stl_env_config_eval = args['config_path'] + "ex_split_spec_cartpole_eval.yaml"

    # Hyperparameters
    random_seeds = [1630, 2241, 2320, 2990, 3281, 4930, 5640, 8005, 9348, 9462]
    ac_kwargs = dict(hidden_sizes=(64, 64,))
    steps_per_epoch = 4000
    epochs = 50
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
    exp1_name = "single"
    exp2_name = "split"
    exp3_name = "stlgym"
    plot_legend = [exp1_name, exp2_name, exp3_name]

    if args['train_all']:
        # Overwrite the default false values to train all the experiments
        args['single'] = True
        args['split'] = True
        args['stlgym'] = True

    # single performance
    if args['single']:
        for i in range(len(random_seeds)):
            env_fn = partial(stlgym.make, single_env_config)
            test_env_fn = partial(stlgym.make, stl_env_config_eval)
            log_dest = log_directory + "single/rand_seed_" + str(random_seeds[i])
            logger_kwargs = dict(output_dir=log_dest, exp_name=exp1_name)
            print(f"Training PPO single, random seed: {random_seeds[i]}...")
            ppo(env_fn, test_env_fn=test_env_fn, ac_kwargs=ac_kwargs, seed=random_seeds[i], 
                steps_per_epoch=steps_per_epoch, epochs=epochs, gamma=gamma, clip_ratio=clip_ratio, pi_lr=pi_lr,
                vf_lr=vf_lr, train_pi_iters=train_pi_iters, train_v_iters=train_v_iters, lam=lam, num_test_episodes=num_test_episodes, 
                max_ep_len=max_ep_len, target_kl=target_kl, logger_kwargs=logger_kwargs, save_freq=save_freq)

            # After the policy is trained, evaluate it for a given number of steps
            env, get_action = load_policy_and_env(fpath=log_dest, itr='last', deterministic=True)
            original_env = gym.make('CartPole-v0')
            stl_env = stlgym.make(stl_env_config_eval)
            evaluate_policy_in_2_environments(env1=original_env, env2=stl_env, get_action=get_action, log_dest=log_dest, max_ep_len=200, num_episodes=num_evals)

    # Spliting the spec without weights
    if args['split']:
        for i in range(len(random_seeds)):
            env_fn = partial(stlgym.make, split_env_config)
            test_env_fn = partial(stlgym.make, stl_env_config_eval)
            log_dest = log_directory + "split/rand_seed_" + str(random_seeds[i])
            logger_kwargs = dict(output_dir=log_dest, exp_name=exp2_name)
            print(f"Training PPO split, random seed: {random_seeds[i]}...")
            ppo(env_fn, test_env_fn=test_env_fn, ac_kwargs=ac_kwargs, seed=random_seeds[i], 
                steps_per_epoch=steps_per_epoch, epochs=epochs, gamma=gamma, clip_ratio=clip_ratio, pi_lr=pi_lr,
                vf_lr=vf_lr, train_pi_iters=train_pi_iters, train_v_iters=train_v_iters, lam=lam, num_test_episodes=num_test_episodes, 
                max_ep_len=max_ep_len, target_kl=target_kl, logger_kwargs=logger_kwargs, save_freq=save_freq)

            # After the policy is trained, evaluate it for a given number of steps
            env, get_action = load_policy_and_env(fpath=log_dest, itr='last', deterministic=True)
            original_env = gym.make('CartPole-v0')
            stl_env = stlgym.make(stl_env_config_eval)
            evaluate_policy_in_2_environments(env1=original_env, env2=stl_env, get_action=get_action, log_dest=log_dest, max_ep_len=200, num_episodes=num_evals)
    
    # Spliting the spec with weights (STLGym exclusive)
    if args['stlgym']:
        for i in range(len(random_seeds)):
            env_fn = partial(stlgym.make, stlgym_env_config)
            test_env_fn = partial(stlgym.make, stl_env_config_eval)
            log_dest = log_directory + "stlgym/rand_seed_" + str(random_seeds[i])
            logger_kwargs = dict(output_dir=log_dest, exp_name=exp3_name)
            print(f"Training PPO stlgym, random seed: {random_seeds[i]}...")
            ppo(env_fn, test_env_fn=test_env_fn, ac_kwargs=ac_kwargs, seed=random_seeds[i], 
                steps_per_epoch=steps_per_epoch, epochs=epochs, gamma=gamma, clip_ratio=clip_ratio, pi_lr=pi_lr,
                vf_lr=vf_lr, train_pi_iters=train_pi_iters, train_v_iters=train_v_iters, lam=lam, num_test_episodes=num_test_episodes, 
                max_ep_len=max_ep_len, target_kl=target_kl, logger_kwargs=logger_kwargs, save_freq=save_freq)

            # After the policy is trained, evaluate it for a given number of steps
            env, get_action = load_policy_and_env(fpath=log_dest, itr='last', deterministic=True)
            original_env = gym.make('CartPole-v0')
            stl_env = stlgym.make(stl_env_config_eval)
            evaluate_policy_in_2_environments(env1=original_env, env2=stl_env, get_action=get_action, log_dest=log_dest, max_ep_len=200, num_episodes=num_evals)
    

    if args['plot_all']:
        log_dirs = []
        for i in range(len(plot_legend)):
            log_dirs.append(log_directory + plot_legend[i])
        save_name = fig_directory + "sample_complexity_stl.png"
        make_plots(log_dirs, legend=plot_legend, xaxis='TotalEnvInteracts', values=['AverageTestEpRet'],
                #    ylim=(0, 1100), 
                   count=False, smooth=1, select=None, exclude=None, estimator='mean', save_name=save_name)

        save_name = fig_directory + "episode_length_stl.png"
        make_plots(log_dirs, legend=plot_legend, xaxis='TotalEnvInteracts', values=['TestEpLen'],
                #    ylim=(0, 240), 
                   count=False, smooth=1, select=None, exclude=None, estimator='mean', save_name=save_name)
    
    if args['plot_traces']:
        # Ensure the figure directory exists
        if not os.path.exists(fig_directory):
            os.mkdir(fig_directory)
        
        # Generate the trace figures for all random seeds
        for i in random_seeds:
            log_dest_single = log_directory + "single/rand_seed_" + str(i)
            log_dest_split = log_directory + "split/rand_seed_" + str(i)
            log_dest_stlgym = log_directory + "stlgym/rand_seed_" + str(i)
            save_name = fig_directory + "split_spec_cartpole_rand_seed_" + str(i) + "_traces.png"
            _, get_action1 = load_policy_and_env(fpath=log_dest_single, itr='last', deterministic=True)
            _, get_action2 = load_policy_and_env(fpath=log_dest_split, itr='last', deterministic=True)
            _, get_action3 = load_policy_and_env(fpath=log_dest_stlgym, itr='last', deterministic=True)
            trajectories1, trajectories2, trajectories3 = do_rollouts(10, get_action1, get_action2, get_action3)
            plot_trajectories(trajectories1, trajectories2, trajectories3, save_name)

    if args['table']:
        log_dirs = []
        for i in range(len(plot_legend)):
            log_dirs.append(log_directory + plot_legend[i])
        make_table(log_dirs, legend=plot_legend, separate=args['show_seeds'], select=None, exclude=None)

