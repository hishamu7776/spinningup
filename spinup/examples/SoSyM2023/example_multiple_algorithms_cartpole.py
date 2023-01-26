"""
file: example_multiple_algorithms_cartpole.py
author: Nathaniel Hamilton
email: nathaniel_hamilton@outlook.com

description:
    TODO

"""
# general libraries
import gym
from matplotlib.patches import Rectangle
import torch
import tensorflow as tf
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
from spinup import sac_pytorch as sac
from spinup import td3_pytorch as td3
from spinup import vpg_pytorch as vpg
from spinup import trpo_tf1 as trpo
from spinup.utils.test_policy import *
from spinup.utils.plot import *
from spinup.utils.tables import *

# environment libraries
import stlgym
from environments import *


# Function for plotting evaluation traces
def do_rollouts(num_rollouts, policy1, policy2):
    """
    TODO
    """
    env = gym.make('CartPole-v1')
    trajectories1 = []
    trajectories2 = []
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

    return trajectories1, trajectories2

def plot_trajectories(trajectories1, trajectories2, save_name):
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
        axis[0].plot(x1, np.linspace(0, len(x1)-1, len(x1)), 'b-')
        axis[1].plot(theta1, np.linspace(0, len(theta1)-1, len(theta1)), 'b-')
        x2, _, theta2, _ = zip(*trajectories2[i])
        axis[0].plot(x2, np.linspace(0, len(x2)-1, len(x2)), 'g-')
        axis[1].plot(theta2, np.linspace(0, len(theta2)-1, len(theta2)), 'g-')
    x1, _, theta1, _ = zip(*trajectories1[-1])
    axis[0].plot(x1, np.linspace(0, len(x1)-1, len(x1)), 'b-', label='ppo')
    axis[1].plot(theta1, np.linspace(0, len(theta1)-1, len(theta1)), 'b-', label='ppo')
    x2, _, theta2, _ = zip(*trajectories2[-1])
    axis[0].plot(x2, np.linspace(0, len(x2)-1, len(x2)), 'g-', label='trpo')
    axis[1].plot(theta2, np.linspace(0, len(theta2)-1, len(theta2)), 'g-', label='trpo')
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
    parser.add_argument('--ppo', help="TODO", action="store_true")
    parser.add_argument('--trpo', help="TODO", action="store_true")
    parser.add_argument('--sac',
                        help="TODO",
                        action="store_true")
    parser.add_argument('--td3',
                        help="TODO",
                        action="store_true")
    parser.add_argument('--vpg',
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
            if "SoSyM2023" in cwd:
                log_directory = "./logs/ex_multiple_algorithms_cartpole/"
                fig_directory = "./logs/figures/ex_multiple_algorithms_cartpole/"
            else:
                log_directory = "./SoSyM2023/logs/ex_multiple_algorithms_cartpole/"
                fig_directory = "./SoSyM2023/logs/figures/ex_multiple_algorithms_cartpole/"
        else:
            log_directory = "./examples/SoSyM2023/logs/ex_multiple_algorithms_cartpole/"
            fig_directory = "./examples/SoSyM2023/logs/figures/ex_multiple_algorithms_cartpole/"
    else:
        log_directory = "/tmp/logs/ppo/cartpole/"
        fig_directory = "/tmp/logs/ppo/cartpole/figures/ex_multiple_algorithms_cartpole/"
    # Reusing the specifications written for the previous cartpole example
    stl_env_config = args['config_path'] + "ex_split_spec_cartpole_single.yaml"
    stl_env_config_eval = args['config_path'] + "ex_split_spec_cartpole_eval.yaml"

    # Shared hyperparameters
    random_seeds = [1630, 2241, 2320, 2990, 3281, 4930, 5640, 8005, 9348, 9462]
    ac_kwargs = dict(hidden_sizes=(64, 64,))
    steps_per_epoch = 4000
    epochs = 100 # SAC & TD3 learn in ~10epochs but PPO & VPG require ~100
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
    ppo_name = "ppo"
    trpo_name = "trpo"
    sac_name = "sac"
    td3_name = "td3"
    vpg_name = "vpg"
    plot_legend = [ppo_name, trpo_name]  # sac_name, td3_name, vpg_name]

    if args['train_all']:
        # Overwrite the default false values to train all the experiments
        args['ppo'] = True
        args['trpo'] = True
        # args['sac'] = True  # SAC does not work with discrete action spaces like are used in cartpole
        # args['td3'] = True  # TD3 does not work with discrete action spaces like are used in cartpole
        # args['vpg'] = True

    # ppo performance
    if args['ppo']:
        for i in range(len(random_seeds)):
            env_fn = partial(stlgym.make, stl_env_config)
            test_env_fn = partial(stlgym.make, stl_env_config_eval)
            log_dest = log_directory + "ppo/rand_seed_" + str(random_seeds[i])
            logger_kwargs = dict(output_dir=log_dest, exp_name=ppo_name)
            print(f"Training PPO, random seed: {random_seeds[i]}...")
            ppo(env_fn, test_env_fn=test_env_fn, ac_kwargs=ac_kwargs, seed=random_seeds[i], 
                steps_per_epoch=steps_per_epoch, epochs=epochs, gamma=gamma, clip_ratio=clip_ratio, pi_lr=pi_lr,
                vf_lr=vf_lr, train_pi_iters=train_pi_iters, train_v_iters=train_v_iters, lam=lam, 
                num_test_episodes=num_test_episodes, 
                max_ep_len=max_ep_len, target_kl=target_kl, logger_kwargs=logger_kwargs, save_freq=save_freq)
    
    # trpo performance
    if args['trpo']:
        for i in range(len(random_seeds)):
            env_fn = partial(stlgym.make, stl_env_config)
            test_env_fn = partial(stlgym.make, stl_env_config_eval)
            log_dest = log_directory + "trpo/rand_seed_" + str(random_seeds[i])
            logger_kwargs = dict(output_dir=log_dest, exp_name=trpo_name)
            print(f"Training TRPO, random seed: {random_seeds[i]}...")
            trpo(env_fn, test_env_fn=test_env_fn, ac_kwargs=ac_kwargs,  seed=random_seeds[i], 
                steps_per_epoch=steps_per_epoch, epochs=epochs, gamma=gamma, delta=0.01, vf_lr=1e-3,
                train_v_iters=80, damping_coeff=0.1, cg_iters=10, backtrack_iters=10, 
                backtrack_coeff=0.8, lam=0.97, num_test_episodes=num_test_episodes, 
                max_ep_len=max_ep_len, logger_kwargs=logger_kwargs, save_freq=save_freq)
            tf.reset_default_graph()

    # sac performance
    if args['sac']:
        for i in range(len(random_seeds)):
            env_fn = partial(stlgym.make, stl_env_config)
            test_env_fn = partial(stlgym.make, stl_env_config_eval)
            log_dest = log_directory + "sac/rand_seed_" + str(random_seeds[i])
            logger_kwargs = dict(output_dir=log_dest, exp_name=sac_name)
            print(f"Training SAC, random seed: {random_seeds[i]}...")
            sac(env_fn, test_env_fn=test_env_fn, ac_kwargs=ac_kwargs, seed=random_seeds[i], 
                steps_per_epoch=steps_per_epoch, epochs=epochs, replay_size=int(1e6), gamma=gamma, 
                polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000, 
                update_after=1000, update_every=50, num_test_episodes=num_test_episodes, max_ep_len=max_ep_len, 
                logger_kwargs=logger_kwargs, save_freq=save_freq)
    
    # td3 performance
    if args['td3']:
        for i in range(len(random_seeds)):
            env_fn = partial(stlgym.make, stl_env_config)
            test_env_fn = partial(stlgym.make, stl_env_config_eval)
            log_dest = log_directory + "td3/rand_seed_" + str(random_seeds[i])
            logger_kwargs = dict(output_dir=log_dest, exp_name=td3_name)
            print(f"Training TD3, random seed: {random_seeds[i]}...")
            td3(env_fn, test_env_fn=test_env_fn, ac_kwargs=ac_kwargs, seed=random_seeds[i], 
                steps_per_epoch=steps_per_epoch, epochs=epochs, replay_size=int(1e6), gamma=gamma, 
                polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000, 
                update_after=1000, update_every=50, act_noise=0.1, target_noise=0.2, 
                noise_clip=0.5, policy_delay=2, num_test_episodes=num_test_episodes, max_ep_len=max_ep_len, 
                logger_kwargs=logger_kwargs, save_freq=save_freq)
    
    # # vpg performance
    # if args['trpo']:
    #     for i in range(len(random_seeds)):
    #         env_fn = partial(stlgym.make, stl_env_config)
    #         test_env_fn = partial(stlgym.make, stl_env_config_eval)
    #         log_dest = log_directory + "vpg/rand_seed_" + str(random_seeds[i])
    #         logger_kwargs = dict(output_dir=log_dest, exp_name=vpg_name)
    #         print(f"Training VPG, random seed: {random_seeds[i]}...")
    #         vpg(env_fn, test_env_fn=test_env_fn, ac_kwargs=ac_kwargs,  seed=random_seeds[i], 
    #             steps_per_epoch=steps_per_epoch, epochs=epochs, gamma=gamma, pi_lr=pi_lr,
    #             vf_lr=vf_lr, train_v_iters=train_v_iters, lam=0.97, num_test_episodes=num_test_episodes, 
    #             max_ep_len=max_ep_len,
    #             logger_kwargs=logger_kwargs, save_freq=save_freq)
    
    

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
            log_dest_ppo = log_directory + "ppo/rand_seed_" + str(i)
            log_dest_vpg = log_directory + "trpo/rand_seed_" + str(i)
            save_name = fig_directory + "cartpole_trace_rand_seed_" + str(i) + "_traces.png"
            _, get_action1 = load_policy_and_env(fpath=log_dest_ppo, itr='last', deterministic=True)
            _, get_action2 = load_policy_and_env(fpath=log_dest_vpg, itr='last', deterministic=True)
            env = gym.make('CartPole-v0')
            trajectories1, trajectories2 = do_rollouts(10, get_action1, get_action2)
            plot_trajectories(trajectories1, trajectories2, save_name)

    if args['table']:
        log_dirs = []
        for i in range(len(plot_legend)):
            log_dirs.append(log_directory + plot_legend[i])
        make_table(log_dirs, legend=plot_legend, separate=args['show_seeds'], select=None, exclude=None)

