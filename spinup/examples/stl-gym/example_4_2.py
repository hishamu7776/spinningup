"""
Multiple RL algorithms used to learn the same specs: docking
Also, conditional success
"""
# general libraries
import gym
import torch
import os
from numpy.lib.function_base import append
from functools import partial
import pandas as pd

# spinup-specific
from spinup import ppo_pytorch as ppo
from spinup import sac_pytorch as sac
from spinup import td3_pytorch as td3
from spinup import trpo_pytorch as trpo
from spinup.utils.test_policy import *
from spinup.utils.plot import *
from spinup.utils.tables import *

# environment libraries
import stlgym
from environments import *

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-all', help="Train all experimental cases.", action="store_true")
    parser.add_argument('--ppo', help="TODO", action="store_true")
    parser.add_argument('--sac',
                        help="TODO",
                        action="store_true")
    parser.add_argument('--td3',
                        help="TODO",
                        action="store_true")
    parser.add_argument('--trpo',
                        help="TODO",
                        action="store_true")
    parser.add_argument('--config-path', help='Path to STLGym config folder with specification.', type=str, default="./configs/")
    parser.add_argument('--plot-all', help="Generate plots for all experimental cases.", action="store_true")
    parser.add_argument('--plot-traces', help="TODO", action="store_true")
    parser.add_argument('--table', help="Generate text for latex table comparing final performance of all experimental cases.",
                        action="store_true")
    parser.add_argument('--show-seeds', help="Split the table results to show individual random seed results.",
                        action="store_true")
    args = vars(parser.parse_args())

    # Setting up the log directory and config file paths for STLGym environments TODO: finish adding steps here
    cwd = os.getcwd()
    if "spinup" in cwd:
        if "examples" in cwd:
            if "stl-gym" in cwd:
                log_directory = "./logs/ex_4_2/"
            else:
                log_directory = "./stl-gym/logs/ex_4_2/"
        else:
            log_directory = "./examples/stl-gym/logs/ex_4_2/"
    else:
        log_directory = "/tmp/logs/"
    stl_env_config = args['config_path'] + "ex_4_2.yaml"

    # Shared parameters
    random_seeds = [1630, 2241, 2320, 2990, 3281] # , 4930, 5640, 8005, 9348, 9462]
    epochs = 100
    num_test_episodes = 10
    max_ep_len = 200
    save_freq = 1

    # Experiment names for plotting
    ppo_name = "PPO"
    sac_name = "SAC"
    td3_name = "TD3"
    trpo_name = "TRPO"
    plot_legend = [ppo_name, sac_name, td3_name, trpo_name]

    if args['train_all']:
        # Overwrite the default false values to train all the experiments
        args['ppo'] = True
        args['sac'] = True
        args['td3'] = True
        args['trpo'] = True
    
    if args['ppo']:
        for i in range(len(random_seeds)):
            env_fn = partial(gym.make, 'CartPole-v1')
            test_env_fn = partial(gym.make, 'CartPole-v1')
            log_dest = log_directory + "ppo/rand_seed_" + str(random_seeds[i])
            logger_kwargs = dict(output_dir=log_dest, exp_name=ppo_name)
            print(f"Training PPO, random seed: {random_seeds[i]}...")
            ppo(env_fn, test_env_fn=test_env_fn, alt_test_env_fn=None, ac_kwargs=ac_kwargs, seed=random_seeds[i], 
                steps_per_epoch=steps_per_epoch, epochs=epochs, gamma=gamma, clip_ratio=clip_ratio, pi_lr=pi_lr,
                vf_lr=vf_lr, train_pi_iters=train_pi_iters, train_v_iters=train_v_iters, lam=lam, num_test_episodes=num_test_episodes, 
                max_ep_len=max_ep_len, target_kl=target_kl, logger_kwargs=logger_kwargs, save_freq=save_freq)
    
    if args['sac']:
        for i in range(len(random_seeds)):
            env_fn = partial(gym.make, 'CartPole-v1')
            test_env_fn = partial(gym.make, 'CartPole-v1')
            log_dest = log_directory + "sac/rand_seed_" + str(random_seeds[i])
            logger_kwargs = dict(output_dir=log_dest, exp_name=sac_name)
            print(f"Training SAC, random seed: {random_seeds[i]}...")
            sac(env_fn, test_env_fn=test_env_fn, alt_test_env_fn=None, ac_kwargs=ac_kwargs, seed=random_seeds[i], 
                steps_per_epoch=steps_per_epoch, epochs=epochs, gamma=gamma, clip_ratio=clip_ratio, pi_lr=pi_lr,
                vf_lr=vf_lr, train_pi_iters=train_pi_iters, train_v_iters=train_v_iters, lam=lam, num_test_episodes=num_test_episodes, 
                max_ep_len=max_ep_len, target_kl=target_kl, logger_kwargs=logger_kwargs, save_freq=save_freq)
    
    if args['td3']:
        for i in range(len(random_seeds)):
            env_fn = partial(gym.make, 'CartPole-v1')
            test_env_fn = partial(gym.make, 'CartPole-v1')
            log_dest = log_directory + "td3/rand_seed_" + str(random_seeds[i])
            logger_kwargs = dict(output_dir=log_dest, exp_name=td3_name)
            print(f"Training TD3, random seed: {random_seeds[i]}...")
            td3(env_fn, test_env_fn=test_env_fn, alt_test_env_fn=None, ac_kwargs=ac_kwargs, seed=random_seeds[i], 
                steps_per_epoch=steps_per_epoch, epochs=epochs, gamma=gamma, clip_ratio=clip_ratio, pi_lr=pi_lr,
                vf_lr=vf_lr, train_pi_iters=train_pi_iters, train_v_iters=train_v_iters, lam=lam, num_test_episodes=num_test_episodes, 
                max_ep_len=max_ep_len, target_kl=target_kl, logger_kwargs=logger_kwargs, save_freq=save_freq)
    
    if args['trpo']:
        for i in range(len(random_seeds)):
            env_fn = partial(gym.make, 'CartPole-v1')
            test_env_fn = partial(gym.make, 'CartPole-v1')
            log_dest = log_directory + "trpo/rand_seed_" + str(random_seeds[i])
            logger_kwargs = dict(output_dir=log_dest, exp_name=trpo_name)
            print(f"Training TRPO, random seed: {random_seeds[i]}...")
            trpo(env_fn, test_env_fn=test_env_fn, alt_test_env_fn=None, ac_kwargs=ac_kwargs, seed=random_seeds[i], 
                steps_per_epoch=steps_per_epoch, epochs=epochs, gamma=gamma, clip_ratio=clip_ratio, pi_lr=pi_lr,
                vf_lr=vf_lr, train_pi_iters=train_pi_iters, train_v_iters=train_v_iters, lam=lam, num_test_episodes=num_test_episodes, 
                max_ep_len=max_ep_len, target_kl=target_kl, logger_kwargs=logger_kwargs, save_freq=save_freq)

    if args['plot_all']:
        log_dirs = []
        for i in range(len(plot_legend)):
            log_dirs.append(log_directory + plot_legend[i])
        make_plots(log_dirs, legend=plot_legend, xaxis='TotalEnvInteracts', values=['AverageTestEpRet'],
                #    ylim=(0, 1100), 
                   count=False, smooth=1, select=None, exclude=None, estimator='mean')
        
        make_plots(log_dirs, legend=plot_legend, xaxis='TotalEnvInteracts', values=['AverageSuccess'],
                #    ylim=(0, 240), 
                   count=False, smooth=1, select=None, exclude=None, estimator='mean')

        make_plots(log_dirs, legend=plot_legend, xaxis='TotalEnvInteracts', values=['TestEpLen'],
                #    ylim=(0, 240), 
                   count=False, smooth=1, select=None, exclude=None, estimator='mean')