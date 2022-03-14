"""
file: inverted_pendulum.py
author: Nathaniel Hamilton
email: nathaniel_hamilton@outlook.com

description:
    TODO: description
    test is OG reward
    alt_test is STL reward

"""
# general libraries
import gym
import torch
import os
from numpy.lib.function_base import append
from functools import partial

# spinup-specific
from spinup import ppo_pytorch as ppo
from spinup.utils.test_policy import *
from spinup.utils.plot import *
# from spinup.utils.tables import *

# environment libraries
import stlgym
from environments import *


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-all', help="Train all experimental cases.", action="store_true")
    parser.add_argument('--baseline', help="TODO", action="store_true")
    parser.add_argument('--stl',
                        help="TODO",
                        action="store_true")
    parser.add_argument('--config-path', help='Path to STLGym config file with specification.', type=str, default=None)
    parser.add_argument('--re-stl', help="TODO",
                        action="store_true")
    parser.add_argument('--load-path', help='TODO', type=str, default=None)
    parser.add_argument('--plot-all', help="Generate plots for all experimental cases.", action="store_true")
    parser.add_argument('--table', help="Generate text for latex table comparing final performance of all experimental cases.",
                        action="store_true")
    parser.add_argument('--show-seeds', help="Split the table results to show individual random seed results.",
                        action="store_true")
    parser.add_argument('--num-evals', help="Set the number of evaluations to occur after training. Default: 100",
                        type=int, default=100)
    args = vars(parser.parse_args())

    # Setting up the log directory and config file paths for STLGym environments TODO: finish adding steps here
    cwd = os.getcwd()
    if "spinup" in cwd:
        if "examples" in cwd:
            if "stl-gym" in cwd:
                log_directory = "./logs/ppo/pendulum/"
            else:
                log_directory = "./stl-gym/logs/ppo/pendulum/"
        else:
            log_directory = "./examples/stl-gym/logs/ppo/pendulum/"
    else:
        log_directory = "/tmp/logs/ppo/pendulum/"
    stl_env_config = args['config_path']

    # Hyperparameters
    random_seeds = [1630, 2241, 2320]  # , 2990, 3281, 4930, 5640, 8005, 9348, 9462]
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
    exp1_name = "baseline"
    exp2_name = "stl"
    exp3_name = "re-stl"
    plot_legend = [exp1_name, exp2_name, exp3_name]

    if args['train_all']:
        # Overwrite the default false values to train all the experiments
        args['baseline'] = True
        args['stl'] = True
        args['re_stl'] = True

    # Baseline performance
    if args['baseline']:
        for i in range(len(random_seeds)):
            env_fn = partial(gym.make, 'InvPendulum-v0')
            test_env_fn = partial(gym.make, 'InvPendulum-v0')
            alt_test_env_fn = partial(stlgym.make, stl_env_config)
            log_dest = log_directory + "baseline/rand_seed_" + str(random_seeds[i])
            logger_kwargs = dict(output_dir=log_dest, exp_name=exp1_name)
            print(f"Training PPO baseline, random seed: {random_seeds[i]}...")
            ppo(env_fn, test_env_fn=test_env_fn, alt_test_env_fn=alt_test_env_fn, ac_kwargs=ac_kwargs, seed=random_seeds[i], 
                steps_per_epoch=steps_per_epoch, epochs=epochs, gamma=gamma, clip_ratio=clip_ratio, pi_lr=pi_lr,
                vf_lr=vf_lr, train_pi_iters=train_pi_iters, train_v_iters=train_v_iters, lam=lam, num_test_episodes=num_test_episodes, 
                max_ep_len=max_ep_len, target_kl=target_kl, logger_kwargs=logger_kwargs, save_freq=save_freq)

            # After the policy is trained, evaluate it for a given number of steps
            env, get_action = load_policy_and_env(fpath=log_dest, itr='last', deterministic=True)
            original_env = gym.make('InvPendulum-v0')
            stl_env = stlgym.make(stl_env_config)
            evaluate_policy_in_2_environments(env1=original_env, env2=stl_env, get_action=get_action, log_dest=log_dest, max_ep_len=200, num_episodes=num_evals)

    # Training with STLGym reward function
    if args['stl']:
        for i in range(len(random_seeds)):
            env_fn = partial(stlgym.make, stl_env_config)
            test_env_fn = partial(gym.make, 'InvPendulum-v0')
            alt_test_env_fn = partial(stlgym.make, stl_env_config)
            log_dest = log_directory + "stl/rand_seed_" + str(random_seeds[i])
            logger_kwargs = dict(output_dir=log_dest, exp_name=exp1_name)
            print(f"Training PPO stl, random seed: {random_seeds[i]}...")
            ppo(env_fn, test_env_fn=test_env_fn, alt_test_env_fn=alt_test_env_fn, ac_kwargs=ac_kwargs, seed=random_seeds[i], 
                steps_per_epoch=steps_per_epoch, epochs=epochs, gamma=gamma, clip_ratio=clip_ratio, pi_lr=pi_lr,
                vf_lr=vf_lr, train_pi_iters=train_pi_iters, train_v_iters=train_v_iters, lam=lam, num_test_episodes=num_test_episodes, 
                max_ep_len=max_ep_len, target_kl=target_kl, logger_kwargs=logger_kwargs, save_freq=save_freq)

            # After the policy is trained, evaluate it for a given number of steps
            env, get_action = load_policy_and_env(fpath=log_dest, itr='last', deterministic=True)
            original_env = gym.make('InvPendulum-v0')
            stl_env = stlgym.make(stl_env_config)
            evaluate_policy_in_2_environments(env1=original_env, env2=stl_env, get_action=get_action, log_dest=log_dest, max_ep_len=200, num_episodes=num_evals)

    # Retraining with STLGym reward function
    if args['re_stl']:
        for i in range(len(random_seeds)):
            env_fn = partial(stlgym.make, stl_env_config)
            test_env_fn = partial(gym.make, 'InvPendulum-v0')
            alt_test_env_fn = partial(stlgym.make, stl_env_config)
            log_dest = log_directory + "re_stl/rand_seed_" + str(random_seeds[i])
            logger_kwargs = dict(output_dir=log_dest, exp_name=exp1_name)
            print(f"Training PPO re-stl, random seed: {random_seeds[i]}...")
            ppo(env_fn, test_env_fn=test_env_fn, alt_test_env_fn=alt_test_env_fn, ac_kwargs=ac_kwargs, seed=random_seeds[i], 
                steps_per_epoch=steps_per_epoch, epochs=epochs, gamma=gamma, clip_ratio=clip_ratio, pi_lr=pi_lr,
                vf_lr=vf_lr, train_pi_iters=train_pi_iters, train_v_iters=train_v_iters, lam=lam, num_test_episodes=num_test_episodes, 
                max_ep_len=max_ep_len, target_kl=target_kl, logger_kwargs=logger_kwargs, save_freq=save_freq)

            # After the policy is trained, evaluate it for a given number of steps
            env, get_action = load_policy_and_env(fpath=log_dest, itr='last', deterministic=True)
            original_env = gym.make('InvPendulum-v0')
            stl_env = stlgym.make(stl_env_config)
            evaluate_policy_in_2_environments(env1=original_env, env2=stl_env, get_action=get_action, log_dest=log_dest, max_ep_len=200, num_episodes=num_evals)
    
    if args['plot_all']:
        log_dirs = []
        for i in range(len(plot_legend)):
            log_dirs.append(log_directory + plot_legend[i])
        make_plots(log_dirs, legend=plot_legend, xaxis='TotalEnvInteracts', values=['AverageTestEpRet'],
                   ylim=(0, 1100), count=False, smooth=1, select=None, exclude=None, estimator='mean')

        make_plots(log_dirs, legend=plot_legend, xaxis='TotalEnvInteracts', values=['AverageAltTestEpRet'],
                   ylim=(0, 1100), count=False, smooth=1, select=None, exclude=None, estimator='mean')

        make_plots(log_dirs, legend=plot_legend, xaxis='TotalEnvInteracts', values=['TestEpLen'],
                   ylim=(0, 240), count=False, smooth=1, select=None, exclude=None, estimator='mean')

        make_plots(log_dirs, legend=plot_legend, xaxis='TotalEnvInteracts', values=['AltTestEpLen'],
                   ylim=(0, 240), count=False, smooth=1, select=None, exclude=None, estimator='mean')

    if args['table']:
        log_dirs = []
        for i in range(len(plot_legend)):
            log_dirs.append(log_directory + plot_legend[i])
        make_table(log_dirs, legend=plot_legend, separate=args['show_seeds'], select=None, exclude=None)

