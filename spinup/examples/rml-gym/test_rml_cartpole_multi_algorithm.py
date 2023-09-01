"""
file: example_multiple_algorithms_cartpole.py
author: Hisham Unniyankal
email: uaj.hisham@gmail.com

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
import tensorflow as tf
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
import rmlgym
from environments import *

from plot_evaluation import *

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-all', help="Train all experimental cases.", action="store_true")
    parser.add_argument('--ppo', help="TODO", action="store_true")
    parser.add_argument('--sac', help="TODO", action="store_true")
    parser.add_argument('--td3', help="TODO", action="store_true")
    parser.add_argument('--vpg', help="TODO", action="store_true")
    parser.add_argument('--trpo', help="TODO", action="store_true")
    parser.add_argument('--config-path', help='Path to STLGym config folder with specification.', type=str, default="./configs/")
    parser.add_argument('--plot-all', help="Generate plots for all experimental cases.", action="store_true")
    parser.add_argument('--plot-traces', help="TODO", action="store_true")
    parser.add_argument('--table', help="Generate text for latex table comparing final performance of all experimental cases.", action="store_true")
    parser.add_argument('--random-seed', help='Provide seed for algorithm.', type=int, default=0)
    parser.add_argument('--show-seeds', help="Split the table results to show individual random seed results.", action="store_true")
    parser.add_argument('--num-evals', help="Set the number of evaluations to occur after training. Default: 100", type=int, default=50)
    args = vars(parser.parse_args())

    # Setting up the log directory and config file paths for STLGym environments TODO: finish adding steps here
    cwd = os.getcwd()
    if "spinup" in cwd:
        if "examples" in cwd:
            if "rml" in cwd:
                log_directory = "./logs/test_rml_multiple_algorithms_cartpole/"
                fig_directory = "./logs/figures/test_rml_multiple_algorithms_cartpole/"
            else:
                log_directory = "./rml/logs/test_rml_multiple_algorithms_cartpole/"
                fig_directory = "./rml/logs/figures/test_rml_multiple_algorithms_cartpole/"
        else:
            log_directory = "./examples/rml/logs/test_rml_multiple_algorithms_cartpole/"
            fig_directory = "./examples/rml/logs/figures/test_rml_multiple_algorithms_cartpole/"
    else:
        log_directory = "/tmp/logs/ppo/cartpole/"
        fig_directory = "/tmp/logs/ppo/cartpole/figures/test_rml_multiple_algorithms_cartpole/"
    
    # Reusing the specifications written for the previous pendulum example
    rml_env_config = args['config_path'] + "rml_cartpole_stability.yaml"
    rml_env_config_eval = args['config_path'] + "rml_cartpole_stability_eval.yaml"

    # Shared hyperparameters
    random_seeds = [1630, 2241, 2320, 3281, 5640, 9348]#, 2990, 3281, 4930, 5640, 8005, 9348, 9462]
    if args['random_seed'] != 0:
        random_seeds = [args['random_seed']]

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
    #sac_name = "sac"
    #td3_name = "td3"
    #vpg_name = "vpg"
    plot_legend = [ppo_name, trpo_name]#sac_name, td3_name, vpg_name]

    if args['train_all']:
        # Overwrite the default false values to train all the experiments
        args['ppo'] = True
        args['trpo'] = True
        #args['sac'] = True
        #args['td3'] = True
        #args['vpg'] = True

    # ppo performance
    if args['ppo']:
        for i in range(len(random_seeds)):
            env_fn = partial(rmlgym.make, rml_env_config)
            test_env_fn = partial(rmlgym.make, rml_env_config_eval)
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
            env_fn = partial(rmlgym.make, rml_env_config)
            test_env_fn = partial(rmlgym.make, rml_env_config_eval)
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
            env_fn = partial(rmlgym.make, rml_env_config)
            test_env_fn = partial(rmlgym.make, rml_env_config_eval)
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
            env_fn = partial(rmlgym.make, rml_env_config)
            test_env_fn = partial(rmlgym.make, rml_env_config_eval)
            log_dest = log_directory + "td3/rand_seed_" + str(random_seeds[i])
            logger_kwargs = dict(output_dir=log_dest, exp_name=td3_name)
            print(f"Training TD3, random seed: {random_seeds[i]}...")
            td3(env_fn, test_env_fn=test_env_fn, ac_kwargs=ac_kwargs, seed=random_seeds[i], 
                steps_per_epoch=steps_per_epoch, epochs=epochs, replay_size=int(1e6), gamma=gamma, 
                polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000, 
                update_after=1000, update_every=50, act_noise=0.1, target_noise=0.2, 
                noise_clip=0.5, policy_delay=2, num_test_episodes=num_test_episodes, max_ep_len=max_ep_len, 
                logger_kwargs=logger_kwargs, save_freq=save_freq)
    
    # vpg performance
    if args['vpg']:
        for i in range(len(random_seeds)):
            env_fn = partial(rmlgym.make, rml_env_config)
            test_env_fn = partial(rmlgym.make, rml_env_config_eval)
            log_dest = log_directory + "vpg/rand_seed_" + str(random_seeds[i])
            logger_kwargs = dict(output_dir=log_dest, exp_name=vpg_name)
            print(f"Training VPG, random seed: {random_seeds[i]}...")
            vpg(env_fn, test_env_fn=test_env_fn, ac_kwargs=ac_kwargs,  seed=random_seeds[i], 
                steps_per_epoch=steps_per_epoch, epochs=epochs, gamma=gamma, pi_lr=pi_lr,
                vf_lr=vf_lr, train_v_iters=train_v_iters, lam=0.97, num_test_episodes=num_test_episodes, 
                max_ep_len=max_ep_len,
                logger_kwargs=logger_kwargs, save_freq=save_freq)

    if args['plot_all']:
        log_dirs = []
        for i in range(len(plot_legend)):
            log_dirs.append(log_directory + plot_legend[i])
        save_name = fig_directory + "sample_complexity_rml.png"
        make_plots(log_dirs, legend=plot_legend, xaxis='TotalEnvInteracts', values=['AverageTestEpRet'],
                #    ylim=(0, 1100), 
                   count=False, smooth=1, select=None, exclude=None, estimator='mean', save_name=save_name)

        save_name = fig_directory + "episode_length_rml.png"
        make_plots(log_dirs, legend=plot_legend, xaxis='TotalEnvInteracts', values=['TestEpLen'],
                #    ylim=(0, 240), 
                   count=False, smooth=1, select=None, exclude=None, estimator='mean', save_name=save_name)
    
    if args['plot_traces']:
        # Ensure the figure directory exists
        if not os.path.exists(fig_directory):
            os.mkdir(fig_directory)
        
        # Generate the trace figures for all random seeds
        for i in random_seeds:
            log_dest_baseline = log_directory + "ppo/rand_seed_" + str(i)
            log_dest_sparse = log_directory + "sac/rand_seed_" + str(i)
            log_dest_dense = log_directory + "td3/rand_seed_" + str(i)
            log_dest_rml = log_directory + "vpg/rand_seed_" + str(i)
            save_name = fig_directory + "rml_ppo_rand_seed_" + str(i) + "_traces.png"
            _, get_action1 = load_policy_and_env(fpath=log_dest_baseline, itr='last', deterministic=True)
            _, get_action2 = load_policy_and_env(fpath=log_dest_sparse, itr='last', deterministic=True)
            _, get_action3 = load_policy_and_env(fpath=log_dest_dense, itr='last', deterministic=True)
            _, get_action4 = load_policy_and_env(fpath=log_dest_rml, itr='last', deterministic=True)
            env = gym.make('CartPole-v0')
            trajectories1, trajectories2, trajectories3,trajectories4 = do_rollout_cartpole(num_evals, get_action1, get_action2, get_action3,get_action4)
            plot_trajectories_cartpole(trajectories1, trajectories2, trajectories3,trajectories4, save_name)

    if args['table']:
        log_dirs = []
        for i in range(len(plot_legend)):
            log_dirs.append(log_directory + plot_legend[i])
        make_table(log_dirs, legend=plot_legend, separate=args['show_seeds'], select=None, exclude=None)

