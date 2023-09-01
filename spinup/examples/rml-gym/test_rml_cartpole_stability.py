"""
file: example_retrain.py
author: Hisham Unniyankal
email: uaj.hisham@gmail.com

description:
    This example demonstrates how RMLGym can be used to train an agent to satisfy a more specific goal. 
    This example uses the pendulum environment. The baseline reward function is very effective at training an agent to keep the pole up.
    However, many different solutions can be learned from tis vague reward function. 
    Not all learned solutions are stable and some cannot last longer than the 200 step minimum made default.
    With RMLGym, we can quickly retrain these solutions to learn specified behavior.

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
import rmlgym
#from rmlgym import RMLGym
from environments import *

from plot_evaluation import *

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-all', help="Train all experimental cases.", action="store_true")
    parser.add_argument('--baseline', help="TODO", action="store_true")
    parser.add_argument('--sparse', help="TODO", action="store_true")
    parser.add_argument('--dense', help="TODO", action="store_true")
    parser.add_argument('--rml', help="TODO", action="store_true")
    parser.add_argument('--config-path', help='Path to config folder with specification.', type=str, default="./configs/")
    parser.add_argument('--random-seed', help='Provide seed for algorithm.', type=int, default=0)
    parser.add_argument('--plot-all', help="Generate plots for all experimental cases.", action="store_true")
    parser.add_argument('--plot-traces', help="TODO", action="store_true")
    parser.add_argument('--table', help="Generate text for latex table comparing final performance of all experimental cases.", action="store_true")
    parser.add_argument('--show-seeds', help="Split the table results to show individual random seed results.",action="store_true")
    parser.add_argument('--num-evals', help="Set the number of evaluations to occur after training. Default: 100", type=int, default=50)
    args = vars(parser.parse_args())

    # Setting up the log directory and config file paths for STLGym environments TODO: finish adding steps here
    cwd = os.getcwd()
    print(cwd)
    
    if "spinup" in cwd:
        if "examples" in cwd:
            if "rml-gym" in cwd:
                log_directory = "./logs/rml_cartpole_stability/"
                fig_directory = "./logs/figures/rml_cartpole_stability/"
            else:
                log_directory = "./rml-gym/logs/rml_cartpole_stability/"
                fig_directory = "./rml-gym/logs/figures/rml_cartpole_stability/"
        else:
            log_directory = "./examples/rml-gym/logs/rml_cartpole_stability/"
            fig_directory = "./examples/rml-gym/logs/figures/rml_cartpole_stability/"
    else:
        log_directory = "/tmp/logs/ppo/cartpole/"
        fig_directory = "/tmp/logs/ppo/cartpole/figures/rml_cartpole_stability/"

    stl_env_config_dense = args['config_path'] + "stl_dense_cartpole_stability.yaml"
    stl_env_config_sparse = args['config_path'] + "stl_sparse_cartpole_stability.yaml"
    rml_env_config = args['config_path'] + "rml_cartpole_stability.yaml"
    rml_env_config_eval = args['config_path'] + "rml_cartpole_stability_eval.yaml"
    
    # Hyperparameters
    random_seeds = [1630, 2241, 2320] # 1630, 2241, 2320, 2990, 3281, 4930, 5640, 8005, 9348, 9462]
    if args['random_seed'] != 0:
        random_seeds = [args['random_seed']]
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
    exp2_name = "sparse"
    exp3_name = "dense"
    exp4_name = "rml"

    plot_legend = [exp1_name, exp2_name, exp3_name, exp4_name]
    
    if args['train_all']:
        # Overwrite the default false values to train all the experiments
        args['baseline'] = True
        args['sparse'] = True
        args['dense'] = True
        args['rml'] = True


    # Baseline performance
    if args['baseline']:
        for i in range(len(random_seeds)):
            env_fn = partial(gym.make, 'CartPole-v0')
            test_env_fn = partial(gym.make, 'CartPole-v0')
            alt_test_env_fn = partial(rmlgym.make, rml_env_config)
            log_dest = log_directory + "baseline/rand_seed_" + str(random_seeds[i])
            logger_kwargs = dict(output_dir=log_dest, exp_name=exp1_name)
            print(f"Training PPO baseline, random seed: {random_seeds[i]}...")
            ppo(env_fn, test_env_fn=test_env_fn, alt_test_env_fn=alt_test_env_fn, ac_kwargs=ac_kwargs, seed=random_seeds[i], 
                steps_per_epoch=steps_per_epoch, epochs=epochs, gamma=gamma, clip_ratio=clip_ratio, pi_lr=pi_lr,
                vf_lr=vf_lr, train_pi_iters=train_pi_iters, train_v_iters=train_v_iters, lam=lam, num_test_episodes=num_test_episodes, 
                max_ep_len=max_ep_len, target_kl=target_kl, logger_kwargs=logger_kwargs, save_freq=save_freq)

            # After the policy is trained, evaluate it for a given number of steps
            env, get_action = load_policy_and_env(fpath=log_dest, itr='last', deterministic=True)
            original_env = gym.make('CartPole-v0')
            stl_env = stlgym.make(stl_env_config_dense)
            evaluate_policy_in_2_environments(env1=original_env, env2=stl_env, get_action=get_action, log_dest=log_dest, max_ep_len=200, num_episodes=num_evals)
    
    # Training with STLGym reward function sparsely-defined
    if args['sparse']:
        for i in range(len(random_seeds)):
            env_fn = partial(stlgym.make, stl_env_config_sparse)
            test_env_fn = partial(gym.make, 'CartPole-v0')
            alt_test_env_fn = partial(rmlgym.make, rml_env_config)
            log_dest = log_directory + "sparse/rand_seed_" + str(random_seeds[i])
            logger_kwargs = dict(output_dir=log_dest, exp_name=exp2_name)
            print(f"Training PPO sparse, random seed: {random_seeds[i]}...")
            ppo(env_fn, test_env_fn=test_env_fn, alt_test_env_fn=alt_test_env_fn, ac_kwargs=ac_kwargs, seed=random_seeds[i], 
                steps_per_epoch=steps_per_epoch, epochs=epochs, gamma=gamma, clip_ratio=clip_ratio, pi_lr=pi_lr,
                vf_lr=vf_lr, train_pi_iters=train_pi_iters, train_v_iters=train_v_iters, lam=lam, num_test_episodes=num_test_episodes, 
                max_ep_len=max_ep_len, target_kl=target_kl, logger_kwargs=logger_kwargs, save_freq=save_freq)

            # After the policy is trained, evaluate it for a given number of steps
            env, get_action = load_policy_and_env(fpath=log_dest, itr='last', deterministic=True)
            original_env = gym.make('CartPole-v0')
            stl_env = stlgym.make(stl_env_config_sparse)
            evaluate_policy_in_2_environments(env1=original_env, env2=stl_env, get_action=get_action, log_dest=log_dest, max_ep_len=200, num_episodes=num_evals)
    
    # Training with STLGym reward function densely-defined
    if args['dense']:
        for i in range(len(random_seeds)):
            env_fn = partial(stlgym.make, stl_env_config_dense)
            test_env_fn = partial(gym.make, 'CartPole-v0')
            alt_test_env_fn = partial(rmlgym.make, rml_env_config)
            log_dest = log_directory + "dense/rand_seed_" + str(random_seeds[i])
            logger_kwargs = dict(output_dir=log_dest, exp_name=exp3_name)
            print(f"Training PPO dense, random seed: {random_seeds[i]}...")
            ppo(env_fn, test_env_fn=test_env_fn, alt_test_env_fn=alt_test_env_fn, ac_kwargs=ac_kwargs, seed=random_seeds[i], 
                steps_per_epoch=steps_per_epoch, epochs=epochs, gamma=gamma, clip_ratio=clip_ratio, pi_lr=pi_lr,
                vf_lr=vf_lr, train_pi_iters=train_pi_iters, train_v_iters=train_v_iters, lam=lam, num_test_episodes=num_test_episodes, 
                max_ep_len=max_ep_len, target_kl=target_kl, logger_kwargs=logger_kwargs, save_freq=save_freq)

            # After the policy is trained, evaluate it for a given number of steps
            env, get_action = load_policy_and_env(fpath=log_dest, itr='last', deterministic=True)
            original_env = gym.make('CartPole-v0')
            stl_env = stlgym.make(stl_env_config_sparse)
            evaluate_policy_in_2_environments(env1=original_env, env2=stl_env, get_action=get_action, log_dest=log_dest, max_ep_len=200, num_episodes=num_evals)
    
    # Training with RMLGym reward function defined uning runtime monitoring
    if args['rml']:
        for i in range(len(random_seeds)):
            env_fn = partial(rmlgym.make, rml_env_config)
            test_env_fn = partial(gym.make, 'CartPole-v0')
            alt_test_env_fn = partial(rmlgym.make, rml_env_config_eval)
            log_dest = log_directory + "rml/rand_seed_" + str(random_seeds[i])
            logger_kwargs = dict(output_dir=log_dest, exp_name=exp4_name)
            print(f"Training PPO dense, random seed: {random_seeds[i]}...")
            
            ppo(env_fn, test_env_fn=test_env_fn, alt_test_env_fn=alt_test_env_fn, ac_kwargs=ac_kwargs, seed=random_seeds[i], 
                steps_per_epoch=steps_per_epoch, epochs=epochs, gamma=gamma, clip_ratio=clip_ratio, pi_lr=pi_lr,
                vf_lr=vf_lr, train_pi_iters=train_pi_iters, train_v_iters=train_v_iters, lam=lam, num_test_episodes=num_test_episodes, 
                max_ep_len=max_ep_len, target_kl=target_kl, logger_kwargs=logger_kwargs, save_freq=save_freq)
            
            # After the policy is trained, evaluate it for a given number of steps
            env, get_action = load_policy_and_env(fpath=log_dest, itr='last', deterministic=True)
            original_env = gym.make('CartPole-v0')
            stl_env = stlgym.make(stl_env_config_sparse)
            evaluate_policy_in_2_environments(env1=original_env, env2=stl_env, get_action=get_action, log_dest=log_dest, max_ep_len=200, num_episodes=num_evals)

    if args['plot_all']:
        log_dirs = []
        for i in range(len(plot_legend)):
            log_dirs.append(log_directory + plot_legend[i])
        save_name = fig_directory + "sample_complexity_baseline.png"
        make_plots(log_dirs, legend=plot_legend, xaxis='TotalEnvInteracts', values=['AverageTestEpRet'],
                #    ylim=(0, 1100), 
                   count=False, smooth=1, select=None, exclude=None, estimator='mean', save_name=save_name)

        save_name = fig_directory + "sample_complexity_rml.png"
        make_plots(log_dirs, legend=plot_legend, xaxis='TotalEnvInteracts', values=['AverageAltTestEpRet'],
                #    ylim=(0, 1100), 
                   count=False, smooth=1, select=None, exclude=None, estimator='mean', save_name=save_name)

        save_name = fig_directory + "episode_length_baseline.png"
        make_plots(log_dirs, legend=plot_legend, xaxis='TotalEnvInteracts', values=['TestEpLen'],
                #    ylim=(0, 240), 
                   count=False, smooth=1, select=None, exclude=None, estimator='mean', save_name=save_name)

        save_name = fig_directory + "episode_length_rml.png"
        make_plots(log_dirs, legend=plot_legend, xaxis='TotalEnvInteracts', values=['AltTestEpLen'],
                #    ylim=(0, 240), 
                   count=False, smooth=1, select=None, exclude=None, estimator='mean', save_name=save_name)
        
    
    if args['plot_traces']:
        # Ensure the figure directory exists
        if not os.path.exists(fig_directory):
            os.mkdir(fig_directory)
        
        # Generate the trace figures for all random seeds
        for i in random_seeds:
            log_dest_baseline = log_directory + "baseline/rand_seed_" + str(i)
            log_dest_sparse = log_directory + "sparse/rand_seed_" + str(i)
            log_dest_dense = log_directory + "dense/rand_seed_" + str(i)
            log_dest_rml = log_directory + "rml/rand_seed_" + str(i)
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

