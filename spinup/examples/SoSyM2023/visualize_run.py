"""
file: visualize_run.py
author: Nathaniel Hamilton
email: nathaniel_hamilton@outlook.com

description:
    TODO: description

"""
# general libraries
import gym
from gym.wrappers import Monitor
import torch
import os
from numpy.lib.function_base import append
from functools import partial

# spinup-specific
from spinup import ppo_pytorch as ppo
from spinup.utils.test_policy import *
from spinup.utils.plot import *
from spinup.utils.tables import *

# environment libraries
import stlgym
from environments import *

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--load-path', help='TODO', type=str, default=None)
    parser.add_argument('--save-path', help='TODO', type=str, default=None)
    parser.add_argument('--num-evals', help="Set the number of evaluations to occur after training. Default: 1",
                        type=int, default=1)
    parser.add_argument('--max-len', help="Longest an evaluation can run for. Default: 200",
                        type=int, default=200)
    args = vars(parser.parse_args())

    load_path = args['load_path']
    max_ep_len = args['max_len']
    save_dest = args['save_path']

    env, get_action = load_policy_and_env(fpath=load_path, itr='last', deterministic=True)

    if save_dest is not None:
        env = Monitor(env, save_dest, force=True)
    theta = []
    reward = []

    for i in range(args['num_evals']):
        o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
        while n < max_ep_len:
            env.render()
            time.sleep(1e-2)

            a = get_action(o)
            o, r, d, info = env.step(a)
            # theta.append(round(info['theta'], 2))
            # reward.append(round(r, 2))
            ep_ret += r
            ep_len += 1

            if d or (ep_len == max_ep_len):
                print('Episode %d \t EpRet %.3f \t EpLen %d'%(i, ep_ret, ep_len))
                # print('Theta: ' + str(theta))
                # print('Reward: ' + str(reward))
                break
            n += 1