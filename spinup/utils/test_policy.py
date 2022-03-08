import time
import joblib
import os
import os.path as osp
import tensorflow as tf
import torch
import pandas as pd
from spinup import EpochLogger
from spinup.utils.logx import restore_tf_graph


def load_policy_and_env(fpath, itr='last', deterministic=False):
    """
    Load a policy from save, whether it's TF or PyTorch, along with RL env.

    Not exceptionally future-proof, but it will suffice for basic uses of the 
    Spinning Up implementations.

    Checks to see if there's a tf1_save folder. If yes, assumes the model
    is tensorflow and loads it that way. Otherwise, loads as if there's a 
    PyTorch save.
    """

    # determine if tf save or pytorch save
    if any(['tf1_save' in x for x in os.listdir(fpath)]):
        backend = 'tf1'
    else:
        backend = 'pytorch'

    # handle which epoch to load from
    if itr=='last':
        # check filenames for epoch (AKA iteration) numbers, find maximum value

        if backend == 'tf1':
            saves = [int(x[8:]) for x in os.listdir(fpath) if 'tf1_save' in x and len(x)>8]

        elif backend == 'pytorch':
            pytsave_path = osp.join(fpath, 'pyt_save')
            # Each file in this folder has naming convention 'modelXX.pt', where
            # 'XX' is either an integer or empty string. Empty string case
            # corresponds to len(x)==8, hence that case is excluded.
            saves = [int(x.split('.')[0][5:]) for x in os.listdir(pytsave_path) if len(x)>8 and 'model' in x]

        itr = '%d'%max(saves) if len(saves) > 0 else ''

    else:
        assert isinstance(itr, int), \
            "Bad value provided for itr (needs to be int or 'last')."
        itr = '%d'%itr

    # load the get_action function
    if backend == 'tf1':
        get_action = load_tf_policy(fpath, itr, deterministic)
    else:
        get_action = load_pytorch_policy(fpath, itr, deterministic)

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:
        state = joblib.load(osp.join(fpath, 'vars'+itr+'.pkl'))
        env = state['env']
    except:
        env = None

    return env, get_action


def load_tf_policy(fpath, itr, deterministic=False):
    """ Load a tensorflow policy saved with Spinning Up Logger."""

    fname = osp.join(fpath, 'tf1_save'+itr)
    print('\n\nLoading from %s.\n\n'%fname)

    # load the things!
    sess = tf.Session()
    model = restore_tf_graph(sess, fname)

    # get the correct op for executing actions
    if deterministic and 'mu' in model.keys():
        # 'deterministic' is only a valid option for SAC policies
        print('Using deterministic action op.')
        action_op = model['mu']
    else:
        print('Using default action op.')
        action_op = model['pi']

    # make function for producing an action given a single state
    get_action = lambda x : sess.run(action_op, feed_dict={model['x']: x[None,:]})[0]

    return get_action


def load_pytorch_policy(fpath, itr, deterministic=False):
    """ Load a pytorch policy saved with Spinning Up Logger."""
    
    fname = osp.join(fpath, 'pyt_save', 'model'+itr+'.pt')
    print('\n\nLoading from %s.\n\n'%fname)

    model = torch.load(fname)

    # make function for producing an action given a single state
    def get_action(x):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            action = model.act(x)
        return action

    return get_action


def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True):

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    logger = EpochLogger()
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)

        a = get_action(o)
        o, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret, ep_len))
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()

def evaluate_policy_in_2_environments(env1, env2, get_action, log_dest='.', max_ep_len=None, num_episodes=100, render=False):
    """
    This function evaluates the policy in 2 different environments and records the results in comparative_results.csv.

    Args:
        env1 (gym.core.Env):    One of the Gym environments the policy will be evaluated in.
        env2 (gym.core.Env):    The other Gym environment the policy will be evaluated in.
        get_action (function):  Function representing the policy being evaluated. Create using the load_*_policy functions above.
        log_dest (str):         Path to folder where results will be saved.
        max_ep_len (int):       Maximum length for an evaluation episode. If None, the default for the environment will be used.
        num_episodes (int):     The number of times the evaluation will be run in each environment.
        render (bool):          Boolean value indicating whether or not the environment should be rendered during the evaluations. Default is False, i.e. do not render.
    Returns:
        None
    """

    assert env1 is not None, \
        "First environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."
    
    assert env2 is not None, \
        "Second environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    episodes = range(num_episodes)
    returns1 = []
    lengths1 = []
    dones1 = []
    returns2 = []
    lengths2 = []
    dones2 = []

    for ep in episodes:
        # Evaluate in environment 1
        o, r, done_int, ep_ret, ep_len = env1.reset(), 0, 0, 0, 0
        while ep_len < max_ep_len:
            a = get_action(o)
            o, r, d, env_info = env1.step(a)
            ep_ret += r
            ep_len += 1
            if d:
                done_int = 1
                break

        # Record the data
        print(f'Env1 episode {ep}: \t Return {ep_ret} \t Length {ep_len}  \t Done {d}')
        returns1.append(ep_ret)
        lengths1.append(ep_len)
        dones1.append(done_int)

        # Evaluate in environment 2
        o, r, done_int, ep_ret, ep_len = env2.reset(), 0, 0, 0, 0
        while ep_len < max_ep_len:
            a = get_action(o)
            o, r, d, env_info = env2.step(a)
            ep_ret += r
            ep_len += 1
            if d:
                done_int = 1
                break

        # Record the data
        print(f'Env2 episode {ep}: \t Return {ep_ret} \t Length {ep_len}  \t Done {d}')
        returns2.append(ep_ret)
        lengths2.append(ep_len)
        dones2.append(done_int)
    
    # Log the results in a .csv file header is ['index', 'Episode', 'EpRet1', 'EpLen1', 'Done1', 'EpRet2', 'EpLen2', 'Done2']
    dict = {'Episode': episodes, 'EpRet1': returns1, 'EpLen1': lengths1, 'Done1': dones1, 'EpRet2': returns2, 'EpLen2': lengths2, 'Done2': dones2}
    df = pd.DataFrame(dict, index=dict['Episode'])
    df.to_csv(log_dest + '/evaluation.csv')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    args = parser.parse_args()
    env, get_action = load_policy_and_env(args.fpath, 
                                          args.itr if args.itr >=0 else 'last',
                                          args.deterministic)
    run_policy(env, get_action, args.len, args.episodes, not(args.norender))