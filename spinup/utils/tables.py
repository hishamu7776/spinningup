"""
File: tables.py
Author: Nathaniel Hamilton

Description: A collection of functions, similar to those found in plot.py, for generating latex-format tables. These are
             designed specifically for making tables of the final, trained policy evaluations.
"""
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import os.path as osp
import numpy as np

DIV_LINE_WIDTH = 50

# Global vars for tracking and labeling data at load time.
exp_idx = 0
units = dict()


def get_table_datasets(logdir, condition=None, separate=False):
    """
    Recursively look through logdir for output files produced by
    spinup.logx.Logger.

    Assumes that any file "evaluation.csv" is a valid hit.
    """
    global exp_idx
    global units
    datasets = pd.DataFrame()
    table_data = pd.DataFrame()
    for root, _, files in os.walk(logdir):
        if 'evaluation.csv' in files:
            exp_name = None
            try:
                config_path = open(os.path.join(root, 'config.json'))
                config = json.load(config_path)
                if 'exp_name' in config:
                    exp_name = config['exp_name']
            except:
                print('No file named config.json')
            condition1 = condition or exp_name or 'exp'
            condition2 = condition1 + '-' + str(exp_idx)
            exp_idx += 1
            if condition1 not in units:
                units[condition1] = 0
            unit = units[condition1]
            units[condition1] += 1

            try:
                exp_data = pd.read_csv(os.path.join(root, 'evaluation.csv'), header=0,
                                       names=['index', 'Episode', 'EpRet1', 'EpLen1', 'Done1', 'EpRet2', 'EpLen2', 'Done2'])
            except:
                print('Could not read from %s' % os.path.join(root, 'evaluation.csv'))
                continue
            if separate:
                print('Not implemented yet')
                # cols = ['Configuration', 'RTAEpRet', 'RTAEplen', 'Interventions', 'EpRet', 'EpLen']
                # rta_on_ret = exp_data['RTAEpRet'].to_numpy()
                # st_rta_on_ret = '{:.4f}+-{:.4f}'.format(np.mean(rta_on_ret), np.std(rta_on_ret))
                # rta_on_len = exp_data['RTAEplen'].to_numpy()
                # st_rta_on_len = '{:.4f}+-{:.4f}'.format(np.mean(rta_on_len), np.std(rta_on_len))
                # interventions = exp_data['Interventions'].to_numpy()
                # st_interventions = '{:.4f}+-{:.4f}'.format(np.mean(interventions), np.std(interventions))
                # rta_off_ret = exp_data['EpRet'].to_numpy()
                # st_rta_off_ret = '{:.4f}+-{:.4f}'.format(np.mean(rta_off_ret), np.std(rta_off_ret))
                # rta_off_len = exp_data['EpLen'].to_numpy()
                # st_rta_off_len = '{:.4f}+-{:.4f}'.format(np.mean(rta_off_len), np.std(rta_off_len))
                # extracted_data = np.array([condition2, st_rta_on_ret, st_rta_on_len, st_interventions, st_rta_off_ret,
                #                            st_rta_off_len])
                # tempdata = pd.DataFrame([extracted_data.T], columns=cols)
                # table_data = pd.concat([table_data, tempdata], ignore_index=True)
            else:
                datasets = pd.concat([datasets, exp_data], ignore_index=True)

    if not separate:
        cols = ['Return 1', 'Length 1', 'Done 1', 'Return 2', 'Length 2', 'Done 2']
        env1_ret = datasets['EpRet1'].to_numpy()
        st_env1_ret = '{:.4f}+-{:.4f}'.format(np.mean(env1_ret), np.std(env1_ret))
        env1_len = datasets['EpLen1'].to_numpy()
        st_env1_len = '{:.4f}+-{:.4f}'.format(np.mean(env1_len), np.std(env1_len))
        env1_done = datasets['Done1'].to_numpy()
        st_env1_done = '{:.4f}+-{:.4f}'.format(np.mean(env1_done), np.std(env1_done))
        env2_ret = datasets['EpRet1'].to_numpy()
        st_env2_ret = '{:.4f}+-{:.4f}'.format(np.mean(env2_ret), np.std(env2_ret))
        env2_len = datasets['EpLen1'].to_numpy()
        st_env2_len = '{:.4f}+-{:.4f}'.format(np.mean(env2_len), np.std(env2_len))
        env2_done = datasets['Done1'].to_numpy()
        st_env2_done = '{:.4f}+-{:.4f}'.format(np.mean(env2_done), np.std(env2_done))
        extracted_data = np.array([st_env1_ret, st_env1_len, st_env1_done, st_env2_ret, st_env2_len, st_env2_done])
        table_data = pd.DataFrame([extracted_data.T], columns=cols)

    return table_data

def get_all_table_datasets(all_logdirs, legend=None, separate=False, select=None, exclude=None):
    """
    For every entry in all_logdirs,
        1) check if the entry is a real directory and if it is,
           pull data from it;

        2) if not, check to see if the entry is a prefix for a
           real directory, and pull data from that.
    """
    logdirs = []
    for logdir in all_logdirs:
        if osp.isdir(logdir) and logdir[-1] == os.sep:
            logdirs += [logdir]
        else:
            basedir = osp.dirname(logdir)
            fulldir = lambda x: osp.join(basedir, x)
            prefix = logdir.split(os.sep)[-1]
            listdir = os.listdir(basedir)
            logdirs += sorted([fulldir(x) for x in listdir if prefix in x])

    """
    Enforce selection rules, which check logdirs for certain substrings.
    Makes it easier to look at graphs from particular ablations, if you
    launch many jobs at once with similar names.
    """
    if select is not None:
        logdirs = [log for log in logdirs if all(x in log for x in select)]
    if exclude is not None:
        logdirs = [log for log in logdirs if all(not (x in log) for x in exclude)]

    # Verify logdirs
    print('Pulling from...\n' + '=' * DIV_LINE_WIDTH + '\n')
    for logdir in logdirs:
        print(logdir)
    print('\n' + '=' * DIV_LINE_WIDTH)

    # Make sure the legend is compatible with the logdirs
    assert not (legend) or (len(legend) == len(logdirs)), \
        "Must give a legend title for each set of experiments."

    # Load data from logdirs
    data = pd.DataFrame()
    if legend:
        for log, leg in zip(logdirs, legend):
            data = pd.concat([data, get_table_datasets(log, leg, separate=separate)], ignore_index=True)
    else:
        for log in logdirs:
            data = pd.concat([data, get_table_datasets(log, separate=separate)], ignore_index=True)
    return data


def make_table(all_logdirs, legend=None, separate=False, select=None, exclude=None):
    data = get_all_table_datasets(all_logdirs, legend, separate, select, exclude)
    print(data.to_latex(index=False))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', nargs='*')
    parser.add_argument('--legend', '-l', nargs='*')
    parser.add_argument('--xaxis', '-x', default='TotalEnvInteracts')
    parser.add_argument('--value', '-y', default='Performance', nargs='*')
    parser.add_argument('--count', action='store_true')
    parser.add_argument('--smooth', '-s', type=int, default=1)
    parser.add_argument('--select', nargs='*')
    parser.add_argument('--exclude', nargs='*')
    parser.add_argument('--est', default='mean')
    args = parser.parse_args()
    """

    Args: 
        logdir (strings): As many log directories (or prefixes to log 
            directories, which the plotter will autocomplete internally) as 
            you'd like to plot from.

        legend (strings): Optional way to specify legend for the plot. The 
            plotter legend will automatically use the ``exp_name`` from the
            config.json file, unless you tell it otherwise through this flag.
            This only works if you provide a name for each directory that
            will get plotted. (Note: this may not be the same as the number
            of logdir args you provide! Recall that the plotter looks for
            autocompletes of the logdir args: there may be more than one 
            match for a given logdir prefix, and you will need to provide a 
            legend string for each one of those matches---unless you have 
            removed some of them as candidates via selection or exclusion 
            rules (below).)

        xaxis (string): Pick what column from data is used for the x-axis.
             Defaults to ``TotalEnvInteracts``.

        value (strings): Pick what columns from data to graph on the y-axis. 
            Submitting multiple values will produce multiple graphs. Defaults
            to ``Performance``, which is not an actual output of any algorithm.
            Instead, ``Performance`` refers to either ``AverageEpRet``, the 
            correct performance measure for the on-policy algorithms, or
            ``AverageTestEpRet``, the correct performance measure for the 
            off-policy algorithms. The plotter will automatically figure out 
            which of ``AverageEpRet`` or ``AverageTestEpRet`` to report for 
            each separate logdir.

        count: Optional flag. By default, the plotter shows y-values which
            are averaged across all results that share an ``exp_name``, 
            which is typically a set of identical experiments that only vary
            in random seed. But if you'd like to see all of those curves 
            separately, use the ``--count`` flag.

        smooth (int): Smooth data by averaging it over a fixed window. This 
            parameter says how wide the averaging window will be.

        select (strings): Optional selection rule: the plotter will only show
            curves from logdirs that contain all of these substrings.

        exclude (strings): Optional exclusion rule: plotter will only show 
            curves from logdirs that do not contain these substrings.

    """

    # make_plots(args.logdir, args.legend, args.xaxis, args.value, args.count,
    #            smooth=args.smooth, select=args.select, exclude=args.exclude,
    #            estimator=args.est)


if __name__ == "__main__":
    main()
