"""Plot rewards with error bar

    References:
    ----------
    low pass filter:
    https://stackoverflow.com/questions/47970683/vectorize-a-numpy-discount-calculation
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html
"""

__author__ = 'Guilherme Varela'
__date__ = '2020-03-05'
import argparse
import os
from os.path import dirname, basename
from pathlib import Path
import json
from glob import glob
from collections import defaultdict, OrderedDict

# third-party libs
import numpy as np
from numpy.random import choice
import scipy.stats as ss
from scipy.signal import lfilter

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')


FIGURE_X = 15.0
FIGURE_Y = 7.0

RED_COLOR = (0.886, 0.29, 0.20)

MEAN_CURVE_COLOR = (0.184,0.545,0.745)

GRAY_COLOR = (0.37,0.37,0.37)
GRAY_COLOR_2 = (0.43,0.43,0.43)


ROLLOUT_WARM_UP_PERIOD = 20

def get_arguments():
    parser = argparse.ArgumentParser(
        description="""
            This script processes an evaluation file which has possible
            multiple rollouts for a single experiment
        """
    )

    parser.add_argument('batch_path', type=str, nargs='?',
                         help='Path to an `eval.info.json` file in json format')

    parser.add_argument('--max_rollouts', '-r', dest='max_rollouts',
                        type=int, default=-1, nargs='?',
                        help='''Max. number of paths considered for each table
                                rollout. If -1 use as many as possible.''')

    return parser.parse_args()

def calculate_CI_bootstrap(x_hat, samples, num_resamples=20000):
    """
        Calculates 95 % interval using bootstrap.

        REF: https://ocw.mit.edu/courses/mathematics/
            18-05-introduction-to-probability-and-statistics-spring-2014/
            readings/MIT18_05S14_Reading24.pdf

    """
    resampled = np.random.choice(samples,
                                size=(len(samples), num_resamples),
                                replace=True)

    means = np.mean(resampled, axis=0)

    diffs = means - x_hat

    bounds = (x_hat - np.percentile(diffs, 5), x_hat - np.percentile(diffs, 95))

    return bounds

def main(batch_path=None):

    print('\nRUNNING analysis/rollouts.py\n')

    if not batch_path:
        args = get_arguments()
        batch_path = Path(args.batch_path)
        max_rollouts = args.max_rollouts
    else:
        batch_path = Path(batch_path)
        max_rollouts = -1


    suffix = '.l.eval.info.json' 
    if batch_path.is_file():
        file_path = batch_path
        batch_path = batch_path.parent
    else:
        pattern = f'*{suffix}'
        file_path = list(batch_path.glob(pattern))[0]

    # Prepare output folder.
    output_folder_path = os.path.join(batch_path, 'plots/rollouts')
    print('\tOutput folder: {0}'.format(output_folder_path))
    os.makedirs(output_folder_path, exist_ok=True)

    filename = file_path.name.replace(suffix, '')
    rewards = []
    with file_path.open('r') as f:
        db = json.load(f)

    cycle = db['cycle']
    discount = [1, -db['discount']]
    horizon = db['horizon']
    rollout_ids = db['rollouts']
    ex_ids = db['id']
    num_trials = len(ex_ids)
    num_cycles = int(horizon) / cycle
    num_rollouts = db['num_rollouts']
    if max_rollouts == -1:
        max_rollouts = num_rollouts

    returns = defaultdict(list)

    # Iterate for each experiment.
    for idx, ex_id in enumerate(ex_ids):
        roll_idxs = choice(num_rollouts, size=max_rollouts, replace=False)

        # Iterate for each Q table.
        for rid, rewards in db['rewards'][idx].items():

            rewards = np.concatenate(rewards, axis=1)

            # rewards shape: (cycles, num_rollouts)

            # Select subset.
            rewards = rewards[:, roll_idxs]

            # Remove samples from warm-up period.
            rewards = rewards[ROLLOUT_WARM_UP_PERIOD:,:]

            # Discount obtained rewards.
            rewards = np.flip(rewards, axis=0)
            gain = lfilter([1], discount, x=rewards, axis=0)

            # Concatenate.
            returns[int(rid)].append(gain[-1, :])

    returns = OrderedDict({
        k: returns[k] for k in sorted(returns.keys())
    })

    y = []
    CI_tstudent = []
    CI_bootstrap = []

    # This loop agreggates for each Q-table.
    for rid, ret in returns.items():
        ret = np.concatenate(ret)
        mean_ret = np.mean(ret)

        y.append(mean_ret)

        # Calculate 95% confidence interval (bootstrap).
        CI_bootstrap.append(calculate_CI_bootstrap(mean_ret, ret))

        # Calculate 95% confidence interval (t-student)
        CI_tstudent.append(ss.t.ppf(0.95, df=len(ret)-1) * (np.std(ret) / np.sqrt(len(ret))))

    # Extra processing for matplotlib.
    CI_bootstrap = np.array(CI_bootstrap).T
    CI_bootstrap = np.flip(CI_bootstrap, axis=0)
    error_bars_lengths = np.abs(np.subtract(CI_bootstrap,y))

    """
        Error bar plot.
    """
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)
            
    plt.plot(list(returns.keys()), y, label='Mean', c=MEAN_CURVE_COLOR)

    plt.errorbar(list(returns.keys()), y, yerr=error_bars_lengths,
                    c=MEAN_CURVE_COLOR, label='95% confidence interval', capsize=3)

    title = \
        f'Rollout num cycles: {num_cycles}, R: {max_rollouts}, T: {num_trials}'
    title = \
        f'{filename}\n({title})'
    plt.title(title)

    plt.xlabel(f'Train cycle')
    plt.ylabel('Discounted return')
    plt.xticks()

    plt.legend(loc=4)

    plt.savefig(f'{output_folder_path}/rollouts.png')
    plt.savefig(f'{output_folder_path}/rollouts.pdf')

    """
        Violin plot.
    """
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    data = []
    for _, ret in returns.items():
        data.append(np.concatenate(ret))

    violin_parts = plt.violinplot(data, positions=list(returns.keys()),
                                    showextrema=True, widths=150)

    plt.plot(list(returns.keys()), y, label='Mean', c=RED_COLOR)

    plt.errorbar(list(returns.keys()), y, yerr=error_bars_lengths,
                    label='95% confidence interval', capsize=3,
                    color=RED_COLOR)
    
    # Make all the violin statistics marks red:
    for partname in ('cbars','cmins','cmaxes'):
        vp = violin_parts[partname]
        vp.set_edgecolor(GRAY_COLOR_2)

    for pc in violin_parts['bodies']:
        pc.set_facecolor(GRAY_COLOR)
        pc.set_edgecolor(GRAY_COLOR)

    title = \
        f'Rollout num cycles: {num_cycles}, R: {max_rollouts}, T: {num_trials}'
    title = \
        f'{filename}\n({title})'
    plt.title(title)

    plt.xlabel(f'Train cycle')
    plt.ylabel('Discounted return')
    plt.xticks()

    plt.legend(loc=4)
    
    plt.savefig(f'{output_folder_path}/rollouts_violin_plot.png')
    plt.savefig(f'{output_folder_path}/rollouts_violin_plot.pdf')

if __name__ == '__main__':
    main()