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
from os.path import dirname, basename
import json
from glob import glob
from collections import defaultdict

# third-party libs
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import choice
import scipy.stats as ss
from scipy.signal import lfilter

def get_arguments():
    parser = argparse.ArgumentParser(
        description="""
            This script processes an evaluation file which has possible
            multiple rollouts for a single experiment
        """
    )

    parser.add_argument('batch_eval_path', type=str, nargs='?',
                         help='Path to an `eval.info.json` file in json format')

    parser.add_argument('--max_rollouts', '-r', dest='max_rollouts',
                        type=int, default=-1, nargs='?',
                        help='''Max. number of paths considered for each table
                                rollout. If -1 use as many as possible.''')

    return parser.parse_args()

if __name__ == '__main__':
        args = get_arguments()
        batch_eval_path = args.batch_eval_path
        max_rollouts = args.max_rollouts

        batch_eval_dir = dirname(batch_eval_path)
        filename = basename(batch_eval_path). \
            replace('.l.eval.info.json', '')
        rewards = []
        with open(batch_eval_path, 'r') as f:
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

        roll_returns = defaultdict(list)
        # consolidate experiments' rewards
        for idx, ex_id in enumerate(ex_ids):
            roll_idxs = choice(num_rollouts, size=max_rollouts, replace=False)

            # iterate for each experiment extracting the paths
            # while also discounting than
            for roll_id, rewards in db['rewards'][idx].items():
                # _rewards (cycles, num_rollouts)
                # select paths
                rewards = np.concatenate(rewards, axis=1)
                rewards = rewards[:, roll_idxs]
                gain = lfilter([1], discount, x=rewards, axis=0)
                roll_returns[int(roll_id)] += [gain[:, 0]]

        y = {}
        y_error = {}
        legends = []
        # This loop agreggates for each cycle # == roll_id
        # The resulting paths
        for roll_id, returns in roll_returns.items():
            returns = sorted(np.concatenate(returns))
            y[roll_id] = np.mean(returns)
            y_error[roll_id] = ss.t.ppf(0.95, len(returns)) * np.std(returns)
            legends.append(f'Q[{roll_id}]')

        # Must savefig.
        fig, ax = plt.subplots()
        plt.xlabel(f'Q-tables[train_cycles]')
        plt.ylabel('Reward')
        i = 0
        for roll_id, yy in y.items():
            plt.errorbar([str(roll_id)], yy, yerr=y_error[roll_id], fmt='-o')
            i += 1
        title = \
            f'cycles:{num_cycles},R:{max_rollouts}, T:{num_trials}'
        title = \
            f'{filename}\n(95% CI, {title})'
        ax.set_xticklabels(legends)
        plt.title(title)
        plt.savefig(f'{batch_eval_dir}/rollouts.png')
        plt.show()
