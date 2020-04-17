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
from pathlib import Path
import json
from glob import glob
from collections import defaultdict, OrderedDict

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

    parser.add_argument('batch_path', type=str, nargs='?',
                         help='Path to an `eval.info.json` file in json format')

    parser.add_argument('--max_rollouts', '-r', dest='max_rollouts',
                        type=int, default=-1, nargs='?',
                        help='''Max. number of paths considered for each table
                                rollout. If -1 use as many as possible.''')

    return parser.parse_args()

if __name__ == '__main__':
        args = get_arguments()
        batch_path = Path(args.batch_path)
        max_rollouts = args.max_rollouts

        suffix = '.l.eval.info.json' 
        if batch_path.is_file():
            file_path = batch_path
            batch_path = batch_path.parent
        else:
            pattern = f'*{suffix}'
            file_path = list(batch_path.glob(pattern))[0]

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
        # consolidate experiments' rewards
        for idx, ex_id in enumerate(ex_ids):
            roll_idxs = choice(num_rollouts, size=max_rollouts, replace=False)

            # iterate for each experiment extracting the paths
            # while also discounting than
            for rid, rewards in db['rewards'][idx].items():
                # _rewards (cycles, num_rollouts)
                # select paths
                rewards = np.concatenate(rewards, axis=1)
                rewards = rewards[:, roll_idxs]
                gain = lfilter([1], discount, x=rewards, axis=0)
                returns[int(rid)] += [gain[:, 0]]

        returns = OrderedDict({
            k: returns[k] for k in sorted(returns.keys())
        })
        y = {}
        y_error = {}
        legends = []
        figure_path = batch_path / 'rollouts.png'
        # This loop agreggates for each cycle # == rid
        # The resulting paths
        for rid, ret in returns.items():
            ret = sorted(np.concatenate(ret))
            y[rid] = np.mean(ret)
            y_error[rid] = ss.t.ppf(0.95, len(ret)) * np.std(ret)
            legends.append(f'Q[{rid}]')

        # Must savefig.
        fig, ax = plt.subplots()
        plt.xlabel(f'Q-tables[train_cycles]')
        plt.ylabel('Reward')
        i = 0
        for rid, yy in y.items():
            plt.errorbar([str(rid)], yy, yerr=y_error[rid], fmt='-o')
            i += 1
        title = \
            f'cycles:{num_cycles},R:{max_rollouts}, T:{num_trials}'
        title = \
            f'{filename}\n(95% CI, {title})'
        ax.set_xticklabels(legends)
        plt.title(title)
        plt.savefig(figure_path.as_posix())
        plt.show()
