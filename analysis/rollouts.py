"""Plot rewards with error bar
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
import scipy.stats as ss

def get_arguments():
    parser = argparse.ArgumentParser(
        description="""
            This script processes an evaluation file which has possible
            multiple rollouts for a single experiment
        """
    )

    parser.add_argument('batch_eval_path', type=str, nargs='?',
                         help='Path to an `eval.info.json` file in json format')

    parser.add_argument('--max_paths', '-p', dest='max_paths',
                        type=int, default=1, nargs='?',
                        help='''Max. number of paths considered for each table''')

    return parser.parse_args()

if __name__ == '__main__':
        args = get_arguments()
        batch_eval_path = args.batch_eval_path
        max_paths = args.max_paths

        batch_eval_dir = dirname(batch_eval_path)
        filename = basename(batch_eval_path). \
            replace('.l.eval.info.json', '')
        rewards = []
        with open(batch_eval_path, 'r') as f:
            db = json.load(f)

        cycle = db['cycle']

        horizon = db['horizon']
        rollout_ids = db['rollouts']
        ex_ids = db['id']
        num_cycles = int(horizon) / cycle
        num_rollouts = db['num_rollouts']

        roll_paths = defaultdict(list)
        # consolidate experiments' rewards
        for idx, ex_id in enumerate(db['id']):
            path_idxs = np.random.choice(num_rollouts, size=max_paths)

            # iterate for each experiment extracting the paths
            for roll_id, roll_rews in db['rewards'][idx].items():
                # _rewards (cycles, num_rollouts)
                roll_paths[int(roll_id)] += \
                    [rw for i, rw in enumerate(roll_rews) if i in path_idxs]

        y = {}
        y_error = {}
        legends = []
        # This loop agreggates for each cycle # == roll_id
        # The resulting paths
        for roll_id, rewards in roll_paths.items():
            rewards = sorted(np.concatenate(rewards))
            y[roll_id] = np.mean(rewards)
            y_error[roll_id] = ss.t.ppf(0.95, len(rewards)) * np.std(rewards)
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
            f'{filename}\n(95% CI, cycles:{num_cycles},N:{num_rollouts})'
        ax.set_xticklabels(legends)
        plt.title(title)
        plt.savefig(f'{batch_eval_dir}/rollouts.png')
        plt.show()
