"""Plot rewards with error bar
"""

__author__ = 'Guilherme Varela'
__date__ = '2020-03-05'
import argparse
from os.path import dirname, basename
import json
from glob import glob

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

    # TODO: validate against existing networks
    parser.add_argument('evaluation_path', type=str, nargs='?',
                         help='Path to an evaluation file in json format')

    return parser.parse_args()

if __name__ == '__main__':
        args = get_arguments()
        file_path = args.evaluation_path
        file_dir = dirname(file_path)
        filename = basename(file_path). \
            replace('.l.eval.info.json', '') 
        rewards = []
        with open(file_path, 'r') as f:
            db = json.load(f)


        y = {}
        y_error = {}
        legends = []
        cycle = db['cycle']

        # TODO: change horizon to int
        horizon = db['horizon']
        if not isinstance(horizon, int):
            horizon = np.mean(list(db['horizon'].values()))
        num_cycles = int(horizon) / cycle
        num_rollouts = db.get('num_rollouts', 12)

        for cycles, rewards in db['rewards'].items():
            # _rewards (cycles, num_rollouts)
            rewards = sorted(np.concatenate(rewards))
            y[cycles] = np.mean(rewards)
            y_error[cycles] = ss.t.ppf(0.95, len(rewards)) * np.std(rewards)
            legends.append(f'Q[{cycles}]')

        # Must savefig.
        fig, ax = plt.subplots()
        plt.xlabel(f'Q-tables[train_cycles]')
        plt.ylabel('Reward')
        i = 0
        for cycles, yy in y.items():
            plt.errorbar([cycles], yy, yerr=y_error[cycles], fmt='-o')
            i += 1
        title = \
            f'{filename}\n(95% CI, cycles:{num_cycles},N:{num_rollouts})'
        ax.set_xticklabels(legends)
        plt.title(title)
        plt.savefig(f'{file_dir}/rollouts.png')
        plt.show()
