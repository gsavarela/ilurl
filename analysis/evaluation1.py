"""Plot rewards with error bar
"""

__author__ = 'Guilherme Varela'
__date__ = '2020-03-05'
import argparse
import pdb
import os
import json
from glob import glob

# third-party libs
import matplotlib.pyplot as plt
import numpy as np

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
        file_dir = os.path.dirname(file_path)
        config_dir = str(file_dir).split('/')[-1]
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
            # rewards (num_rollouts, 1, cycles)
            # _rewards (cycles, num_rollouts)
            _rewards = np.vstack(rewards).T
            y[cycles] = np.mean(_rewards)
            y_error[cycles] = np.std(_rewards)
            legends.append(f'Q[{cycles}]')

        # Must savefig.
        label = f'{config_dir[:2]}x{config_dir[2:]}'
        plt.xlabel(f'Q-tables[train_cycles]')
        plt.ylabel('Reward')
        for cycles, yy in y.items():
            plt.errorbar([cycles], yy, yerr=y_error[cycles], fmt='-o')
        title = \
            f'Evaluation Phases: {label}\n(cycles:{num_cycles},N:{num_rollouts})'
        plt.title(title)
        plt.legend(legends)
        plt.savefig(f'{file_dir}/{label}_rollouts.png')
        plt.show()
