"""Analyses aggregate experiement files e.g info"""
__author__ = 'Guilherme Varela'
__date__ = '2020-02-13'
import os
import argparse
import json
from glob import glob

import numpy as np
import matplotlib.pyplot as plt

ROOT_PATH = os.environ['ILURL_HOME']
EXPERIMENTS_PATH = f'{ROOT_PATH}/data/experiments/0x02/'
CYCLE = 90
TIME = 9000
# CONFIG_DIRS = ('4545', '5040', '5436', '6030')
CONFIG_DIRS = ('6030',)

def get_arguments():
    parser = argparse.ArgumentParser(
        description="""
            This script runs a traffic light simulation based on
            custom environment with with presets saved on data/networks
        """
    )

    # TODO: validate against existing networks
    parser.add_argument('period', type=str, nargs='?',
                        choices=('cycles', 'episodes'),
                        default='episodes', help='Cycle or episode')

    parser.add_argument('--type', '-t', dest='db_type', nargs='?',
                        choices=('evaluate', 'train'), default='evaluate',
                        help="db type `evaluate` or `train`")

    return parser.parse_args()


if __name__ == '__main__':
    parser = get_arguments()
    period = parser.period

    db_type = parser.db_type
    if db_type == 'evaluate':
        ext = 'eval.info.json'
    elif db_type == 'train':
        ext = '[w|l].info.json'
    else:
        raise ValueError(f'db_type<{db_type}> not recognized!')

    for config_dir in CONFIG_DIRS:
        files_dir = f'{EXPERIMENTS_PATH}{config_dir}/'

        paths = sorted(glob(f"{files_dir}*.{ext}"))
        num_dbs = len(paths)
        phase_split = f'{config_dir[:2]}x{config_dir[2:]}'
        try:
            cycle_time = int(config_dir[:2]) + int(config_dir[2:])
        except Exception:
            cycle_time = TIME

        for nf, path in enumerate(paths):
            with open(path, 'r') as f:
                data = json.load(f)

            num_iter = len(data['per_step_returns'])
            num_steps = len(data['per_step_returns'][0])

            if period == 'cycles':
                p = CYCLE
            elif period == 'episodes':
                p = num_steps
            else:
                raise ValueError(f'period<{period}> not recognized!')

            N = int((num_steps * num_iter) / p)
            rets = np.zeros((N,), dtype=float)
            vels = np.zeros((N,),  dtype=float)
            vehs = np.zeros((N,),  dtype=float)

            print(data.keys())
            for i in range(num_iter):
                _vels = data["velocities"][i]
                _rets = data["mean_returns"][i]
                _vehs = data["vehicles"][i]

                if period == 'episodes':
                    vels[i] = (nf * vels[i] + _vels) / (nf + 1)
                    rets[i] = (nf * rets[i] + _rets) / (nf + 1)
                    vehs[i] = (nf * vehs[i] + _vehs) / (nf + 1)
                else:

                    _rets = data['per_step_returns'][i]
                    _vels = data['per_step_velocities'][i]
                    _vehs = data['per_step_vehs'][i]
                    for t in range(0, num_steps, CYCLE):

                        cycle = int(i * (num_steps / CYCLE) + t / CYCLE)
                        ind = slice(t, t + cycle_time)
                        
                        vels[cycle] = \
                            (nf * vels[cycle] + np.mean(_vels[ind])) / (nf + 1)
                        rets[cycle] = \
                            (nf * rets[cycle] + np.sum(_rets[ind])) / (nf + 1)
                        vehs[cycle] = \
                            (nf * vehs[cycle] + np.mean(_vehs[ind])) / (nf + 1)

        _, ax1 = plt.subplots()
        if period == 'cycle':
            ax1.set_xlabel(f'Cycles ({cycle_time} sec)')
        else:
            ax1.set_xlabel(f'Episodes ({num_steps} sec)')

        color = 'tab:blue'
        ax1.set_ylabel('Avg. speed', color=color)
        ax1.plot(vels, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        color = 'tab:red'
        ax2 = ax1.twinx()
        ax2.set_ylabel('Avg. vehicles', color=color)
        ax2.plot(vehs, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title(f'{phase_split}:speed and count\n({db_type}, n={num_dbs})')
        plt.savefig(f'{files_dir}{phase_split}_{db_type}_velsvehs_{period}.png')
        plt.show()


        color = 'tab:cyan'
        _, ax1 = plt.subplots()
        if period == 'cycle':
            ax1.set_xlabel(f'Cycles ({cycle_time} sec)')
        else:
            ax1.set_xlabel(f'Episodes ({num_steps} sec)')

        ax1.set_ylabel('Avg. Reward per Cycle', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax1.plot(rets, color=color)
        plt.title(f'{phase_split}:avg. cycle return\n({db_type}, n={num_dbs})')
        plt.savefig(f'{files_dir}{phase_split}_{db_type}_rewards_{period}.png')
        plt.show()
