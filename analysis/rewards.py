"""Analyses aggregate experiment files e.g info and envs"""
__author__ = 'Guilherme Varela'
__date__ = '2020-02-13'
import pdb
import os
import argparse
import json
from glob import glob

import numpy as np
import matplotlib.pyplot as plt

ROOT_PATH = os.environ['ILURL_HOME']
EXPERIMENTS_PATH = f'{ROOT_PATH}/data/experiments/0x03/'
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
                        choices=('evaluate', 'train'),
                        default='evaluate',
                        help="db type `evaluate` or `train`")

    return parser.parse_args()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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
            cycle_time = CYCLE

        for nf, path in enumerate(paths):
            with open(path, 'r') as f:
                data = json.load(f)

            num_iter = len(data['per_step_returns'])
            num_steps = len(data['per_step_returns'][0])

            if period == 'cycles':
                p = cycle_time
            elif period == 'episodes':
                p = num_steps
            else:
                raise ValueError(f'period<{period}> not recognized!')

            if nf == 0:
                N = int((num_steps * num_iter) / p)
                rets = np.zeros((N,), dtype=float)
                vels = np.zeros((N,),  dtype=float)
                vehs = np.zeros((N,),  dtype=float)
                acts = np.zeros((N, num_dbs),  dtype=int)

            for i in range(num_iter):

                if period == 'episodes':
                    _vels = data["velocities"][i]
                    _rets = data["mean_returns"][i]
                    _vehs = data["vehicles"][i]

                    vels[i] = (nf * vels[i] + _vels) / (nf + 1)
                    rets[i] = (nf * rets[i] + _rets) / (nf + 1)
                    vehs[i] = (nf * vehs[i] + _vehs) / (nf + 1)
                else:

                    _rets = data['per_step_returns'][i]
                    _vels = data['per_step_velocities'][i]
                    _vehs = data['per_step_vehs'][i]
                    _acts = np.array(data["rl_actions"][i])
                    for t in range(0, num_steps, cycle_time):

                        cycle = int(i * (num_steps / cycle_time) + t / cycle_time)
                        ind = slice(t, t + cycle_time)
                        vels[cycle] = \
                            (nf * vels[cycle] + np.mean(_vels[ind])) / (nf + 1)
                        rets[cycle] = \
                            (nf * rets[cycle] + np.sum(_rets[ind])) / (nf + 1)
                        vehs[cycle] = \
                            (nf * vehs[cycle] + np.mean(_vehs[ind])) / (nf + 1)
                        # 0x02 per_step_actions
                        if len(_acts) == num_steps:
                            acts[cycle, nf] = \
                                round(np.array(_acts[ind]).mean())
                        else:
                            # 0x03 actions per decision
                            acts[cycle, nf] = _acts[cycle]

        _, ax1 = plt.subplots()
        if period == 'cycles':
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
        if period == 'cycles':
            ax1.set_xlabel(f'Cycles ({cycle_time} sec)')
        else:
            ax1.set_xlabel(f'Episodes ({num_steps} sec)')

        ax1.set_ylabel('Avg. Reward per Cycle', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax1.plot(rets, color=color)
        plt.title(f'{phase_split}:avg. cycle return\n({db_type}, n={num_dbs})')
        plt.savefig(f'{files_dir}{phase_split}_{db_type}_rewards_{period}.png')
        plt.show()

        # optimal action
        # TODO: allow for customize action
        optact = 0.0
        _, ax1 = plt.subplots()
        color = 'tab:orange'
        if period == 'cycles':
            ax1.set_xlabel(f'Cycles ({cycle_time} sec)')
        else:
            ax1.set_xlabel(f'Episodes ({num_steps} sec)')

        ax1.set_ylabel('ratio optimal action')

        cumacts = np.cumsum(acts == optact, axis=0)
        weights = np.arange(1, len(acts) + 1)

        plt.title(f'{phase_split}:% optimal action\n({db_type}, n={num_dbs})')
        legends = []
        for j in range(num_dbs):
            ax1.plot(cumacts[:, j] / weights)
            legends.append(f'e#{j}')
        plt.legend(legends, loc='lower right')
        plt.savefig(f'{files_dir}{phase_split}_{db_type}_optimal_cycles.png')
        plt.show()

