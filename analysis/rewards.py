"""Analyses aggregate experiment files e.g info and envs"""
__author__ = 'Guilherme Varela'
__date__ = '2020-02-13'
import os
import argparse
import json
from glob import glob

import numpy as np
import matplotlib.pyplot as plt

ROOT_PATH = os.environ['ILURL_HOME']

# EXPERIMENTS_PATH = f'{ROOT_PATH}/data/emissions/'
EXPERIMENTS_PATH = f'{ROOT_PATH}/data/experiments/0x04/'

CYCLE = 90
TIME = 9000
CONFIG_DIRS = ('4545', '5040', '5436', '6030')
# CONFIG_DIRS = ('5040',)

def get_arguments():
    parser = argparse.ArgumentParser(
        description="""
            This script runs a traffic light simulation based on
            custom environment with with presets saved on data/networks
        """
    )

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

        for nf, path in enumerate(paths):
            with open(path, 'r') as f:
                data = json.load(f)
                print(data.keys())

            try:
                if 'cycle' in data:
                    # new: information already comes in cycles
                    cycle_time = data['cycle']
                    group_by = False
                    
                else:
                    # legacy: information was either by episode or sim_step
                    # 0x00, 0x01, 0x02, 0x03
                    cycle_time = int(config_dir[:2]) + int(config_dir[2:])
                    group_by = True
            except Exception:
                cycle_time = CYCLE

            num_iter = len(data['rewards'])
            num_steps = len(data['rewards'][0])

            if group_by:
                if nf == 0:
                    N = int((num_steps * num_iter) / cycle_time)
                    rets = np.zeros((N,), dtype=float)
                    vels = np.zeros((N,),  dtype=float)
                    vehs = np.zeros((N,),  dtype=float)
                    acts = np.zeros((N, num_dbs),  dtype=int)

                for i in range(num_iter):

                    _rets = data['rewards'][i]
                    _vels = data['velocities'][i]
                    _vehs = data['vehicles'][i]
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
            else:
                if nf == 0:
                    N = num_steps * num_iter
                    rets = np.zeros((N, num_dbs), dtype=float)
                    vels = np.zeros((N, num_dbs), dtype=float)
                    vehs = np.zeros((N, num_dbs), dtype=float)
                    acts = np.zeros((N, num_dbs), dtype=int)

                # concatenates vertically the contents
                for i in range(num_iter):

                    start = i * num_steps
                    finish = (i + 1) * num_steps
                    _vehs = data['vehicles'][i]
                    _acts = np.array(data["rl_actions"][i])

                    vels[start:finish, nf] = data['velocities'][i]
                    rets[start:finish, nf] = data['rewards'][i]
                    vehs[start:finish, nf] = data['vehicles'][i]
                    acts[start:finish, nf] = np.array(data["rl_actions"][i]).flatten()
        _, ax1 = plt.subplots()
        ax1.set_xlabel(f'Cycles ({np.mean(cycle_time)} sec)')

        color = 'tab:blue'
        ax1.set_ylabel('Avg. speed', color=color)
        ax1.plot(np.mean(vels, axis=1), color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        color = 'tab:red'
        ax2 = ax1.twinx()
        ax2.set_ylabel('Avg. vehicles', color=color)
        ax2.plot(np.mean(vehs, axis=1), color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title(f'{phase_split}:speed and count\n({db_type}, n={num_dbs})')
        plt.savefig(f'{files_dir}{phase_split}_{db_type}_velsvehs.png')
        plt.show()


        color = 'tab:cyan'
        _, ax1 = plt.subplots()
        ax1.set_xlabel(f'Cycles ({np.mean(cycle_time)} sec)')

        ax1.set_ylabel('Avg. Reward per Cycle', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax1.plot(np.mean(rets, axis=1), color=color)
        plt.title(f'{phase_split}:avg. cycle return\n({db_type}, n={num_dbs})')
        plt.savefig(f'{files_dir}{phase_split}_{db_type}_rewards.png')
        plt.show()

        # optimal action
        # TODO: allow for customize action
        optact = 0.0
        _, ax1 = plt.subplots()
        color = 'tab:orange'
        ax1.set_xlabel(f'Cycles ({np.mean(cycle_time)} sec)')
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

