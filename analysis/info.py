"""Analyses aggregate experiement files e.g info"""
__author__ = 'Guilherme Varela'
__date__ = '2020-02-13'
import argparse
import json
from glob import glob
import pdb
import numpy as np
import matplotlib.pyplot as plt


# TODO: make optional input params / discover
EMISSION_PATH = 'data/emissions/6030/'
#FILENAME = 'intersection_20200217-1325031581945903.078002.9000.w.info.json'
FILENAME = 'intersection_20200217-1325031581945903.078002'
CYCLE = 90
TIME = 9000


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
    # number of cycles per iteration

    db_type = parser.db_type
    if db_type == 'evaluate':
        ext = 'eval.info.json'
    elif db_type == 'train':
        ext = '[w|l].info.json'
    else:
        raise ValueError(f'db_type<{db_type}> not recognized!')

    paths = glob(f"{EMISSION_PATH}*.{ext}")
    num_dbs = len(paths)
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

                    vels[cycle] = \
                        (nf * vels[cycle] + np.mean(_vels[t:t + CYCLE])) / (nf + 1)
                    rets[cycle] = \
                         (nf * rets[cycle] + np.sum(_rets[t:t + CYCLE])) / (nf + 1)
                    vehs[cycle] = \
                        (nf * vens[cycle] + np.mean(_vehs[t:t + CYCLE])) / (nf + 1)
    _, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    if period == 'cycle':
        ax1.set_xlabel(f'Cycles ({CYCLE} sec)')
    else:
        ax1.set_xlabel(f'Episodes ({num_steps} sec)')

    ax1.set_ylabel('Avg. speed', color='b')
    ax2.set_ylabel('Avg. number of cars', color='r')
    ax1.plot(vels, 'b-')
    ax2.plot(vehs, 'r-')
    plt.legend(('Speed', 'Count'))
    plt.title(f'Speed and Count by {period} (n={num_dbs})')
    plt.show()


    _, ax1 = plt.subplots()
    ax1.set_ylabel('Avg. Reward per Cycle', color='c')
    ax1.plot(rets, 'c-')
    plt.title(f'Avg. Cycle Return per {period} (n={num_dbs})')
    ax1.plot(rets, 'c-')
    if period == 'cycle':
        ax1.set_xlabel(f'Cycles ({CYCLE} sec)')
    else:
        ax1.set_xlabel(f'Episodes ({num_steps} sec)')
    plt.show()
