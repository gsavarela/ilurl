"""Makes a leaderboard out of experiments"""

__author__ = 'Guilherme Varela'
__date__ = '2019-01-14'

import json
import os

import numpy as np
import matplotlib.pyplot as plt

ROOT = os.environ['ILURL_HOME']
CYCLE = 90      # agg. unit corresponding to all phases
data_dir = f'{ROOT}/data/emissions/'

paths = (
    '4545/intersection_20200116-1351371579182697.540679.eval.info.json',
    '5040/intersection_20200115-1952491579117969.542847.eval.info.json',
    '5436/intersection_20200115-1951581579117918.4341419.eval.info.json',
    '6030/intersection_20200115-1950341579117834.795777.eval.info.json',
)

if __name__ == '__main__':
    series = []
    labels = []
    for i, path in enumerate(paths):
        print(os.path.join(data_dir, path))
        with open(os.path.join(data_dir, path), 'r') as f:
            stats = json.load(f)

        returns = stats['per_step_returns']
        ni = len(stats['per_step_returns'])
        total = len(stats['per_step_returns'][0])
        nc = int(total / CYCLE)


        board = np.zeros((ni, nc), dtype=np.float)
        # number of iterations
        for ii in range(ni):
            # number of cycles
            for cc in range(nc):
                start = cc * CYCLE
                finish = (cc + 1) * CYCLE
                trial = returns[ii][start:finish]
                board[ii, cc] = np.nanmean(trial)

            series.append(np.mean(board, axis=0))
            labels.append(path.split('/')[0])

    fig, ax = plt.subplots()

    ax.plot(range(nc), np.cumsum(series[0]), color='r',label=labels[0])
    ax.plot(range(nc), np.cumsum(series[1]), color='m',label=labels[1])
    ax.plot(range(nc), np.cumsum(series[2]), color='c',label=labels[2])
    ax.plot(range(nc), np.cumsum(series[3]), color='b',label=labels[3])

    ax.set_title('Accumulated sum of speeds')
    ax.set_xlabel('Cycles')
    ax.set_ylabel('Excess speed')
    plt.legend()
    plt.show()


