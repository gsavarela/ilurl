"""Makes a leaderboard out of experiments"""

__author__ = 'Guilherme Varela'
__date__ = '2019-01-14'

import json
import os

import numpy as np
import matplotlib.pyplot as plt

ROOT = os.environ['ILURL_HOME']
CYCLE = 90      # agg. unit corresponding to all phases

labels = ('5040', '5436')
filenames = ('intersection_20200114-1000431578996043.230082',
             'intersection_20200114-0932251578994345.995415')

path = f'{ROOT}/data/emissions/experiments/{labels[0]}'
path = f'{path}/{filenames[0]}-info.json'

with open(path, 'r') as f:
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
        board[ii, cc] = np.mean(returns[ii][cc*CYCLE:(cc+1)*CYCLE])

    board[ii, :] = board[ii, :]
performance = np.mean(board, axis=0)

plt.plot(np.cumsum(performance))
plt.show()


