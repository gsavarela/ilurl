"""Makes a leaderboard out of experiments"""

__author__ = 'Guilherme Varela'
__date__ = '2019-01-14'

import json
import os
import pdb
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

ROOT = os.environ['ILURL_HOME']
CYCLE = 90      # agg. unit corresponding to all phases

EXPERIMENTS_DIR = \
    f"{ROOT}/data/experiments/0x00"

BASELINE_DIR = \
    f"{EXPERIMENTS_DIR}/4545"

paths = (
    '4545/intersection_20200116-1351371579182697.540679.eval.info.json',
    '5040/intersection_20200115-1952491579117969.542847.eval.info.json',
    '5436/intersection_20200115-1951581579117918.4341419.eval.info.json',
    '6030/intersection_20200115-1950341579117834.795777.eval.info.json',
)


if __name__ == '__main__':
    categories = defaultdict(list)
    
    for dirpath, dirnames, filenames in os.walk(EXPERIMENTS_DIR):

        if dirnames == []:
            # Get only .json and .csv
            # json aren't really needed but they have the demand
            # code on their name: `w` switch, `l` uniform
            filenames = sorted([
                f for f in filenames
                if f.split('.')[-1] in ('json')
            ])
            
            cycle = dirpath.split('/')[-1]
            if dirpath == BASELINE_DIR:
                # no training for baseline dir
                traineval = zip(filenames, filenames)
            else:
                traineval = zip(filenames[::2], filenames[1::2])

            for filetrain, fileeval in traineval:
                
                # assert the timestamps are equal
                tstrain = filetrain.split('.')[0]
                tseval = fileeval.split('.')[0]
                assert tstrain == tseval
                    
                # we don't really need the training files
                # but they have the demand code on their
                # name: `w` switch, `l` uniform

                demand_code = filetrain.split('.')[-3]
                demand = 'switch' if demand_code == 'w' else 'uniform'
                categories['demands'].append(demand)
                categories['scenarios'].append(filetrain.split('_')[0])
                categories['splits'].append(f'{cycle[:2]}/{cycle[2:]}')
                path = os.path.join(dirpath, fileeval)
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
                        start = cc * CYCLE
                        finish = (cc + 1) * CYCLE
                        trial = returns[ii][start:finish]
                        board[ii, cc] = np.nanmean(trial)

                    categories['series'].append(np.mean(board, axis=0))

    # paginate for all combinatios
    for scenario in sorted(set(categories['scenarios']), reverse=True):
        idxscn = [scenario == scn
                  for scn in categories['scenarios']]

        for demand in sorted(set(categories['demands'])):
            idx = [demand == dmd and idxscn[i]
                   for i, dmd in enumerate(categories['demands'])]

            series = [ss
                      for i, ss in enumerate(categories['series']) if idx[i]]
            labels = [lbl
                      for i, lbl in enumerate(categories['splits']) if idx[i]]

            # sort the series by label
            labels, series = zip(*
                sorted(
                    zip(labels, series),
                    key=lambda x: x[0]
                )
            )
            fig, ax = plt.subplots()
            ax.plot(range(nc), np.cumsum(series[0]), color='r',label=labels[0])
            ax.plot(range(nc), np.cumsum(series[1]), color='m',label=labels[1])
            ax.plot(range(nc), np.cumsum(series[2]), color='c',label=labels[2])
            ax.plot(range(nc), np.cumsum(series[3]), color='b',label=labels[3])

            ax.set_title(f'{scenario.title()}: {demand}')
            ax.set_xlabel('Cycles')
            ax.set_ylabel('Excess speed')
            plt.legend()
            plt.savefig(f'{EXPERIMENTS_DIR}/{scenario}-{demand}')
            plt.show()


