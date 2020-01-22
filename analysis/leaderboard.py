"""Makes a leaderboard out of experiments"""

__author__ = 'Guilherme Varela'
__date__ = '2019-01-14'

import json
import os
from collections import defaultdict

from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from ilurl.loaders.emission import (get_emissions, get_vehicles,
                                    get_throughput, add_column_hour)
ROOT = os.environ['ILURL_HOME']
CYCLE = 90      # agg. unit corresponding to all phases

EXPERIMENTS_DIR = \
    f"{ROOT}/data/experiments/0x00"

BASELINE_DIR = \
    f"{EXPERIMENTS_DIR}/4545"


def dataframe2markdown(df, showindex=False):
    return tabulate(df, headers="keys", showindex=showindex, tablefmt="github")

def build_table_flow(emission_df, scenario, cycle, demand):

    df_flow = get_vehicles(emission_df)
    
    # TODO: enhancement include `routes` elements
    # rou besides `vehicle` -- otherwise routes and 
    # vehicles are going to be the same.
    # enable this to spit the states for routes
    # flow_per_hour = pd.pivot_table(
    #     df_flow.reset_index(), index='route',
    #     values=['waiting', 'total'],
    #     aggfunc=(len, np.mean, np.std, np.min, np.median, np.max))


    df_flow['scenario'] = scenario
    df_flow['split'] = f'{cycle[:2]}/{cycle[2:]}'
    df_flow['demand'] = demand

    df_flow = pd.pivot_table(
        df_flow.reset_index(),
        index=('scenario', 'split', 'demand'),
        values=['waiting', 'total', 'speed'],
        aggfunc=(len, np.mean, np.std, np.min, np.median, np.max))
    
    return df_flow

def build_table_throughput(emission_df, scenario, cycle, demand):
    df_throughput = add_column_hour(
        get_throughput(emission_df),
        time_column='time'
    )

    # remove lane information
    df_throughput.reset_index(inplace=True)
    df_throughput['junc_id'] = \
        df_throughput['junc_id'].str.split('_') \
                                .str[0] \
                                .str.replace(':', '')

    # thoughput per hour
    df_throughput = pd.pivot_table(
        df_throughput.reset_index(),
        index=('junc_id', 'hour'),
        values='time',
        aggfunc=len
    )

    df_throughput['scenario'] = scenario
    df_throughput['split'] = f'{cycle[:2]}/{cycle[2:]}'
    df_throughput['demand'] = demand
    df_throughput = pd.pivot_table(
        df_throughput.reset_index(),
        index=('scenario', 'split', 'demand', 'junc_id'),
        values='time',
        aggfunc=(len, np.mean, np.std, np.min, np.median, np.max))

    return df_throughput


def plots():
    """This function saves the plots from the batches' evaluation"""
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


def tables():
    """This function builds tables from evaluations' emission files"""

    flows = []
    throughputs = []

    categories = defaultdict(list)
    for dirpath, dirnames, filenames in os.walk(EXPERIMENTS_DIR):

        if dirnames == []:
            # Get only .json and .csv
            # json aren't really needed but they have the demand
            # code on their name: `w` switch, `l` uniform
            filenames = sorted([
                f for f in filenames
                if f.split('.')[-1] in ('csv', 'json') and 
                 'eval.info.json' not in f
            ])
            
            cycle = dirpath.split('/')[-1]
            csvjsn = zip(filenames[::2], filenames[1::2])
            for csv, jsn in csvjsn:

                demand_code = jsn.split('.')[-3]
                demand = 'switch' if demand_code == 'w' else 'uniform'
                categories['demands'].append(demand)

                scenario = csv.split('_')[0].title()
                categories['scenarios'].append(scenario)
                emission_df = get_emissions(csv,
                                            emission_dir=dirpath)

                df_flow = build_table_flow(emission_df, scenario, cycle, demand)
                flows.append(df_flow)
                

                df_throughput = \
                    build_table_throughput(emission_df, scenario, cycle, demand)
                throughputs.append(df_throughput)
     

    # TODO: add option as latex
    dff = pd.concat(flows, axis=0). \
             sort_index(axis=0,
                        level=(0, 1, 2),
                        ascending=True,
                        inplace=False).round(2)
    
  
    dft = pd.concat(throughputs, axis=0). \
             sort_index(axis=0,
                        level=(0, 1, 2),
                        ascending=True,
                        inplace=False).round(2)

    import pdb
    pdb.set_trace()
    categories['scenarios'] = sorted(set(categories['scenarios']), reverse=True)
    categories['demands'] = sorted(set(categories['demands']), reverse=False)

    with open(os.path.join(EXPERIMENTS_DIR, 'README.md'), 'w') as f:
        idxspd = dff.columns.get_level_values(0) == 'speed'
        idxwat = dff.columns.get_level_values(0) == 'waiting'
        idxtrl = dff.columns.get_level_values(0) == 'total'

        for ii, scenario in enumerate(categories['scenarios']):
            f.write(f"\n# {ii + 1}.{scenario}\n")
            idxscn = dff.index.get_level_values('scenario') == scenario
            
            for jj, demand in enumerate(categories['demands']):

                idxdmd = dff.index.get_level_values('demand') == demand
                
                f.write(f"\n\n## {ii + 1}.{jj + 1} {demand}\n\n")
                f.write(f"\n\n### {ii + 1}.{jj + 1}.A Speed\n\n")

                df = dff.loc[idxscn & idxdmd, idxspd].reset_index()
                df.columns = df.columns.droplevel()
                f.write(dataframe2markdown(df))


                f.write(f"\n\n### {ii + 1}.{jj + 1}.B Travel time\n\n")
                df = dff.loc[idxscn & idxdmd, idxtrl].reset_index()
                df.columns = df.columns.droplevel()
                f.write(dataframe2markdown(df))

                f.write(f"\n\n### {ii + 1}.{jj + 1}.C Wait time\n\n")
                df = dff.loc[idxscn & idxdmd, idxwat].reset_index()
                df.columns = df.columns.droplevel()
                f.write(dataframe2markdown(df))
            
                f.write(f"\n\n### {ii + 1}.{jj + 1}.D Throughput\n\n")
            
                idxscn1 = dft.index.get_level_values('scenario') == scenario
                idxdmd1 = dft.index.get_level_values('demand') == demand
                df = dft.loc[idxscn1 & idxdmd1, :].reset_index()
                f.write(dataframe2markdown(df))

if __name__ == '__main__':
    tables()
    plots()
