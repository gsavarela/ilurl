"""This script helps compare emissions' files

    USAGE:
    ------
    * Making different configurations i.e
        --step-length 0.1 vs --step-length 1.0
    * ONLINE learning vs OFFLINE learning: FLOW vs SUMO

    * Perform experiments comparison

"""
__author__ = 'Guilherme Varela'
__date__ = '2019-12-03'
import pdb
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from ilurl.loaders.emission import (get_emissions, get_vehicles,
                                    get_throughput, add_column_hour)

from ilurl.utils.markdown import dataframe2markdown

EXPERIMENTS_DIR = \
    "data/experiments/0x00"

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


if __name__ == '__main__':
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
