"""This script helps compare emissions' files

    emission1.csv -- 1h36m, FLOW, sim_step=0.1
    emission2.csv --   01m, SUMO, step-length=1.0
    emission3.csv --   12m, SUMO, step-length=0.1

    USAGE:
    ------
    * Making different configurations i.e
        --step-length 0.1 vs --step-length 1.0
    * ONLINE learning vs OFFLINE learning: FLOW vs SUMO

"""
__author__ = 'Guilherme Varela'
__date__ = '2019-12-03'

import numpy as np
import pandas as pd

from ilurl.loaders.emission import get_emissions, get_vehicles
from ilurl.loaders.emission import add_column_hour, get_intersections, get_throughput

INTERSECTION_ID = \
    "intersection_20191202-1633431575304423.8852081"

def make_path(token):
    return \
        f"{INTERSECTION_ID:s}.emission{token:s}.csv"



if __name__ == '__main__':
    emission_df = get_emissions(make_path(""))
    flow_df = get_vehicles(emission_df)
    add_column_hour(flow_df)
    # fast_df = get_vehicles(get_emissions(make_path("2")))
    # add_column_hour(fast_df)
    # sumo_df = get_vehicles(get_emissions(make_path("3")))
    # add_column_hour(sumo_df)

    flow_per_hour = pd.pivot_table(
        flow_df.reset_index(), index='route',
        values=['waiting', 'total'],
        aggfunc=(len, np.mean, np.std, np.min, np.median, np.max))

    df_throughput = get_throughput(emission_df)
    # df_flow_intersections = get_intersections(emission_df)

    
    # fast_per_hour = pd.pivot_table(
    #     fast_df.reset_index(), index='route',
    #     values=['waiting', 'total'],
    #     aggfunc=(len, np.mean, np.std, np.min, np.median, np.max))

    # sumo_per_hour = pd.pivot_table(
    #     sumo_df.reset_index(), index='route',
    #     values=['waiting', 'total'],
    #     aggfunc=(len, np.mean, np.std, np.min, np.median, np.max))

    import pdb
    pdb.set_trace()



