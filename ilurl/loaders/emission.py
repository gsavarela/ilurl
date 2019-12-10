"""Emission files are the default output for a simulation but have two drawbacks:
    They are quite verbose, with a 24-hour period simulation spannig 2.3 GB emission.xml file
    They come encoded into a even verbosy xml format which doesn't play nice with conversion tool
    

    USAGE:
    > python /home/gsavarela/sumo/tools/xml/xml2csv.py \
            data/emissions/intersection_20191127-1302331574859753.5029278-emission.xml -s , \
            -o data/emissions/intersection_20191127-1302331574859753.5029278-emission.csv
"""

__author__ = 'Guilherme Varela'
__date__ = '2019-10-29'

from os.path import dirname, abspath, join

import pandas as pd
import numpy as np

from ilurl.loaders.induction_loops import get_induction_loops

from ilurl.loaders.induction_loops import groupby_induction_loops

EXCLUDE_EMISSION=['CO', 'CO2', 'HC', 'NOx', 'PMx', 'angle', 'eclass', 'electricity', 'fuel', 'noise']

def get_emissions_dir():
    _path = dirname(abspath(__file__))
    _path = '/'.join(_path.split('/')[:-2])
    _path = join(_path, *('data', 'emissions'))
    return _path


def get_emissions(scenario_id, emission_dir=None, exclude_emissions=EXCLUDE_EMISSION):
    """Gets an emission file

    Parameters:
    ----------
    * scenario_id
    * emission_dir
    * exclude_emissions

    Return:
    ------
    * df pandas.DataFrame

    Updates:
    -------
    * 2019-11-29:Add Column Filter & Rename
    """
    if emission_dir is None:
        path = get_emissions_dir()
    path = join(path, scenario_id)
    df = pd.read_csv(path, sep=',', header=0, encoding='utf-8')
    # The token 'vehicle_' comes when using SUMOS's script
    # referece sumo/tools/xml2csv
    df.columns = [str.replace(str(name), 'vehicle_', '') for name in df.columns]
    df.columns = [str.replace(str(name), 'timestep_', '') for name in df.columns]

    df.set_index(['time'], inplace=True)

    # Drop rows before the first second
    df = df[df.index >= 1.0]

    # Drop columns if needed
    if exclude_emissions is not None:
        df = df.drop(exclude_emissions, axis=1)

    return df


def get_vehicles(emissions_df):
    """Returns vehicle data

    Parameters:
    ----------
    * emissions_df: pandas DataFrame
        SEE get_emission

    Usage:
    -----
    ipdb> vehs_df = get_vehicles(emissions_df)
    ipdb> vehs_df.head()
               route finish  start  wait  total
    id
    flow_00.0  route309265401#0_0   11.3    1.0   0.0   10.3
    flow_00.1  route309265401#0_0   18.4    7.1   0.0   11.3
    flow_00.2  route309265401#0_2   24.0   13.3   0.0   10.7
    flow_00.3  route309265401#0_2   29.7   19.4   0.0   10.3
    flow_00.4  route309265401#0_2   36.1   25.6   0.0   10.5
    """
    # Builds a dataframe with vehicle starts
    start_df = pd.pivot_table(
        emissions_df.reset_index(),
        index=['id', 'route'], values='time',
        aggfunc=min
    ). \
    reset_index('route'). \
    rename(columns={'time': 'start'}, inplace=False)

    # Builds a dataframe with vehicle finish
    finish_df = pd.pivot_table(
        emissions_df.reset_index(),
        index='id', values='time',
        aggfunc=max
    ).\
    rename(columns={'time': 'finish'}, inplace=False)

    # Builds a dataframe with waiting times
    wait_df = pd.pivot_table(
        emissions_df.reset_index(),
        index='id', values='waiting',
        aggfunc=max
    ).\
    rename(columns={'time': 'wait'}, inplace=False)

    speed_df = pd.pivot_table(
        emissions_df.reset_index(),
        index='id', values='speed',
        aggfunc=np.mean
    ).\
    rename(columns={'time': 'speed'}, inplace=False)

    vehs_df = start_df.join(
        finish_df, on='id', how='inner',
    ). \
    sort_values('start', inplace=False). \
    join(wait_df, on='id', how='left')

    vehs_df['total'] = vehs_df['finish'] - vehs_df['start']

    vehs_df = vehs_df.join(
        speed_df, on='id', how='inner',
    )
    return vehs_df

def add_column_hour(df_emission):
    df_emission['hour'] = \
        df_emission['start'].apply(lambda x: int(x / 3600))

    return df_emission

def get_intersections(df_emission):
    """Intersection data"""

    df_intersection = pd.pivot_table(
        df_emission.reset_index(),
        index=['route', 'edge_id'],
        values=['id', 'time'],
        aggfunc=min
    ). \
    sort_values(['route', 'time'], inplace=False)
    return df_intersection

def get_throughput(df_emission):
    """Get throughtput per travel"""

    id_junction = df_emission['edge_id'].str.startswith(':')

    df_junction = pd.pivot_table(
        df_emission[id_junction].reset_index(),
        index=['id', 'edge_id'],
        values='time',
        aggfunc=max
    ). \
    sort_values('time', inplace=False). \
    reset_index(inplace=False)


    df_junction['time'] = df_junction['time'] + 0.1

    df_junction.set_index(['time', 'id'], inplace=True)

    df_lane = pd.pivot_table(
        df_emission[~id_junction].reset_index(),
        index=['id', 'edge_id'],
        values='time',
        aggfunc=min
    ). \
    sort_values('time', inplace=False). \
    reset_index(inplace=False). \
    set_index(['time', 'id'], inplace=False)

    df_throughput = df_junction.join(
        df_lane,
        how='inner',
        lsuffix='junc',
        rsuffix='lane'
    ). \
    rename(
        columns={'edge_idjunc': 'junc_id',
                 'edge_idlane': 'lane_id'}
    ). \
    reset_index(). \
    set_index(['junc_id', 'lane_id']). \
    sort_values('time')

    return df_throughput

if __name__ == '__main__':

    # 120 seconds version
    # intersection_id = \
    #     "intersection_20191029-1153521572350032.090861-emission.csv"

    # 900 seconds version
    # intersection_id = \
    #     "intersection_20191029-2030101572381010.8449962-emission.csv"

    # 4 hours version
    # intersection_id = \
    #     "intersection_20191031-1217351572524255.1619358-emission.csv"

    # 24 hours  version
    # intersection_id = \
    #     "intersection_20191029-2043371572381817.619804-emission.csv"

    # 3 HOURS
    # intersection_id = \
    # "intersection_20191126-1649271574786967.902462-emission.csv"

    # 6 HOURS
    # intersection_id = \
    # "intersection_20191127-1253571574859237.7474365-emission.csv"
    # "intersection_20191127-1302331574859753.5029278-emission.csv"
    # "intersection_20191127-0909171574845757.020673-emission.csv"
    # "intersection_20191126-1655111574787311.310213-emission.csv"
    intersection_id = \
        "intersection_20191209-1752391575913959.248359-emission.csv"
        # "intersection_20191209-1735131575912913.773466-emission.csv"
         # "intersection_20191202-1633431575304423.8852081.emission.csv"
        # "intersection_20191127-1110501574853050.976416-emission.csv"

        # "intersection_20191127-1302331574859753.5029278-emission.csv"
    df = get_emissions(intersection_id)
    vehs_df = get_vehicles(df)
    print(vehs_df)

    loops_df = get_induction_loops(('3:9',), workdays=True)
    loops_df = groupby_induction_loops(loops_df, width=5)

    # Convert time stamps into hours
    loops_df.reset_index(inplace=True)
    loops_df['hour'] = loops_df['Date'].apply(lambda x: x.hour)
    source_df = pd.pivot_table(
        loops_df,
        columns='hour',
        values='Count',
        aggfunc=sum
    ).melt(). \
    sort_values('hour'). \
    rename(columns={'value': 'source'}, inplace=False). \
    set_index('hour')

    # Compare emissions vs source
    vehs_df['hour'] = vehs_df['start'].apply(lambda x: int(x / 3600))
    emitted_df = pd.pivot_table(
        vehs_df.reset_index(),
        columns='hour',
        values='id',
        aggfunc=len
    ).melt().  \
    sort_values('hour').  \
    rename(columns={'value': 'emitted'}, inplace=False).  \
    set_index('hour')

    df = source_df.join(
        emitted_df,
        on='hour',
        how='inner'
    )
    print(df.head(24))


