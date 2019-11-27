"""Help function to read emission filed"""

__author__ = 'Guilherme Varela'
__date__ = '2019-10-29'

from os.path import dirname, abspath, join

import pandas as pd
import numpy as np

from ilurl.loaders.induction_loops import get_induction_loops

from ilurl.loaders.induction_loops import groupby_induction_loops

def get_emissions_dir():
    _path = dirname(abspath(__file__))
    _path = '/'.join(_path.split('/')[:-2])
    _path = join(_path, *('data', 'emissions'))
    return _path


def get_emissions(scenario_id, emission_dir=None):
    """Gets an emission file

    Parameters:
    ----------
    * scenario_id
    * emission_dir

    Return:
    ------
    * df pandas.DataFrame

    """
    if emission_dir is None:
        path = get_emissions_dir()
    path = join(path, scenario_id)
    df = pd.read_csv(path, sep=',', index_col=0, header=0, encoding='utf-8')
    return df

def get_vehicles(emissions_df):
    """Returns vehicle data

    Usage:
    -----
    ipdb> vehs_df = get_vehicles(emissions_df)
    ipdb> vehs_df.head()
               finish  start  wait  total
    id
    flow_00.0    11.3    1.0   0.0   10.3
    flow_00.1    18.4    7.1   0.0   11.3
    flow_00.2    24.0   13.3   0.0   10.7
    flow_00.3    29.7   19.4   0.0   10.3
    flow_00.4    36.1   25.6   0.0   10.5
    """
    # Builds a dataframe with vehicle starts
    start_df = pd.pivot_table(
        emissions_df.reset_index(),
        columns='id', values='time',
        aggfunc=min
    ). \
    melt().\
    set_index('id'). \
    rename(columns={'value': 'start'}, inplace=False)


    # Builds a dataframe with vehicle finish
    finish_df = pd.pivot_table(
        emissions_df.reset_index(),
        columns='id', values='time',
        aggfunc=max
    ).\
    melt(). \
    set_index('id'). \
    rename(columns={'value': 'finish'}, inplace=False)

    # Builds a dataframe with waiting times
    wait_df = pd.pivot_table(
        emissions_df.reset_index(),
        columns='id', values='waiting',
        aggfunc=max
    ).\
    melt(). \
    set_index('id'). \
    rename(columns={'value': 'wait'}, inplace=False)

    speed_df = pd.pivot_table(
        emissions_df.reset_index(),
        columns='id', values='speed',
        aggfunc=np.mean
    ).\
    melt(). \
    set_index('id'). \
    rename(columns={'value': 'speed'}, inplace=False)

    vehs_df = finish_df.join(
        start_df, on='id', how='inner',
    ). \
    sort_values('start', inplace=False). \
    join(wait_df, on='id', how='left')

    vehs_df['total'] = vehs_df['finish'] - vehs_df['start']

    vehs_df = vehs_df.join(
        speed_df, on='id', how='inner',
    ) 
    return vehs_df


def get_routes(emissions_df):
    """Returns route data
    """
    raise NotImplementedError


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
    intersection_id = \
    "intersection_20191127-0909171574845757.020673-emission.csv"
    # "intersection_20191126-1655111574787311.310213-emission.csv"
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
    ).melt(). \
    sort_values('hour'). \
    rename(columns={'value': 'emitted'}, inplace=False). \
    set_index('hour')


    df = source_df.join(
        emitted_df,
        on='hour',
        how='inner'
    )
    print(df.head(24))


