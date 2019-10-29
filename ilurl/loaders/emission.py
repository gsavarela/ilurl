"""Help function to read emission filed"""

__author__ = 'Guilherme Varela'
__date__ = '2019-10-29'

from os.path import dirname, abspath, join

import pandas as pd


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

    vehs_df = finish_df.join(
        start_df, on='id', how='inner',
    ). \
    sort_values('start', inplace=False). \
    join(wait_df, on='id', how='left')

    vehs_df['total'] = vehs_df['finish'] - vehs_df['start']
    return vehs_df

def get_routes(emissions_df):
    """Returns route data
    """
    raise NotImplementedError


if __name__ == '__main__':
    intersection_id = \
        "intersection_20191029-1153521572350032.090861-emission.csv"
    df = get_emissions(intersection_id)
    vehs_df = get_vehicles(df)
