"""Provides analytics from induction loops data
"""

__author__ = 'Guilherme Varela'
__date__ = '2019-10-10'

import pandas as pd
import numpy as np

from ilurl.loaders.induction_loops import get_induction_loops
from ilurl.utils.plots import plot_times

def remove_time(x):
    return x.split(' ')[0]

def remove_date(x):
    return x.split(' ')[1]

if __name__ == '__main__':
    df = get_induction_loops()
    dates = df.index.get_level_values('Data')
    sensors = df.index.get_level_values('ID_Espira')

    print("x--------------------Header--------------------x")
    print("Start date:" , min(dates))
    print("Finish date:", max(dates))
    print("# sensors:", len(sensors))
    print(df.describe())

    print("x--------------------Per reading--------------------x")
    print("Per reading")
    print(df['Count'].describe())


    print("x--------------------Per time--------------------x")
    df['Time'] = df.index.get_level_values('Data')
    df['Time'] = df['Time'].apply(remove_date)

    table = pd.pivot_table(df, values='Count', index='Time', aggfunc=(np.mean, np.std)).reset_index()


    print(table.describe())

    # place that up
    time_tick = table['Time'].values
    means = table['mean'].values
    stds = table['std'].values

    plot_times(time_tick, [means, stds], ["Means", "Stds"], ["Time", "# vehicles"], "Induction Loop Readings")
