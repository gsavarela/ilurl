"""Provides analytics from induction loops data
"""

__author__ = 'Guilherme Varela'
__date__ = '2019-10-10'

import pandas as pd
import numpy as np
from datetime import datetime
from ilurl.loaders.induction_loops import get_induction_loops
from ilurl.loaders.induction_loops import groupby_induction_loops
from ilurl.utils.plots import plot_times

def datetimeftimedelta(x):
    hrs, mns = x.components.hours, x.components.minutes
    return datetime.strptime(f"{hrs:02d}:{mns:02d}:00", "%H:%M:%S")

if __name__ == '__main__':
    # induction_loop_ids = None
    induction_loop_ids = ('3:9',)

    if induction_loop_ids is None:
        df = get_induction_loops(workdays=True)
    else:
        df = get_induction_loops(induction_loop_ids, workdays=True)

    # groupby_days = 0
    groupby_days = 21
    if groupby_days > 0:
        df = groupby_induction_loops(df, width=groupby_days)
    dates = df.index.get_level_values('Date')
    sensors = df.index.get_level_values('ID_Loop')

    print("x--------------------Header--------------------x")
    print("Start date:", min(dates))
    print("Finish date:", max(dates))
    print("# sensors:", len(sensors))
    print(df.describe())

    print("x--------------------Per reading--------------------x")
    print("Per reading")
    print(df['Count'].describe())

    print("x--------------------Per time--------------------x")
    df['Time'] = pd.to_timedelta(
        [np.datetime64(d) - np.datetime64(d, 'D') for d in dates])

    df['Time'] = df['Time'].apply(datetimeftimedelta)
    table = pd.pivot_table(df, values='Count', index='Time',
                           aggfunc=(np.mean, np.std)).reset_index()

    print(table.describe())


    # title descriptor
    if groupby_days  % 5 == 0:
        period = f"{int(groupby_days / 5):02d}-WEEK(S)"
    elif groupby_days  % 21 == 0:
        period = f"{int(groupby_days / 21):02d}-MONTH(S)"
    else:
        period = f"{groupby_days}-DAY(S)"
    title = f"Induction Loop Readings  {period} {induction_loop_ids}"

    time_tick = table['Time'].values
    means = table['mean'].values
    #  'std' column is only present when the data is not yet grouped
    if 'std' in table.columns:
        stds = table['std'].values

        plot_times(time_tick, [means, stds], ["Means", "Stds"],
                   ["Time", "# vehicles"], title)
    else:

        plot_times(time_tick, [means], ["Count"],
                   ["Time", "# vehicles"], title)
