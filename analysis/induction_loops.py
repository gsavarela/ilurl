"""Provides analytics from induction loops data"""
__author__ = 'Guilherme Varela'
__date__ = '2019-10-10'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ilu.loaders.induction_loops import get_induction_loops

def remove_time(x):
    return x.split(' ')[0]

def remove_date(x):
    return x.split(' ')[1]

if __name__ == '__main__':
    print("Per reading")
    df = get_induction_loops()
    print(df['Count'].describe())


    print("Per time")
    df['Time'] = df.index.get_level_values('Data')
    df['Time'] = df['Time'].apply(remove_date)

    table = pd.pivot_table(df, values='Count', index='Time', aggfunc=(np.mean, np.std)).reset_index()


    print(table.describe())

    time_tick = table['Time'].values
    means = table['mean'].values
    stds = table['std'].values
    # fig, ax = plt.subplots()

    plt.figure()
    plt.xlabel('Time')
    plt.ylabel('# vehicles')
    plt.title('Induction Loops Readings')
    plt.plot(time_tick, means, label='Mean')
    plt.plot(time_tick, stds, marker='o', linestyle='--', color='r', label='Std')
    # ax.xaxis_date()
    # fig.autofmt_xdate()
    # plt.xticks(time_tick[::2], rotation=45)
    plt.legend()
    # plt.gcf().autofmt_xdate()

    # fig, ax = plt.subplots()

    # ax.errorbar(table['Time'].values, table['mean'], yerr=table['std'], fmt='o')
    # ax.set_title('Induction Loops By Time')

    plt.show()
