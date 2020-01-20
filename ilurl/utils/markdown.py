"""Provides utilities for writing in markdown"""
__author__ = 'Guilherme Varela'
__date__ = '2020-01-20'

import pandas as pd

from tabulate import tabulate
# from IPython.display import Markdown, display

def dataframe2markdown(df, showindex=False):
    # fmt = ['---' for i in range(len(df.columns))]
    # df_fmt = pd.DataFrame([fmt], columns=df.columns)
    # df_formatted = pd.concat([df_fmt, df])
    # display(Markdown(df_formatted.to_csv(sep="|", index=False)))
    return tabulate(df, headers="keys", showindex=showindex, tablefmt="github")
