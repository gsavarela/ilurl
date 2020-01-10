"""Helper module to query the data"""
__author__ = 'Guilherme Varela'
__date__ = '2019-10-08'

import os
import numpy as np
import pandas as pd

from datetime import datetime
from datetime import timedelta


ILURL_HOME = os.environ['ILURL_HOME']
DIR = \
    f'{ILURL_HOME}/data/'

# pd.set_option('mode.chained_assignment', 'raise')
def get_holidays():
    """Returns the Portugal's Holidays table

        RETURNS:
        --------
        * df dataframe object
            index:
                Date: date
            columns:
                Description: string indicating why it's a holiday

    """
    df = pd.read_csv(f"{DIR}/calendar/pthol2018.txt", sep=",",
                     parse_dates=True, index_col=0, header=0, encoding="utf-8")

    return df


def get_induction_loops(induction_loops=None, workdays=False):
    """Returns induction loops data from db

    PARAMETERS:
    -----------
    * induction_ids None, a list or list-like
        Uses these values to filter

    * workdays boolean
        If True filters weekends and holidays out of results

    RETURNS:
    --------
    * df pd.DateFrame
        index: MultiIndex ("Date", "ID_Loop")
            "Date": is a datatime representation
            "ID_Loop": in format <zone_id>:<espira_number>

        columns:
            "Count":  Reading

    USAGE:
    -----
    >>> df = get_induction_loops()
    >>> df.head()
                                   Count
    Date                ID_Loop
    08-15-2018 00:00:00 3:16         395
                        3:9          496
    08-15-2018 00:15:00 3:16         358
                        3:9          377
    08-15-2018 00:30:00 3:16         365

    >>> series = df[df.index.get_level_values('ID_Loop') == '3:9']
    >>> series.head()
                                   Count
    Date                ID_Loop
    09-01-2018 00:00:00 3:9          173
    10-01-2018 00:00:00 3:9           79
    09-02-2018 00:00:00 3:9          128
    10-02-2018 00:00:00 3:9          103
    09-03-2018 00:00:00 3:9          142

    UPDATES:
    -------
    2019-10-30: Purge holidays
    """
    df = pd.read_csv('data/sensors/induction_loops.csv', sep=',', header=0)
    df.rename({'Data': 'Date', 'ID_Espira': 'ID_Loop'}, axis=1, inplace=True)
    del df['Contadores']

    df['ID_Loop'] = df['Zona'].apply(str) + ':' + \
        df['ID_Loop'].replace(regex='[a-zA-Z]', value='')
    del df['Zona']


    df = df.melt(id_vars=('Date', 'ID_Loop'),
                 var_name='Time', value_name='Count')

    def to_timedelta(x):
        h, m = x.split('h')
        return timedelta(hours=int(h), minutes=int(m))

    df['Date'] = pd.to_datetime(df['Date']) + \
        df['Time'].apply(to_timedelta)

    df = df.set_index(['Date', 'ID_Loop'])
    df = df.sort_values(['Date','ID_Loop'], axis=0)


    if induction_loops is not None and any(induction_loops):
        search_index = df.index. \
                        get_level_values('ID_Loop'). \
                        isin(induction_loops)
        df = df.loc[search_index, :]

    if workdays:
        date_index = df.index.get_level_values('Date')
        # holidays: 2018 removes 2018-08-15
        hols_df = get_holidays()

        # filter by workdays
        search_index = date_index.dayofweek < 5 & \
                      (~date_index.isin(hols_df.index))
        df = df.loc[search_index, :]

    del df['Time']
    return df


def groupby_induction_loops(df, anchor_date=None, width=5, by_hour=True):
    """Groups by sensor data count for having time and id
 

    PARAMETERS:
    -----------
    * df: pd.DateFrame
        Return from get_induction_loops

        index: MultiIndex ("Date", "ID_Loop")
            "Date": is a datatime representation
            "ID_Loop": in format <zone_id>:<espira_number>

        columns:
            "Count":  Reading

    * anchor_date: datetime
        This is the datetime (inclusive) which is the newest observation

    * width: integer

    * by_hour: boolean
        if true groups data by hours

    RETURNS:
    --------
    * groupby_df pd.DateFrame
        index: DateTime

    """
    # handle input argument
    if anchor_date is None:
        anchor_date = max(df.index.get_level_values("Date"))

    if isinstance(anchor_date, str):
        anchor_date = datetime.strptime(
            anchor_date, "%Y-%m-%d %H:%M:%S")
    anchor_date = anchor_date.replace(hour=0, minute=0)

    # finds the upper & lower bound for window selection
    # upper bound: midnight for the day after
    newest = (anchor_date + timedelta(days=1)). \
        replace(hour=0, minute=0, second=0,
                microsecond=0)

    # lower bound: midnight for the day
    oldest = (anchor_date - timedelta(days=width - 1)). \
        replace(hour=0, minute=0, second=0,
                microsecond=0)

    # selects only the a window with width
    index = df.index.get_level_values("Date")
    search_index = (index >= oldest) & (index < newest)
    df = df.iloc[search_index, :]

    # refresh index & create timeonly column
    index = df.index.get_level_values("Date")
    df.loc[:, 'Time'] = \
        [dt - np.datetime64(dt, 'D') for dt in index.values]
    df.reset_index(inplace=True)

    # aggregates "Count" per "Time" while preserving "ID_Loop"
    # converts from time to anchor_date
    df = pd.pivot_table(df, columns='ID_Loop',
                        values='Count', index='Time', aggfunc=np.mean)
    df.index = [anchor_date + pd.to_timedelta(str(i)) for i in df.index]

    # Rename index to the default names
    df.index.name = 'Date'
    df = df.stack().to_frame()
    df.rename({0: 'Count'}, axis=1, inplace=True)

    if by_hour:
        # removes minutes and seconds from timestamps
        def to_hour(dt):
            return dt.replace(minute=0, second=0, microsecond=0)

        new_index = ('Date', 'ID_Loop')
        df = df.reset_index()
        df['Date'] = df['Date'].apply(to_hour)
        df = pd.pivot_table(
            df,
            index=new_index,
            values='Count'
        )

    return df

if __name__ == '__main__':
    # builds a tick graph
    df = get_induction_loops()
    df = df[df.index.get_level_values('ID_Loop') == '3:9']
    assert df.equals(get_induction_loops(('3:9',)))
    assert len(df) > len((get_induction_loops(('3:9',), workdays=True)))

    # df.reset_index(inplace=True)
    # del df['ID_Loop']
    # df['Date'].replace(regex=' dd:dd:dd', value='', inplace=True)
    # from matplotlib import pyplot as plt
    # fig, ax = plt.subplots()
    # plt.title('Induction Loop ("Espira") 3:9')
    # ax.plot_date(df['Date'], df['Count'], marker='', linestyle='-')

    # fig.autofmt_xdate()

    # plt.show()

    df = groupby_induction_loops(df)
    print(df.head())
