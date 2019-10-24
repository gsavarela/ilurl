"""Helper module to query the data"""
__author__ = 'Guilherme Varela'
__date__ = '2019-10-08'

import pandas as pd


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
    df = pd.read_csv("pthol2018.txt", sep=",", 
                     parse_dates=True, index_col=0, header=0, encoding="utf-8")

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
    * df dataframe object
        index: MultiIndex ("Data", "ID_Espira")
            "Data": is a datatime representation
            "ID_Espira": in format <zone_id>:<espira_number>

        columns:
            "Count":  Reading

    USAGE:
    -----
    >>> df = get_induction_loops()
    >>> df.head()
                                   Count
    Data                ID_Espira
    08-15-2018 00:00:00 3:16         395
                        3:9          496
    08-15-2018 00:15:00 3:16         358
                        3:9          377
    08-15-2018 00:30:00 3:16         365

    >>> series = df[df.index.get_level_values('ID_Espira') == '3:9']
    >>> series.head()
                                   Count
    Data                ID_Espira
    09-01-2018 00:00:00 3:9          173
    10-01-2018 00:00:00 3:9           79
    09-02-2018 00:00:00 3:9          128
    10-02-2018 00:00:00 3:9          103
    09-03-2018 00:00:00 3:9          142
    """
    df = pd.read_csv('data/sensors/induction_loops.csv', sep=',', header=0)
    del df['Contadores']
    df['ID_Espira'] = df['Zona'].apply(str) + ':' + df['ID_Espira'].replace(regex='[a-zA-Z]', value='')
    del df['Zona']
    df = df.melt(id_vars=('Data', 'ID_Espira'),
                 var_name='Time', value_name='Count')

    df['Data'] = pd.to_datetime(df['Data']).dt.strftime('%Y-%m-%d')

    def time_fix(x):
        h, m = x.split('h')
        return f'{int(h):02d}:{int(m):02d}:00'
    df['Time'] = df['Time'].apply(time_fix)
    df['Data'] = df['Data'] + ' ' + df['Time']
    del df['Time']
    df = df.set_index(['Data', 'ID_Espira'])
    df = df.sort_values(['Data','ID_Espira'], axis=0)

    if induction_loops is not None and any(induction_loops):
        search_index = df.index. \
                        get_level_values('ID_Espira'). \
                        isin(induction_loops)
        df = df.loc[search_index, :]

    if workdays:
        # filter by workdays
        search_index = pd.to_datetime(
                        df.index. \
                        get_level_values('Data')).dayofweek < 5

        df = df.loc[search_index, :]
    return df

if __name__ == '__main__':
    # builds a tick graph
    df = get_induction_loops()
    df = df[df.index.get_level_values('ID_Espira') == '3:9']
    assert df.equals(get_induction_loops(('3:9',)))
    assert len(df) > len((get_induction_loops(('3:9',), workdays=True)))

    df.reset_index(inplace=True)
    del df['ID_Espira']
    df['Data'].replace(regex=' dd:dd:dd', value='', inplace=True)
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    plt.title('Induction Loop ("Espira") 3:9')
    #ax.plot(df['Data'], df['Count'])
    # ax.xaxis_date()

    # plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')

    ax.plot_date(df['Data'], df['Count'], marker='', linestyle='-')

    fig.autofmt_xdate()
    plt.show()
