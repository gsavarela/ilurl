"""Helper module to query the data"""
__author__ = 'Guilherme Varela'
__date__ = '2019-10-08'

import pandas as pd


def get_network():
    df = pd.read_csv('data/points/points.csv', sep=',', header=0)
    del df['Contadores']
    df['ID_Espira'] = df['Zona'].apply(str) + ':' + df['ID_Espira']
    del df['Zona']
    df = df.melt(id_vars=('Data', 'Zona', 'ID_Espira'),
                 var_name='Time', value_name='Count')
    df.set_index(['Data', 'Time', 'ID_Espira'])
    return df
