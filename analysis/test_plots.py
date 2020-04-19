import os
import json
import pandas as pd
import argparse
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('ggplot')

ILURL_HOME = os.environ['ILURL_HOME']

EMISSION_PATH = \
    f'{ILURL_HOME}/data/emissions'

EXCLUDE_EMISSION = ['CO', 'CO2', 'HC', 'NOx', 'PMx', 'angle', 'eclass', 'electricity', 'fuel', 'noise']

FIGURE_X = 15.0
FIGURE_Y = 7.0

def get_arguments():
    parser = argparse.ArgumentParser(
        description="""
            This script evaluates a traffic light system.
        """
    )
    parser.add_argument('experiment_root_folder', type=str, nargs='?',
                        help='Experiment root folder.')

    return parser.parse_args()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def print_arguments(args):

    print('Arguments:')
    print('\tExperiment root folder: {0}\n'.format(args.experiment_root_folder))


def get_emissions(file_path, exclude_emissions=EXCLUDE_EMISSION):
    """Gets an emission file

    Parameters:
    ----------
    * file_path
    * exclude_emissions

    Return:
    ------
    * df pandas.DataFrame

    """
    df = pd.read_csv(file_path, sep=';', header=0, encoding='utf-8')

    # The token 'vehicle_' comes when using SUMOS's script
    # referece sumo/tools/xml2csv
    df.columns = [str.replace(str(name), 'vehicle_', '') for name in df.columns]
    df.columns = [str.replace(str(name), 'timestep_', '') for name in df.columns]

    df.set_index(['time'], inplace=True)

    # Drop rows where there's no vehicle
    df = df.dropna(axis=0, how='all')

    # Drop columns if needed
    if exclude_emissions is not None:
        df = df.drop(exclude_emissions, axis=1, errors='ignore')

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


def get_throughput(df_emission):

    # depending on the conversion options
    # and net configurations the field
    # might change labels.
    if 'edge_id' in df_emission.columns:
        col_edge = 'edge_id'
    else:
        col_edge = 'lane'

    in_junction = df_emission[col_edge].str.startswith(':')

    df_junction = df_emission[in_junction].sort_values(by=['id', 'time'])

    df_junction = df_junction.drop_duplicates(subset='id', keep='first').reset_index()

    df_junction = df_junction[['time','id']]

    return df_junction


def main(experiment_root_folder=None):

    if not experiment_root_folder:
        args = get_arguments()
        print_arguments(args)
        experiment_root_folder = args.experiment_root_folder

    # Prepare output folder.
    output_folder_path = '{0}/plots/'.format(experiment_root_folder)
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Get cycle length from parameters (*.params.json) file.
    params = None
    for params_path in Path(experiment_root_folder).rglob('*.params.json'):
        with params_path.open('r') as f:
            params = json.load(f)
        break   # There should be only one match.
    if params is None:
        raise ValueError('params is None')
    cycle_time = params['env_args']['additional_params']['cycle_time']

    # Get all *.csv files from experiment root folder.
    csv_files = [str(p) for p in list(Path(experiment_root_folder).rglob('*-emission.csv'))]
    
    print('Number of csv files found: {0}'.format(len(csv_files)))

    vehicles_appended = []
    throughputs = []

    for csv_file in csv_files:
        
        print('CSV file: {0}'.format(csv_file))
        
        # Load CSV data.
        df_csv = get_emissions(csv_file)

        df_per_vehicle = get_vehicles(df_csv)
        vehicles_appended.append(df_per_vehicle)

        df_throughput = get_throughput(df_csv)
        throughputs.append(df_throughput)


    df_vehicles_appended = pd.concat(vehicles_appended)
    df_throughputs_appended = pd.concat(throughputs)
    
    print(df_vehicles_appended.shape)
    print(df_throughputs_appended.shape)

    """
        Waiting time & travel time.
    """
    # Describe waiting time.
    print('Waiting time:')
    df_stats = df_vehicles_appended['waiting'].describe()
    df_stats.to_csv('{0}/test_waiting_time_stats.csv'.format(output_folder_path),
                    float_format='%.3f')
    print(df_stats)
    print('\n')

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    sns.distplot(df_vehicles_appended['waiting'], hist=False, kde=True,
                kde_kws = {'linewidth': 3})

    plt.xlabel('Waiting time (s)')
    plt.ylabel('Density')
    plt.title('Waiting time')
    plt.savefig('{0}/test_waiting_time_hist.pdf'.format(output_folder_path))
    plt.savefig('{0}/test_waiting_time_hist.png'.format(output_folder_path))
    plt.close()

    # Describe travel time.
    print('Travel time:')
    df_stats = df_vehicles_appended['total'].describe()
    df_stats.to_csv('{0}/test_travel_time_stats.csv'.format(output_folder_path),
                    float_format='%.3f')
    print(df_stats)
    print('\n')

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    sns.distplot(df_vehicles_appended['total'], hist=False, kde=True,
                 kde_kws = {'linewidth': 3})

    plt.xlabel('Travel time (s)')
    plt.ylabel('Density')
    plt.title('Travel time')
    plt.savefig('{0}/test_travel_time_hist.pdf'.format(output_folder_path))
    plt.savefig('{0}/test_travel_time_hist.png'.format(output_folder_path))
    plt.close()

    # Describe vehicles' speed.
    print('Speed:')
    df_stats = df_vehicles_appended['speed'].describe()
    df_stats.to_csv('{0}/test_speed_stats.csv'.format(output_folder_path),
                    float_format='%.3f')
    print(df_stats)
    print('\n')

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    sns.distplot(df_vehicles_appended['speed'], hist=False, kde=True,
                 kde_kws = {'linewidth': 3})

    plt.xlabel('Average Speed (m/s)')
    plt.ylabel('Density')
    plt.title('Vehicles\' speed')
    plt.savefig('{0}/test_speeds_hist.pdf'.format(output_folder_path))
    plt.savefig('{0}/test_speeds_hist.png'.format(output_folder_path))
    plt.close()

    # Aggregate results per cycle.
    intervals = np.arange(0, df_vehicles_appended['finish'].max(), cycle_time)
    df_per_cycle = df_vehicles_appended.groupby(pd.cut(df_vehicles_appended["finish"], intervals)).mean()

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    Y = df_per_cycle['waiting'].values
    X = np.linspace(1, len(Y), len(Y))

    plt.plot(X,Y)
    plt.xlabel('Cycle')
    plt.ylabel('Average waiting time (s)')
    plt.title('Waiting time')
    plt.savefig('{0}/test_waiting_time.pdf'.format(output_folder_path))
    plt.savefig('{0}/test_waiting_time.png'.format(output_folder_path))
    plt.close()

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    Y = df_per_cycle['total'].values
    X = np.linspace(1, len(Y), len(Y))

    plt.plot(X,Y)
    plt.xlabel('Cycle')
    plt.ylabel('Average travel time (s)')
    plt.title('Travel time')
    plt.savefig('{0}/test_travel_time.pdf'.format(output_folder_path))
    plt.savefig('{0}/test_travel_time.png'.format(output_folder_path))
    plt.close()

    """
        Throughput.

        (throughput is calculated per cycle length)
    """
    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    intervals = np.arange(0, df_throughputs_appended['time'].max(), cycle_time)
    df = df_throughputs_appended.groupby(pd.cut(df_throughputs_appended["time"], intervals)).count()

    Y = df['time'].values
    X = np.linspace(1, len(Y), len(Y))

    plt.plot(X,Y)

    plt.xlabel('Cycle')
    plt.ylabel('#cars')
    plt.title('Throughput')

    plt.savefig('{0}/test_throughput.pdf'.format(output_folder_path))
    plt.savefig('{0}/test_throughput.png'.format(output_folder_path))

    plt.close()

    """ # JSON file.
    json_file = '{0}/{1}/{2}.eval.json'.format(EMISSION_PATH,
                                               args.experiment,
                                               args.experiment)
    print('JSON file: {0}\n'.format(json_file)) """

    """ # Load JSON data.
    with open(json_file) as f:
        json_data = json.load(f) """

    """
        Average number of vehicles.
    """
    """ fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    Y = json_data['vehicles']
    X = np.linspace(1, len(Y), len(Y))

    plt.plot(X,Y)

    plt.xlabel('Cycle')
    plt.ylabel('Average # of vehicles')
    plt.title('Number of vehicles')

    plt.savefig('{0}/#vehicles.pdf'.format(output_folder_path))
    
    plt.close() """

    """
        Average vehicles' velocity.
    """
    """ fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    Y = json_data['velocities']
    X = np.linspace(1, len(Y), len(Y))

    plt.plot(X,Y)

    plt.xlabel('Cycle')
    plt.ylabel('Average velocities')
    plt.title('Vehicles\' velocities')

    plt.savefig('{0}/velocities.pdf'.format(output_folder_path))

    plt.close() """

if __name__ == "__main__":
    main()