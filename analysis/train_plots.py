"""
    Python script to produce the following train plots:
        - reward per cycle (with mean, std and smoothed curve)
        - number of vehicles per cycle (with mean, std and smoothed curve)
        - vehicles' velocity per cycle (with mean, std and smoothed curve)

    Given the path to the experiment root folder, the script searches
    for all *.train.json files and produces the previous plots by
    averaging over all json files.

    The output plots will go into a folder named 'plots', created inside
    the given experiment root folder.

"""
import os
import json
import pandas as pd
import argparse
import numpy as np

from pathlib import Path

#from scipy.signal import savgol_filter
import statsmodels.api as sm

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

EXPERIMENT_ROOT_FOLDER = '/home/pedro/ILU/ILU-RL/data/experiments/20200413/'

FIGURE_X = 15.0
FIGURE_Y = 7.0

STD_CURVE_COLOR = (0.88,0.70,0.678)
MEAN_CURVE_COLOR = (0.89,0.282,0.192)
SMOOTHING_CURVE_COLOR = (0.33,0.33,0.33)

if __name__ == "__main__":

    print('RUNNING analysis/train_plots.py')

    print('Input files:')
    # Get all *.train.json files from experiment root folder.
    train_files = []
    for path in Path(EXPERIMENT_ROOT_FOLDER).rglob('*.train.json'):
        train_files.append(str(path))
        print('\t{0}'.format(str(path)))

    # Prepare output folder.
    output_folder_path = os.path.join(EXPERIMENT_ROOT_FOLDER, 'plots')
    print('Output folder: {0}'.format(output_folder_path))
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    rewards = []
    vehicles = []
    velocities = []

    for run_name in train_files:

        # Load JSON data.
        with open(run_name) as f:
            json_data = json.load(f)

        """
            Rewards per time-step.
        """
        r = json_data['rewards']
        r = [a[0] for a in r]
        rewards.append(r)

        """ 
            Number of vehicles per time-step.
        """
        vehicles.append(json_data['vehicles'])

        """ 
            Vehicles' velocity per time-step.
        """
        velocities.append(json_data['velocities'])

    # Rewards per time-step.
    rewards = np.array(rewards)

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    Y = np.average(rewards, axis=0)
    Y_std = np.std(rewards, axis=0)
    X = np.linspace(1, rewards.shape[1], rewards.shape[1])

    #Y_hat = savgol_filter(Y, 3001, 3)
    lowess = sm.nonparametric.lowess(Y, X, frac=0.10)

    plt.plot(X,Y, label='Mean', c=MEAN_CURVE_COLOR)
    plt.plot(X,lowess[:,1], c=SMOOTHING_CURVE_COLOR, label='Smoothing')

    if rewards.shape[0] > 1:
        plt.fill_between(X, Y-Y_std, Y+Y_std, color=STD_CURVE_COLOR, label='Std')

    plt.xlabel('Cycle')
    plt.ylabel('Reward')
    plt.title('Rewards')
    plt.legend(loc=4)

    file_name = '{0}/rewards.pdf'.format(output_folder_path)
    plt.savefig(file_name)
    
    plt.close()

    # Number of vehicles per time-step.
    vehicles = np.array(vehicles)

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    Y = np.average(vehicles, axis=0)
    Y_std = np.std(vehicles, axis=0)
    X = np.linspace(1, vehicles.shape[1], vehicles.shape[1])

    lowess = sm.nonparametric.lowess(Y, X, frac=0.10)

    plt.plot(X,Y, label='Mean', c=MEAN_CURVE_COLOR)
    plt.plot(X,lowess[:,1], c=SMOOTHING_CURVE_COLOR, label='Smoothing')

    if vehicles.shape[0] > 1:
        plt.fill_between(X, Y-Y_std, Y+Y_std, color=STD_CURVE_COLOR, label='Std')

    plt.xlabel('Cycle')
    plt.ylabel('#Vehicles')
    plt.title('Number of vehicles')
    plt.legend(loc=4)

    file_name = '{0}/vehicles.pdf'.format(output_folder_path)
    plt.savefig(file_name)
    
    plt.close()

    # Vehicles' velocity per time-step.
    velocities = np.array(velocities)

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    Y = np.average(velocities, axis=0)
    Y_std = np.std(velocities, axis=0)
    X = np.linspace(1, velocities.shape[1], velocities.shape[1])

    lowess = sm.nonparametric.lowess(Y, X, frac=0.10)

    plt.plot(X,Y, label='Mean', c=MEAN_CURVE_COLOR)
    plt.plot(X,lowess[:,1], c=SMOOTHING_CURVE_COLOR, label='Smoothing')

    if velocities.shape[0] > 1:
        plt.fill_between(X, Y-Y_std, Y+Y_std, color=STD_CURVE_COLOR, label='Std')

    plt.xlabel('Cycle')
    plt.ylabel('Velocity')
    plt.title('Velocity of the vehicles')
    plt.legend(loc=4)

    file_name = '{0}/velocities.pdf'.format(output_folder_path)
    plt.savefig(file_name)
    
    plt.close()
