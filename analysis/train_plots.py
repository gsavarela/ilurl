import os
import json
import pandas as pd
import argparse
import numpy as np

#from scipy.signal import savgol_filter
import statsmodels.api as sm

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('ggplot')

ILURL_HOME = os.environ['ILURL_HOME']

EMISSION_PATH = \
    f'{ILURL_HOME}/data/emissions'


RUNS = ['/home/pedro/ILU/ILU-RL/data/experiments/20200413/intersection_20200413-1844511586799891.418139/',
        '/home/pedro/ILU/ILU-RL/data/experiments/20200413/intersection_20200413-1844511586799891.419535/',
        '/home/pedro/ILU/ILU-RL/data/experiments/20200413/intersection_20200413-1844511586799891.4183884/',
        '/home/pedro/ILU/ILU-RL/data/experiments/20200413/intersection_20200413-1844511586799891.4184618/',
        '/home/pedro/ILU/ILU-RL/data/experiments/20200413/intersection_20200413-1844511586799891.4187133/',
        '/home/pedro/ILU/ILU-RL/data/experiments/20200413/intersection_20200413-1844511586799891.4208024/',
        '/home/pedro/ILU/ILU-RL/data/experiments/20200413/intersection_20200413-1844511586799891.4208026/',
        '/home/pedro/ILU/ILU-RL/data/experiments/20200413/intersection_20200413-1844511586799891.4210486/',
        '/home/pedro/ILU/ILU-RL/data/experiments/20200413/intersection_20200413-1844511586799891.4213364/']

FIGURE_X = 15.0
FIGURE_Y = 7.0

OUTPUT_FOLDER_PATH = '/home/pedro/ILU/ILU-RL/data/outputs/intersection_20200407-1921201586283680.3731925/'

if __name__ == "__main__":

    # Prepare output folder.
    if not os.path.exists(OUTPUT_FOLDER_PATH):
        os.makedirs(OUTPUT_FOLDER_PATH)

    rewards = []
    vehicles = []
    velocities = []

    for run in RUNS:

        # JSON file.
        run_name = os.path.basename(os.path.normpath(run))
        json_file = '{0}{1}.train.json'.format(run, run_name)
        print(json_file)

        # Load JSON data.
        with open(json_file) as f:
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

    plt.plot(X,Y, label='Mean', c=(0.89,0.282,0.192))
    plt.plot(X,lowess[:,1], color=(0.33,0.33,0.33), label='Smoothing')
    plt.fill_between(X, Y-Y_std, Y+Y_std, color=(0.88,0.70,0.678), label='Std')

    plt.xlabel('Cycle')
    plt.ylabel('Reward')
    plt.title('Rewards')
    plt.legend(loc=4)

    file_name = '{0}rewards.pdf'.format(OUTPUT_FOLDER_PATH)
    plt.savefig(file_name)
    print(file_name)
    
    plt.close()

    # Number of vehicles per time-step.
    vehicles = np.array(vehicles)

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    Y = np.average(vehicles, axis=0)
    Y_std = np.std(vehicles, axis=0)
    X = np.linspace(1, vehicles.shape[1], vehicles.shape[1])

    lowess = sm.nonparametric.lowess(Y, X, frac=0.10)

    plt.plot(X,Y, label='Mean', c=(0.89,0.282,0.192))
    plt.plot(X,lowess[:,1], color=(0.33,0.33,0.33), label='Smoothing')
    plt.fill_between(X, Y-Y_std, Y+Y_std, color=(0.88,0.70,0.678), label='Std')

    plt.xlabel('Cycle')
    plt.ylabel('#Vehicles')
    plt.title('Number of vehicles')
    plt.legend(loc=4)

    file_name = '{0}vehicles.pdf'.format(OUTPUT_FOLDER_PATH)
    plt.savefig(file_name)
    print(file_name)
    
    plt.close()

    # Vehicles' velocity per time-step.
    velocities = np.array(velocities)

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    Y = np.average(velocities, axis=0)
    Y_std = np.std(velocities, axis=0)
    X = np.linspace(1, velocities.shape[1], velocities.shape[1])

    lowess = sm.nonparametric.lowess(Y, X, frac=0.10)

    plt.plot(X,Y, label='Mean', c=(0.89,0.282,0.192))
    plt.plot(X,lowess[:,1], color=(0.33,0.33,0.33), label='Smoothing')
    plt.fill_between(X, Y-Y_std, Y+Y_std, color=(0.88,0.70,0.678), label='Std')

    plt.xlabel('Cycle')
    plt.ylabel('Velocity')
    plt.title('Velocity of the vehicles')
    plt.legend(loc=4)

    file_name = '{0}velocities.pdf'.format(OUTPUT_FOLDER_PATH)
    plt.savefig(file_name)
    print(file_name)
    
    plt.close()
