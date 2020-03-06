"""Plot rewards with error bar
"""

__author__ = 'Guilherme Varela'
__date__ = '2020-01-22'
import pdb
import os
import json
from glob import glob

# third-party libs
import matplotlib.pyplot as plt
import numpy as np

ROOT_PATH = os.environ['ILURL_HOME']
EXPERIMENTS_PATH = f'{ROOT_PATH}/data/experiments/0x04/'
# EXPERIMENTS_PATH = f'{ROOT_PATH}/data/emissions/'

CONFIG_DIRS = ('4545', '5040', '5436', '6030')

if __name__ == '__main__':
    for config_dir in CONFIG_DIRS:
        path = f'{EXPERIMENTS_PATH}{config_dir}/'
        file_paths = sorted(glob(f"{path}*.eval.info.json"))

        rewards = []
        for file_path in file_paths:
            with open(file_path, 'r') as f:
                db = json.load(f)
                reward = np.concatenate(db['rewards'], axis=0)
                rewards.append(reward)

        # convert rewards to rewards
        rewards = np.vstack(rewards).T
        t, n = rewards.shape
        y = np.mean(rewards, axis=1)
        err = np.std(rewards, axis=1)
        y_error = [err, err]
        num_iterations = len(y)

        # Must savefig.
        label = f'{config_dir[:2]}x{config_dir[2:]}'
        plt.xlabel('Cycles (90 sec.)')
        plt.ylabel('Reward (Km/h)')
        plt.errorbar(np.arange(1, t + 1), y, yerr=y_error, fmt='-o')
        plt.title(f'Evaluation Phases: {label}')
        plt.savefig(f'{path}{label}.png')
        plt.show()

        # Must recover best policy and experiment
        batch, policy = np.unravel_index(
            np.argmax(rewards, axis=None),
            rewards.shape
        )
        
        # policyid == policy_index first policy in the 
        # array is random
        filename = file_path.split('/')[-1].split('.')[0]
        policy_path = f'{path}{filename}.Q.{policy}.pickle' 
        print(f'Best policy for {label}:\n{policy_path}')






