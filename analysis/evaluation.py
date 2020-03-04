"""Plot rewards with error bar
"""

__author__ = 'Guilherme Varela'
__date__ = '2020-01-22'

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
                rew = np.array(db['rewards']).sum(axis=1)
                rewards.append(rew)

        # convert rewards to rewards
        rewards = np.concatenate(rewards)
        y = np.mean(rewards, axis=0)
        err = np.std(rewards, axis=0)
        y_error = [err, err]
        num_iterations = len(y)
        x = [int(t / 90) * i for i in range(num_iterations)]


        # Must savefig.
        label = f'{config_dir[:2]}x{config_dir[2:]}'
        plt.xlabel('Cycles (90 sec.)')
        plt.ylabel('Reward (Km/h)')
        plt.errorbar(x, y, yerr=y_error, fmt='-o')
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






