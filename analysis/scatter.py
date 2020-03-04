"""This script makes a scatter plot from observation spaces

    Use this script to determine a discretization scheme

    USAGE:
    -----
    From root directory with files saved on root
    > python analysis/scatter.py

"""
__author__ = 'Guilherme Varela'
__date__ = '2020-02-22'
# core packages
from collections import defaultdict
import json
import os
from glob import glob


# third-party libs
import dill
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# current project dependencies
from ilurl.envs.base import TrafficLightQLEnv

ROOT_DIR = os.environ['ILURL_HOME']
EMISSION_DIR = f"{ROOT_DIR}/data/experiments/0x04/"
CONFIG_DIR = ('4545', '5040', '5436', '6030')
# CONFIG_DIR = ('5040', '6030')

if __name__ == '__main__':
    # this loop acumulates experiments
    ext = '.l.eval.info.json'
    phases = defaultdict(list)
    labels = []
    desired_velocity = None
    category_speeds = None
    category_counts = None
    for config_dir in CONFIG_DIR:
        lookup_jsons = f'{EMISSION_DIR}{config_dir}/*{ext}'
        for jf in glob(lookup_jsons):
            # Retrieves output data
            with open(jf, 'r') as f:
                output = json.load(f)

            filename = jf.replace(ext, '')
            # Retrieves agent data
            env = TrafficLightQLEnv.load(f"{filename}.pickle")

            if not labels:
                labels = env.ql_params.states_labels

            if not desired_velocity:
                additional_params = env.env_params.additional_params
                if 'target_velocity' in additional_params:
                    desired_velocity = \
                        additional_params['target_velocity']

            if not category_speeds:
                category_speeds = env.ql_params.category_speeds

            if not category_counts:
                category_counts = env.ql_params.category_counts

            # observation spaces
            if 'cycle' not in output:
                # deprecate: 0x00, 0x01, 0x02, 0x03
                observation_spaces = [output['observation_spaces']]
            else:
                observation_spaces = output['observation_spaces']

            for observation_space in observation_spaces:
                for intersection_space in observation_space:
                    for phase_space in intersection_space:
                        for i, phase in enumerate(phase_space):
                            phases[i] += [phase]

        _, ax = plt.subplots()
        for i, label in enumerate(labels):
            if i == 0:
                ax.set_xlabel(label)
            elif i == 1:
                ax.set_ylabel(label)


        ax.axvline(x=desired_velocity,
                   markerfacecoloralt='tab:purple',
                   label='target velocity')

        ax.vlines(category_speeds, 0, 1,
                  transform=ax.get_xaxis_transform(),
                  colors='tab:gray')

        ax.hlines(category_counts, 0, 1,
                  transform=ax.get_yaxis_transform(),
                  colors='tab:gray',
                  label='states')

        colors = ['tab:blue', 'tab:red']
        N = 0
        for i, phase in phases.items():
            x, y = zip(*phase)
            N += len(x)
            ax.scatter(x, y, c=colors[i], label=f'phase#{i}')

        split = f'{config_dir[:2]}x{config_dir[2:]}'
        ax.legend()
        ax.grid(True)
        plt.title(f'{split}: observation space (N={N})')
        plt.savefig(f'{EMISSION_DIR}{config_dir}/{split}_scatter.png')
        plt.show()
