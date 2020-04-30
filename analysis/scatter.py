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
import argparse

# third-party libs
import dill
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# project dependencies
from flow.core.params import EnvParams
 
# ilurl dependencies
from ilurl.core.params import QLParams
from ilurl.utils import TIMESTAMP_PROG


ROOT_DIR = os.environ['ILURL_HOME']


def get_arguments():
    parser = argparse.ArgumentParser(
        description="""
        This script plots the state space (variables) w.r.t the phases
        seem by each agent.
        """
    )
    parser.add_argument('experiment_dir', type=str, nargs='?',
                        help='Directory to the experiment')

    parser.add_argument('--evaluation', '-e', dest='eval_db',
                        type=str2bool, nargs='?', default=True,
                        help='''Either `train` or `evaluation` 
                            defaults to evaluation''')

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

if __name__ == '__main__':
    # this loop acumulates experiments
    args = get_arguments()
    exp_dir = args.experiment_dir
    eval_db = args.eval_db

    if eval_db:
        ext = '.l.eval.info.json'
    else:
        ext = 'train.json'

    phases = defaultdict(list)
    labels = []
    desired_velocity = None
    category_speeds = None
    category_counts = None
    output_json = glob(f'{exp_dir}*{ext}')[0]
    params_json = glob(f'{exp_dir}*params.json')[0]

    with open(output_json, 'r') as f:
        output = json.load(f)
    observation_spaces = output['observation_spaces']

    # on evaluation observation states are bound to
    # the iteration Q was saved.
    if eval_db and isinstance(observation_spaces, dict):
        key = max([int(k) for k in observation_spaces])
        observation_spaces = observation_spaces[str(key)]
    elif not eval_db:
        observation_spaces = [observation_spaces]


    with open(params_json, 'r') as f:
        params = json.load(f)
    ql_params = QLParams(**params['ql_args'])
    env_params = EnvParams(**params['env_args'])

    labels = ql_params.states_labels

    additional_params = env_params.additional_params
    if 'target_velocity' in additional_params:
        desired_velocity = \
            additional_params['target_velocity']

    category_speeds = ql_params.category_speeds

    category_counts = ql_params.category_counts

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

    filename = os.path.basename(output_json)
    filename = filename.replace(ext, '')

    result = TIMESTAMP_PROG.search(filename)
    if result:
        timestamp = f'_{result.group(0,)}'
        filename = filename.replace(timestamp, '')
    ax.legend()
    ax.grid(True)
    plt.title(f'{filename}:\nobservation space (N={N})')
    plt.savefig(f'{exp_dir}/scatter.png')
    plt.show()
