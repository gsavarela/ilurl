'''Evaluation script for smart grid scenario'''

__author__ = 'Guilherme Varela'
__date__ = '2019-09-24'

import os
from glob import glob
import json
import argparse

import dill

from ilurl.core.experiment import Experiment
from ilurl.envs.base import TrafficLightQLEnv

# TODO: Generalize for any parameter
ILURL_HOME = os.environ['ILURL_HOME']

EMISSION_PATH = \
    f'{ILURL_HOME}/data/emissions/'

def get_arguments():
    parser = argparse.ArgumentParser(
        description="""
            This script runs a traffic light simulation based on
            custom environment with with presets saved on data/networks
        """
    )

    # TODO: validate against existing networks
    parser.add_argument('path_to_pickle', type=str, nargs='?',
                        default=f'{EMISSION_PATH}', help='Path to pickle')

    parser.add_argument('--experiment-time', '-t', dest='time', type=int,
                        default=360, nargs='?',
                        help='Simulation\'s real world time in seconds')

    parser.add_argument('--experiment-iterations', '-i',
                        dest='num_iterations', type=int,
                        default=1, nargs='?',
                        help='Number of times to repeat the experiment')

    return parser.parse_args()


def evaluate(envs, num_iterations, horizon, path):
    for i, env in enumerate(envs):
        env_eval = TrafficLightQLEnv(env.env_params,
                                     env.sim_params,
                                     env.ql_params,
                                     env.scenario)
        env_eval.dpq.Q = env.dpq.Q
        env_eval.dpq.stop = True
        env.sim_params.emission_path = path # always emit
        exp_eval = Experiment(env_eval)
        print(f"Running evaluation {i + 1}")
        info = exp_eval.run(num_iterations, horizon)
        ipath = os.path.join(path, f'{env_eval.scenario.name}.eval.info.json')
        with open(ipath, 'w') as f:
            json.dump(info, f)


        


if __name__ == '__main__':
    args = get_arguments()

    path_to_pickle = args.path_to_pickle
    num_iterations = args.num_iterations
    time = args.time

    paths = glob(f"{path_to_pickle}/*.pickle")
    if not any(paths):
        raise Exception("Environment pickles must be saved on root")

    envs = []
    for path in paths:
        with open(path, mode='rb') as f:
            envs.append(dill.load(f))
    evaluate(envs, num_iterations, time, path_to_pickle)
