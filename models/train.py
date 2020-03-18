"""Provides baseline for networks"""
__author__ = 'Guilherme Varela'
__date__ = '2020-01-08'

import os
import json
import argparse
import math
import time

from flow.core.params import SumoParams, EnvParams
from flow.envs.ring.accel import ADDITIONAL_ENV_PARAMS

from ilurl.envs.base import TrafficLightQLEnv, QL_PARAMS
from ilurl.envs.base import ADDITIONAL_TLS_PARAMS

from ilurl.core.params import QLParams
from ilurl.core.experiment import Experiment

from ilurl.networks.base import Network

# TODO: Generalize for any parameter
ILURL_HOME = os.environ['ILURL_HOME']

EMISSION_PATH = \
    f'{ILURL_HOME}/data/emissions/'

def get_arguments():
    parser = argparse.ArgumentParser(
        description="""
            This script runs a traffic light simulation based on
            custom environment with presets saved on data/networks
        """
    )

    parser.add_argument('network', type=str, nargs='?', default='intersection',
                        help='Network to be simulated')

    parser.add_argument('--experiment-time', '-t', dest='time', type=int,
                        default=360, nargs='?',
                        help='Simulation\'s real world time in seconds')

    parser.add_argument('--experiment-pickle', '-p', dest='pickle', type=str2bool,
                        default=True, nargs='?',
                        help='Whether to pickle the environment (allowing to reproduce)')

    parser.add_argument('--experiment-log', '-l', dest='log_info', type=str2bool,
                        default=False, nargs='?',
                        help='Whether to save experiment-related data in a JSON file \
                         thoughout training (allowing to live track training)')

    parser.add_argument('--experiment-log-interval',
                        dest='log_info_interval', type=int, default=20,
                        nargs='?',
                        help='[ONLY APPLIES IF --experiment-log is TRUE] \
                        Log into json file interval (in agent update steps)')

    parser.add_argument('--experiment-save-agent', '-a',
                        dest='save_RL_agent', type=str2bool,
                        default=False, nargs='?',
                        help='Whether to save RL-agent parameters throughout training')


    parser.add_argument('--sumo-render', '-r', dest='render', type=str2bool,
                        default=False, nargs='?',
                        help='Renders the simulation')

    parser.add_argument('--sumo-step', '-s',
                        dest='step', type=float, default=0.1, nargs='?',
                        help='Simulation\'s step size which is a fraction from horizon')

    parser.add_argument('--sumo-emission', '-e',
                        dest='emission', type=str2bool, default=False, nargs='?',
                        help='Saves emission data from simulation on /data/emissions')


    parser.add_argument('--tls-inflows-switch', '-W', dest='switch',
                        type=str2bool, default=False, nargs='?',
                        help='Assign higher probability of spawning a vehicle \
                        every other hour on opposite sides')

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
    print('\tExperiment time: {0}'.format(args.time))
    print('\tExperiment pickle: {0}'.format(args.pickle))
    print('\tExperiment log info: {0}'.format(args.log_info))
    print('\tExperiment log info interval: {0}'.format(args.log_info_interval))
    print('\tExperiment save RL agent: {0}'.format(args.save_RL_agent))

    print('\tSUMO render: {0}'.format(args.render))
    print('\tSUMO emission: {0}'.format(args.emission))
    print('\tSUMO step: {0}'.format(args.step))

    print('\tTLS inflows switch: {0}\n'.format(args.switch))


if __name__ == '__main__':

    args = get_arguments()

    print_arguments(args)

    inflows_type = 'switch' if args.switch else 'lane'
    network = Network(
        network_id=args.network,
        horizon=args.time,
        demand_type=inflows_type
    )

    path = f'{EMISSION_PATH}{network.name}/'
    if not os.path.isdir(path):
        os.mkdir(path)

    print('Experiment: {0}\n'.format(path))

    sumo_args = {
        'render': args.render,
        'print_warnings': False,
        'sim_step': args.step,
        'restart_instance': True
    }

    if args.emission:
        sumo_args['emission_path'] = path

    sim_params = SumoParams(**sumo_args)

    additional_params = {}
    additional_params.update(ADDITIONAL_ENV_PARAMS)
    additional_params.update(ADDITIONAL_TLS_PARAMS)
    additional_params['cycle_split'] = (30, 60)
    additional_params['target_velocity'] = 20

    env_params = EnvParams(evaluate=True,
                           additional_params=additional_params)

    phases_per_tls = [len(network.phases[t]) for t in network.tls_ids]
    ql_params = QLParams(epsilon=0.10, alpha=0.50,
                         states=('speed', 'count'),
                         rewards={'type': 'target_velocity',
                                  'costs': None},
                         phases_per_traffic_light=phases_per_tls,
                         choice_type='eps-greedy')

    env = TrafficLightQLEnv(
        env_params=env_params,
        sim_params=sim_params,
        ql_params=ql_params,
        network=network
    )

    exp = Experiment(env=env,
                    dir_path=path,
                    train=True,
                    log_info=args.log_info,
                    log_info_interval=args.log_info_interval,
                    save_agent=args.save_RL_agent,
                    )

    print('Running experiment...')

    start = time.time()

    info_dict = exp.run(
        int(args.time / args.step)
    )

    print(f'Elapsed time {time.time() - start}')

    # Save train log.
    filename = \
            f"{env.network.name}.train.json"

    info_path = os.path.join(path, filename)
    with open(info_path, 'w') as fj:
        json.dump(info_dict, fj)

    # Save parameters pickle.
    if args.pickle:

        if hasattr(env, 'dump'):
            env.dump(path)

