"""Provides baseline for networks

    References:
    ==========
    * seed:
    https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.RandomState.html#numpy.random.RandomState
    http://sumo.sourceforge.net/userdoc/Simulation/Randomness.html
"""
__author__ = 'Guilherme Varela'
__date__ = '2020-01-08'

import configargparse
import json
import os

import numpy as np
import random

from flow.core.params import EnvParams, SumoParams
from flow.envs.ring.accel import ADDITIONAL_ENV_PARAMS
from ilurl.core.experiment import Experiment
from ilurl.core.params import QLParams
# from ilurl.core.ql.dpq import DPQ, MAIQ

import ilurl.core.ql.dpq as ql
from ilurl.envs.base import TrafficLightEnv
from ilurl.networks.base import Network
# TODO: move this inside networks
from ilurl.loaders.nets import get_tls_custom

ILURL_HOME = os.environ['ILURL_HOME']

EMISSION_PATH = \
    f'{ILURL_HOME}/data/emissions/'

def get_arguments(config_file):

    if config_file is None:
        config_file = []

    flags = configargparse.ArgParser(
        default_config_files=config_file,
        description="""
            This script runs a traffic light simulation based on
            custom environment with presets saved on data/networks
        """
    )

    flags.add('--network', '-n', type=str, nargs='?', dest='network',
                        default='intersection',
                        help='Network to be simulated')


    flags.add('--experiment-buffer', '-b', dest='replay_buffer', type=str2bool,
                        default=True, nargs='?',
                        help='Turn on/off replay buffer')

    flags.add('--experiment-time', '-t', dest='time', type=int,
                        default=90000, nargs='?',
                        help='Simulation\'s real world time in seconds')

    flags.add('--experiment-log', '-l', dest='log_info',
						type=str2bool, default=False, nargs='?',
                        help='Whether to save experiment-related data in a JSON file \
                        thoughout training (allowing to live track training)')

    flags.add('--experiment-log-interval',
                        dest='log_info_interval', type=int, default=200,
                        nargs='?',
                        help='[ONLY APPLIES IF --experiment-log is TRUE] \
                        Log into json file interval (in agent update steps)')

    flags.add('--experiment-save-agent', '-a',
                        dest='save_agent', type=str2bool,
                        default=False, nargs='?',
                        help='Whether to save RL-agent parameters throughout training')

    flags.add('--experiment-save-agent-interval',
                        dest='save_agent_interval', type=int, default=500,
                        nargs='?',
                        help='[ONLY APPLIES IF --experiment-save-agent is TRUE] \
                        Save agent interval (in agent update steps)')

    flags.add('--experiment-seed', '-d', dest='seed', type=int,
                        default=None, nargs='?',
                        help='''Sets seed value for both rl agent and Sumo.
                               `None` for rl agent defaults to RandomState() 
                               `None` for Sumo defaults to a fixed but arbitrary seed''')

    flags.add('--sumo-render', '-r', dest='render', type=str2bool,
                        default=False, nargs='?',
                        help='Renders the simulation')

    flags.add('--sumo-step', '-s',
                        dest='step', type=float, default=1, nargs='?',
                        help='Simulation\'s step size which is a fraction from horizon')

    flags.add('--sumo-emission', '-e',
                        dest='emission', type=str2bool, default=False, nargs='?',
                        help='Saves emission data from simulation on /data/emissions')

    flags.add('--inflows-switch', '-W', dest='switch',
                        type=str2bool, default=False, nargs='?',
                        help='Assign higher probability of spawning a vehicle \
                        every other hour on opposite sides')

    flags.add('--env-normalize', dest='normalize',
                        type=str2bool, default=True, nargs='?',
                        help='If true will normalize grid and target')

    return flags.parse_args()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise configargparse.ArgumentTypeError('Boolean value expected.')

def print_arguments(args):

    print('Arguments (train.py):')
    print('\tExperiment network: {0}'.format(args.network))
    print('\tExperiment time: {0}'.format(args.time))
    print('\tExperiment seed: {0}'.format(args.seed))
    print('\tExperiment log info: {0}'.format(args.log_info))
    print('\tExperiment log info interval: {0}'.format(args.log_info_interval))
    print('\tExperiment save RL agent: {0}'.format(args.save_agent))
    print('\tExperiment save RL agent interval: {0}'.format(args.save_agent_interval))

    print('\tSUMO render: {0}'.format(args.render))
    print('\tSUMO emission: {0}'.format(args.emission))
    print('\tSUMO step: {0}'.format(args.step))

    print('\tInflows switch: {0}\n'.format(args.switch))

    print('\tNormalize state-space (speeds): {0}\n'.format(args.normalize))


def main(train_config=None):

    flags = get_arguments(train_config)
    print_arguments(flags)

    inflows_type = 'switch' if flags.switch else 'lane'
    network_args = {
        'network_id': flags.network,
        'horizon': flags.time,
        'demand_type': inflows_type,
        'insertion_probability': 0.1,
    }
    network = Network(**network_args)
    normalize = flags.normalize

    # Create directory to store data.
    path = f'{EMISSION_PATH}{network.name}/'
    if not os.path.isdir(path):
        os.mkdir(path)
    print('Experiment: {0}\n'.format(path))


    sumo_args = {
        'render': flags.render,
        'print_warnings': False,
        'sim_step': flags.step,
        'restart_instance': True
    }

    # Setup seeds.
    if flags.seed is not None:
        random.seed(flags.seed)
        np.random.seed(flags.seed)
        sumo_args['seed'] = flags.seed

    if flags.emission:
        sumo_args['emission_path'] = path
    sim_params = SumoParams(**sumo_args)

    # Load cycle time and TLS programs.
    cycle_time, programs = get_tls_custom(flags.network)

    additional_params = {}
    additional_params.update(ADDITIONAL_ENV_PARAMS)
    additional_params['target_velocity'] = 1.0 if normalize else 20
    additional_params['cycle_time'] = cycle_time
    env_args = {
        'evaluate': True,
        'additional_params': additional_params
    }
    env_params = EnvParams(**env_args)

    # Agent.
    phases_per_tls = [len(network.tls_phases[t]) for t in network.tls_ids]
    agent_id = 'DPQ' if len(network.tls_ids) == 1 else 'MAIQ'

    # Assumes all agents have the same number of actions.
    num_actions = len(programs[network.tls_ids[0]])

    category_counts = [5, 10, 15, 20, 25, 30]
    if normalize:
        category_speeds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    else:
        category_speeds = [2,3,4,5,6,7]

    ql_args = {
                'agent_id': agent_id,
                'epsilon': 0.10,
                'alpha': 0.50,
                'gamma': 0.90,
                'states': ('speed', 'count'),
                'rewards': {'type': 'target_velocity',
                         'costs': None},
                'phases_per_traffic_light': phases_per_tls,
                'num_actions': num_actions,
                'choice_type': 'eps-greedy',
                'category_counts': category_counts,
                'category_speeds': category_speeds,
                'normalize': normalize,
                'replay_buffer': flags.replay_buffer,
                'replay_buffer_size': 500,
                'replay_buffer_batch_size': 64,
                'replay_buffer_warm_up': 200,
    }
    ql_params = QLParams(**ql_args)

    cls_agent = getattr(ql, ql_params.agent_id)
    QL_agent = cls_agent(ql_params)

    env = TrafficLightEnv(
        env_params=env_params,
        sim_params=sim_params,
        agent=QL_agent,
        network=network,
        TLS_programs=programs
    )

    exp = Experiment(env=env,
                     dir_path=path,
                     train=True,
                     log_info=flags.log_info,
                     log_info_interval=flags.log_info_interval,
                     save_agent=flags.save_agent,
                     save_agent_interval=flags.save_agent_interval
                    )

    # Store parameters.
    parameters = {}
    parameters['network_args'] = network_args
    parameters['sumo_args'] = sumo_args
    parameters['env_args'] = env_args
    parameters['ql_args'] = ql_args
    parameters['programs'] = programs

    filename = \
            f"{env.network.name}.params.json"

    params_path = os.path.join(path, filename)
    with open(params_path, 'w') as f:
        json.dump(parameters, f)

    # Run experiment.
    print('Running experiment...')

    info_dict = exp.run(
        int(flags.time / flags.step)
    )

    # Save train log.
    filename = \
            f"{env.network.name}.train.json"

    info_path = os.path.join(path, filename)
    with open(info_path, 'w') as f:
        json.dump(info_dict, f)

    return path

if __name__ == '__main__':
    main()
