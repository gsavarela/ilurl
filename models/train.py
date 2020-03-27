"""Provides baseline for networks"""
__author__ = 'Guilherme Varela'
__date__ = '2020-01-08'

import os
import json
import argparse

from flow.core.params import SumoParams, EnvParams
from flow.envs.ring.accel import ADDITIONAL_ENV_PARAMS

from ilurl.envs.base import TrafficLightEnv

from ilurl.core.ql.dpq import DPQ, MAIQ
from ilurl.core.params import QLParams
from ilurl.core.experiment import Experiment

from ilurl.networks.base import Network

ILURL_HOME = os.environ['ILURL_HOME']

EMISSION_PATH = \
    f'{ILURL_HOME}/data/emissions/'

NETWORKS_PATH = \
    f'{ILURL_HOME}/data/networks/'

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
                        dest='save_agent', type=str2bool,
                        default=False, nargs='?',
                        help='Whether to save RL-agent parameters throughout training')

    parser.add_argument('--experiment-save-agent-interval',
                        dest='save_agent_interval', type=int, default=100,
                        nargs='?',
                        help='[ONLY APPLIES IF --experiment-save-agent is TRUE] \
                        Save agent interval (in agent update steps)')

    parser.add_argument('--sumo-render', '-r', dest='render', type=str2bool,
                        default=False, nargs='?',
                        help='Renders the simulation')

    parser.add_argument('--sumo-step', '-s',
                        dest='step', type=float, default=1, nargs='?',
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
    print('\tExperiment log info: {0}'.format(args.log_info))
    print('\tExperiment log info interval: {0}'.format(args.log_info_interval))
    print('\tExperiment save RL agent: {0}'.format(args.save_agent))
    print('\tExperiment save RL agent interval: {0}'.format(args.save_agent_interval))

    print('\tSUMO render: {0}'.format(args.render))
    print('\tSUMO emission: {0}'.format(args.emission))
    print('\tSUMO step: {0}'.format(args.step))

    print('\tTLS inflows switch: {0}\n'.format(args.switch))


def tls_configs(network_name):
    """

    Loads TLS settings (cycle time and programs)
    from tls_config.json file.

    Parameters
    ----------
    network_name : string
        network id

    Return
    ----------
    cycle_time: int
        the cycle time for the TLS system

    programs: dict
        the programs (timings) for the TLS system
        defines the actions that the agent can pick
    
    """
    tls_config_file = '{0}/{1}/tls_config.json'.format(
                    NETWORKS_PATH, network_name)

    if os.path.isfile(tls_config_file):

        with open(tls_config_file, 'r') as f:
            tls_config = json.load(f)

        if 'cycle_time' not in tls_config:
            raise KeyError(
                f'Missing `cycle_time` key in tls_config.json')

        # Setup cycle time.
        cycle_time = tls_config['cycle_time']

        # Setup programs.
        programs = {}
        for tls_id in network.tls_ids:

            if tls_id not in tls_config.keys():
                raise KeyError(
                f'Missing timings for id {tls_id} in tls_config.json.')

            # TODO: check timings correction.

            # Setup actions (programs) for given TLS.
            programs[tls_id] = {int(action): tls_config[tls_id][action]
                                    for action in tls_config[tls_id].keys()}

    else:
        raise FileNotFoundError("tls_config.json file not provided "
            "for network {0}.".format(network.network_id))

    return cycle_time, programs


if __name__ == '__main__':

    args = get_arguments()
    print_arguments(args)

    inflows_type = 'switch' if args.switch else 'lane'
    network_args = {
        'network_id': args.network,
        'horizon': args.time,
        'demand_type': inflows_type,
        'insertion_probability': 0.2,
    }
    network = Network(**network_args)

    # Create directory to store data.
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

    # Load cycle time and TLS programs.
    cycle_time, programs = tls_configs(args.network)

    additional_params = {}
    additional_params.update(ADDITIONAL_ENV_PARAMS)
    additional_params['target_velocity'] = 20
    additional_params['cycle_time'] = cycle_time
    env_args = {
        'evaluate': True,
        'additional_params': additional_params
    }
    env_params = EnvParams(**env_args)

    # Agent.

    phases_per_tls = [len(network.phases[t]) for t in network.tls_ids]

    # Assumes all agents have the same number of actions.
    num_actions = len(programs[network.tls_ids[0]])

    ql_args = {
                'epsilon': 0.10,
                'alpha': 0.50,
                'states': ('speed', 'count'),
                'rewards': {'type': 'target_velocity',
                         'costs': None},
                'phases_per_traffic_light': phases_per_tls,
                'num_actions': num_actions,
                'choice_type': 'eps-greedy',
                'category_counts': [8.56, 13.00],
                'category_speeds': [2.28, 5.50]
    }
    ql_params = QLParams(**ql_args)


    #  QL_agent = DPQ(ql_params)
    QL_agent =  MAIQ(ql_params)

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
                     log_info=args.log_info,
                     log_info_interval=args.log_info_interval,
                     save_agent=args.save_agent,
                     save_agent_interval=args.save_agent_interval
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
        int(args.time / args.step)
    )

    # Save train log.
    filename = \
            f"{env.network.name}.train.json"

    info_path = os.path.join(path, filename)
    with open(info_path, 'w') as f:
        json.dump(info_dict, f)
