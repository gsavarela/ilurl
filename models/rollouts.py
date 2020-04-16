'''Evaluation: evaluates a batch of experiments.

   * loads multi experiment files.
   * verify parameters -- if they're compatible proceed
   * for each experiment
        loads all Q-tables, from that experiment.
        filter Q-tables from S to S steps.
        for each table runs R rollouts (defaults 1).
   
'''

__author__ = 'Guilherme Varela'
__date__ = '2020-04-07'
from os import environ
from pathlib import Path
import json
import re
import random
from copy import deepcopy

import configargparse
import dill
import numpy as np

from flow.core.params import SumoParams, EnvParams
from ilurl.core.params import QLParams
from ilurl.core.experiment import Experiment
import ilurl.core.ql.dpq as ql
from ilurl.envs.base import TrafficLightEnv
from ilurl.networks.base import Network
#TODO: move this into network
from ilurl.loaders.nets import get_tls_custom

ILURL_HOME = environ['ILURL_HOME']

Q_FINDER_PROG = re.compile(r'Q.1-(\d+)')

def search_Q(x):
    found = Q_FINDER_PROG.search(x)
    if found is None:
        raise ValueError('Q-table rollout number not found')
    res, = found.groups()
    return int(res)

def get_arguments(config_file_path):
    if config_file_path is None:
        config_file_path = []

    parser = configargparse.ArgumentParser(
        default_config_files=config_file_path,
        description="""
            This script performs a single rollout from a Q table
        """
    )

    parser.add_argument('--rollout-path', '-q', dest='rollout_path',
                        type=str, nargs='?',
                        help='''The path Q.1-xxx.pickle files''')

    parser.add_argument('--cycles', '-c', dest='cycles', type=int,
                        default=100, nargs='?',
                        help='Number of cycles for a single rollout of a Q-table.')

    parser.add_argument('--emission', '-e', dest='emission', type=str2bool,
                        default=False, nargs='?',
                        help='Enabled will perform saves')

    parser.add('--rollout-seed', '-d', dest='seed', type=int,
                        default=None, nargs='?',
                        help='''Sets seed value for both rl agent and Sumo.
                               `None` for rl agent defaults to RandomState() 
                               `None` for Sumo defaults to a fixed but arbitrary seed''')

    parser.add_argument('--num-rollouts', '-r', dest='num_rollouts',
                        type=int, default=1, nargs='?',
                        help='''Number of repetitions for each table''')


    parser.add_argument('--switch', '-w', dest='switch', type=str2bool,
                        default=False, nargs='?',
                        help=
                        '''Rollout demand distribution can be either
                        `lane` or `switch` defaults to lane''')

    return parser.parse_args()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise configparser.ArgumentTypeError('Boolean value expected.')


def evaluate(env_params, sim_params, programs,
             agent, network, horizon, ex_id, roll_id, qtb):
    """Evaluate

    Params:
    -------
        * env_params: ilurl.core.params.EnvParams
            objects parameters

        * sim_params:  ilurl.core.params.SumoParams
            objects parameters

        * programs: dict<string, dict<int, list<int>>
            keys: junction_id, action_id
            values: list of durations

        * agent: ilurl.dpq.MAIQ
            tabular Q-learning agent

        * network: ilurl.networks.Network
            object representing the network

        * horizon: int
            number of steps

        * qtb:  dict<string, dict>
            keys: q-table id, values: number of steps

    Returns:
    --------
        * info: dict
        evaluation metrics for experiment

    """
    if sim_params.emission_path:
        dir_path = sim_params.emission_path
        # old emission pattern: network.name-emission.xml
        # new emission pattern: network.name.ex_id-emission.xml
        network = deepcopy(network)
        network.name = f'{network.name}.{roll_id}'
    else:
        dir_path = None


    env1 = TrafficLightEnv(
        env_params,
        sim_params,
        agent,
        network,
        TLS_programs=programs
    )
    if qtb is not None:
        env1.Q = qtb

    env1.stop = True
    exp = Experiment(
        env1,
        dir_path=dir_path,
        train=False,
    )
    result = exp.run(horizon)
    result['id'] = ex_id
    result['discount'] = agent.ql_params.gamma
    if sim_params.seed:
        result['seed'] = [sim_params.seed]
    result['rollouts'] = [roll_id]
    return result


def roll(config_file_path=None):
    args = get_arguments(config_file_path)
    rollout_path = Path(args.rollout_path)

    # rollout_number = args.rollout_number
    x = 'w' if args.switch else 'l'
    cycles = args.cycles
    with rollout_path.open('rb') as f:
        qtb = dill.load(f)
    if qtb is None:
        raise ValueError('Q is None')
    rollout_number = search_Q(str(rollout_path))

    pattern = '*.params.json'
    params = None
    for params_path in rollout_path.parent.glob(pattern):
        with params_path.open('r') as f:
            params = json.load(f)
        break   # There should be only one match
    if params is None:
        raise ValueError('params is None')

    # TODO: test ground truth
    params['sumo_args']['render'] = False
    if args.emission:
        params['sumo_args']['emission_path'] = batch_dir
    else:
        if 'emission_path' in params['sumo_args']:
            del params['sumo_args']['emission_path']

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        params['sumo_args']['seed'] = args.seed


    # Load cycle time and TLS programs.
    ex_id = rollout_path.parts[-2]
    network = Network(**params['network_args'])
    cycle_time, programs = get_tls_custom(network.network_id)
    ql_params = QLParams(**params['ql_args'])

    cls_agent = getattr(ql, ql_params.agent_id)
    agent = cls_agent(ql_params)
    env_params = EnvParams(**params['env_args'])
    sim_params = SumoParams(**params['sumo_args'])

    horizon = int((cycle_time * cycles) / sim_params.sim_step)

    info = evaluate(env_params, sim_params, programs,
                    agent, network, horizon, ex_id, rollout_number, qtb)

    info['horizon'] = horizon
    return info

if __name__ == '__main__':
    roll()
