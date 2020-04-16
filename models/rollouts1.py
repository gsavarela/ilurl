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
from datetime import datetime
import multiprocessing as mp
from collections import defaultdict
import os
from pathlib import Path
from glob import glob
import json
import re
import random

# import argparse
import configargparse
from copy import deepcopy
import dill
import numpy as np

from flow.core.params import SumoParams, EnvParams
from ilurl.core.params import QLParams
from ilurl.core.experiment import Experiment
import ilurl.core.ql.dpq as ql
from ilurl.envs.base import TrafficLightEnv
from ilurl.networks.base import Network
from ilurl.utils.q_tables import (parse, parse_all, sort_tables, sort_all,
                                  filter_tables, load_all)
#TODO: move this into network
from ilurl.loaders.nets import get_tls_custom

ILURL_HOME = os.environ['ILURL_HOME']

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
    # parser.add_argument('--limit', '-l', dest='limit',
    #                     type=int, default=500, nargs='?',
    #                     help='Use only Q-tables generated until `-l` cycle')

    # parser.add_argument('--num-processors', '-p', dest='num_processors',
    #                     type=int, default=1, nargs='?',
    #                     help='Number of synchronous num_processors')

    parser.add_argument('--num-rollouts', '-r', dest='num_rollouts',
                        type=int, default=1, nargs='?',
                        help='''Number of repetitions for each table''')

    # parser.add_argument('--sample', '-s', dest='skip',
    #                     type=int, default=500, nargs='?',
    #                     help='''Sample every `sample` experiments.''')

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
        raise argparse.ArgumentTypeError('Boolean value expected.')


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
        print(dir_path, network.name)
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
    result['rollout'] = rollout_number
    return result


# def concat(evaluations):
#     """Receives an experiments' json and merges it's contents
# 
#     Params:
#     -------
#         * evaluations: list
#         list of rollout evaluations
# 
#     Returns:
#     --------
#         * result: dict
#         where `id` key certifies that experiments are the same
#               `list` params are united
#               `numeric` params are appended
# 
#     """
#     result = defaultdict(list)
#     for id_qtb in evaluations:
#         qid, qtb = id_qtb
#         exid = qtb.pop('id')
#         # can either be a rollout from the prev
#         # exid or a new experiment
#         if exid not in result['id']:
#             result['id'].append(exid)
# 
#         ex_idx = result['id'].index(exid)
#         for k, v in qtb.items():
#             append = isinstance(v, list) or isinstance(v, dict)
#             # check if integer fields match
#             # such as cycle, save_step, etc
#             if not append:
#                 if k in result:
#                     if result[k] != v:
#                         raise ValueError(
#                             f'key:\t{k}\t{result[k]} and {v} should match'
#                         )
#                 else:
#                     result[k] = v
#             else:
#                 if ex_idx == len(result[k]):
#                     result[k].append(defaultdict(list))
#                 result[k][ex_idx][qid[1]].append(v)
#     return result


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
    # num_processors = args.num_processors
    # num_rollouts = args.num_rollouts
    # skip = args.skip
    # limit = args.limit

    # if num_processors >= mp.cpu_count():
    #     num_processors = mp.cpu_count() - 1
    #     print(f'Number of processors downgraded to {num_processors}')

    # process data: converts paths into dictionary
    # get train parameters
    # inner dir must contain experiments
    pattern = '*.params.json'
    # a mapping from experiments to q-tables
    # qtb2path = {}
    params = None
    for params_path in rollout_path.parent.glob(pattern):
        # rel_path = os.path.relpath(path)
        with params_path.open('r') as f:
            params = json.load(f)
        break   # There should be only one match
    if params is None:
        raise ValueError('params is None')

    # TODO: test ground truth
    # Force render False
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
        # Force save xml
        # exp_dir = os.path.dirname(rel_path)


    # data = None
    # pattern = '*.train.json'
    # for train_path in rollout_path.parent.glob(pattern):
    #     # get train data
    #     with train_path.open('r') as f:
    #         data = json.load(f)
    #     break   # There should be only one match
    # if data is None:
    #     raise ValueError('data is None')

    # build Q-tables pattern
    # pattern = f'{exp_dir}/*.Q.*'
    # qtb2path.update(
    #     parse_all(glob(pattern))
    # )
    # remove paths
    # qtb2path = filter_tables(qtb2path, skip, limit)
    # sort experiments by instances and cycles
    # qtb2path = sort_all(qtb2path)
    # converts paths into objects
    # qtb2obj = load_all(qtb2path)

    # results = []
    # roll_per_ex = [len(q) * num_rollouts for q in qtb2obj.values()]

    # if num_processors > min(roll_per_ex):
    #     print(f'Number of processors downgraded to {min(roll_per_ex)}')
    #     num_processors = min(roll_per_ex)

    # roll_total = sum(roll_per_ex)
    # roll_counter = 0
    # ex_counter = 0
    #for exid, qtbs in qtb2obj.items():
    # ex_id = '_'.join(exid[:-1])
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

    # def fn(id_qtb):
    #     qid, qtb = id_qtb
    #     ret = evaluate(env_params, sim_params, programs,
    #                    agent, network, horizon, ex_id, qid[-1], qtb)
    #     return (qid, ret)

    # _qtbs = list(qtbs.items()) * num_rollouts

    # ex_counter += 1
    # roll_counter += len(_qtbs)
    # print(f"""
    #         experiment:\t{ex_counter}/{len(qtb2path)}
    #         network_id:\t{exid[0]}
    #         timestamp:\t{exid[1]}
    #         rollouts:\t{roll_counter}/{roll_total}
    #       """)

    # if num_processors > 1:
    #     pool = mp.Pool(num_processors)
    #     results += pool.map(fn, _qtbs)
    #     pool.close()
    # else:
    info = evaluate(env_params, sim_params, programs,
                    agent, network, horizon, ex_id, rollout_number, qtb)

    # results = [fn(qtb) for qtb in _qtbs]
    # info = concat([ret])

    # keys = list(qtbs.keys())
    # timestamp = datetime.now().strftime('%Y%m%d%H%M%S.%f')
    # filename = f'{network.network_id}_{timestamp}'
    # info['horizon'] = horizon
    # info['rollouts'] = [k[1] for k in keys]
    # info['num_rollouts'] = num_rollouts
    # info['limit'] = limit
    # info['skip'] = skip
    # info['processed_at'] = timestamp
    # file_path = f'{batch_dir}/{filename}.{x}.eval.info.json'

    # with open(file_path, 'w') as f:
    #    json.dump(info, f)
    # print(file_path)
    return info

if __name__ == '__main__':
    roll()
