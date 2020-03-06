'''Evaluation: evaluates 1 experiment

   * loads one experiment file.
   * loads all Q-tables, from that experiment.
   * filter Q-tables from S to S steps.
   * for each table runs K experiments.
   
'''

__author__ = 'Guilherme Varela'
__date__ = '2020-03-04'

import pdb
from datetime import datetime
import multiprocessing as mp
from collections import defaultdict, OrderedDict
import re
import os
from glob import glob
import json
import argparse

import dill

from ilurl.core.experiment import Experiment
from ilurl.envs.base import TrafficLightQLEnv
from ilurl.networks.base import Network
from ilurl.utils import parse

def get_arguments():
    parser = argparse.ArgumentParser(
        description="""
            This script runs a traffic light simulation based on saved
            Q-tables.
        """
    )

    # TODO: validate against existing networks
    parser.add_argument('pickle_path', type=str, nargs='?',
                         help='Path to an environment saved as pickle')


    parser.add_argument('--cycles', '-c', dest='cycles', type=int,
                        default=100, nargs='?',
                        help='Number of cycles for a single rollout of a Q-table.')


    parser.add_argument('--limit', '-l', dest='limit',
                        type=int, default=500, nargs='?',
                        help='Use only Q-tables generated until `-l` cycle')

    parser.add_argument('--num_processors', '-p', dest='num_processors',
                        type=int, default=1, nargs='?',
                        help='Number of synchronous num_processors')

    parser.add_argument('--num_rollouts', '-r', dest='num_rollouts',
                        type=int, default=12, nargs='?',
                        help='''Number of repetitions for each table''')

    parser.add_argument('--sample', '-s', dest='skip',
                        type=int, default=500, nargs='?',
                        help='''Sample every `sample` experiments.''')

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


def evaluate(env_params, sim_params, ql_params, network, horizon, qtb):
    """Evaluate

    Params:
    -------
        * env_params: ilurl.core.params.EnvParams
            objects parameters

        * sim_params:  ilurl.core.params.SumoParams
            objects parameters

        * ql_params:  ilurl.core.params.QLParams
            objects parameters

        * network: ilurl.networks.Network 

    Returns:
    --------
        * info: dict
        evaluation metrics for experiment

    """
    env1 = TrafficLightQLEnv(
        env_params,
        sim_params,
        ql_params,
        network
    )
    if qtb is not None:
        env1.Q = qtb
    env1.stop = True
    eval = Experiment(
        env1,
        dir_path=None,
        train=False,
    )
    result = eval.run(1, horizon)
    return result


def concat(evaluations):
    """Receives an experiments' json and merges it's contents

    Params:
    -------
        * evaluations: list
        list of rollout evaluations

    Returns:
    --------
        * result: dict
        where `id` key certifies that experiments are the same
              `list` params are united
              `numeric` params are appended

    """
    result = {}
    for id_qtb in evaluations:
        qid, qtb = id_qtb
        id = qtb.pop('id')
        if 'id' not in result:
            result['id'] = id
        elif result['id'] != id:
            raise ValueError(
                'Tried to concatenate evaluations from different experiments')
        for k, v in qtb.items():
            if k != 'id':
                # TODO: check if integer fields match
                # such as cycle, save_step, etc
                if isinstance(v, list):
                    if k not in result:
                        result[k] = defaultdict(list)
                    result[k][qid[1]].append(v)
                else:
                    if k in result:
                        if result[k] != v:
                            raise ValueError(f'key:\t{k}\t{result[k]} and {v} should match')
                    else:
                        result[k] = v
    return result


def parse_all(paths):
    """Parse paths: splitting into environments and experiments.

    Params:
    -------
        * paths: list
            list of source paths pointing to either environment
            or experiment pickle files

    Returns:
    -------
        * env2path: dict
            dict with paths pointing to pickled env instances

        * qtb2path: dict of dicts
            dict with paths pointing to pickled cycles to Q-tables
            mappings

    """
    env2path = {}
    qtb2path = defaultdict(dict)
    for path in paths:
        nuple = parse(path)
        # store argument condition
        if nuple is not None:
            # this nuple encodes an env path
            if len(nuple) == 4:
                key = nuple[1:]
                env2path[key] = path
            elif len(nuple) == 6:
                # this nuple encodes a q-table
                key = tuple(list(nuple[1:3]) + [nuple[-1]])
                key1 = nuple[3:5]  # nested key

                qtb2path[key][key1] = path
            else:
                raise ValueError(f'{nuple} not recognized')

    return env2path, qtb2path

def sort_all(qtb2path):
    """Performs sort accross multiple experiments, within
    each experiment

    Params:
    ------
        * qtbs: dict
            keys are tuples (<iter_num>,<cycles_num>)
            values are Q-tables

    Returns:
    -------
        * OrderedDict

    """
    result = defaultdict(OrderedDict)
    for exid, qtbs in qtb2path.items():
        for trial, path in sort_tables(qtbs.items()).items():
            result[exid][trial] = path
    return result


def sort_tables(qtbs):
    """Sort Q-tables dictionary

    Params:
    ------
        * qtbs: dict
            keys are tuples (<iter_num>,<cycles_num>)
            values are Q-tables

    Returns:
    -------
        * OrderedDict

    """
    qtbs = sorted(qtbs, key=lambda x: x[0][1])
    qtbs = sorted(qtbs, key=lambda x: x[0][0])
    return OrderedDict(qtbs)


def filter_tables(qtbs2path, skip, limit):
    """Remove qids which are not multiple of skip


    Params:
    ------
    *   qtbs2path: dictionary of dictionary
            keys: <tuple> parsed experiment id
            keys: <tuple> parsed qtb id
            values: <string>

    *   skip: int
            keep multiples of cycles indicated by skip

    *   limit: int
            keep Q-tables trained up until limit

    Returns:
    -------
    *   qtbs2path: dictionary of dictionary
            possibly with some elements removed

    """
    def fn(x):
        return x[1] % skip == 0 and x[1] <= limit

    return {
            expid:
            {qid: qtb for qid, qtb in qtbs.items() if fn(qid)}
            for expid, qtbs in qtbs2path.items()
    }


def load_all(data):
    """Converts path variable into objects

    Params:
    ------
        * qtbs: dict
            keys are tuples (<iter_num>,<cycles_num>)
            values are Q-tables

    Returns:
    -------
        * OrderedDict

    """
    result = defaultdict(OrderedDict)
    for exid, path_or_dict in data.items():
        if isinstance(path_or_dict, str):
            # traffic light object
            result[exid] = TrafficLightQLEnv.load(path_or_dict)
        elif isinstance(path_or_dict, dict):
            
            # q-table
            for key, path in path_or_dict.items():
                with open(path, 'rb') as f:
                    result[exid][key] = dill.load(f)
        else:
            raise ValueError(
                f'path_or_dict must be str, dict or None -- got {type(path_or_dict)}')
    return result


if __name__ == '__main__':
    args = get_arguments()

    pickle_path = args.pickle_path
    pickle_dir = '/'.join(pickle_path.split('/')[:-1])
    x = 'w' if args.switch else 'l'
    cycles = args.cycles
    num_processors = args.num_processors
    num_rollouts = args.num_rollouts
    skip = args.skip
    limit = args.limit

    if num_processors >= mp.cpu_count():
        num_processors = mp.cpu_count() - 1
        print(f'Number of processors downgraded to {num_processors}')

    # process data: converts paths into dictionary
    env2path, _ = parse_all([pickle_path])
    # build Q-tables pattern
    prefix = '.'.join(pickle_path.split('.')[:2])
    pattern = f'{prefix}.Q.*'
    _, qtb2path = parse_all(glob(pattern))
    # remove paths
    qtb2path = filter_tables(qtb2path, skip, limit)
    # sort experiments by instances and cycles
    qtb2path = sort_all(qtb2path)
    # converts paths into objects
    env2obj = load_all(env2path)
    qtb2obj = load_all(qtb2path)
    num_experiments = len(env2obj)
    i = 1
    results = []
    for exid, qtbs in qtb2obj.items():

        env = env2obj[exid]
        cycle_time = getattr(env, 'cycle_time', 1)
        env_params = env.env_params
        sim_params = env.sim_params

        horizon = int((cycle_time * cycles) / sim_params.sim_step)
        ql_params = env.ql_params
        network = env.network

        def fn(id_qtb):
            qid, qtb = id_qtb
            ret = evaluate(env_params, sim_params, ql_params,
                           network, horizon, qtb)
            return (qid, ret)


        # rollouts x qtbs
        # it doest have the blank table
        if (1, 0) not in qtbs:
            _qtbs = [((1, 0), None)] * num_rollouts
        _qtbs += list(qtbs.items()) * num_rollouts

        print(f"""
                experiment:\t{i}/{num_experiments}
                network_id:\t{exid[0]}
                timestamp:\t{exid[1]}
                rollouts:\t{len(_qtbs)}
              """)
        if num_processors > 1:
            pool = mp.Pool(num_processors)
            results = pool.map(fn, _qtbs)
            pool.close()
        else:
            results = [fn(qtb) for qtb in _qtbs]
        info = concat(results)

        # add some metadata into it
        # TODO: generate tables Q0
        keys = list(qtbs.keys())
        if (1, 0) not in keys:
            keys = [(1, 0)] + keys
        info['horizon'] = horizon 
        info['rollouts'] = [k[1] for k in keys]
        info['num_rollouts'] = num_rollouts
        info['limit'] = limit
        info['skip'] = skip
        info['processed_at'] = \
            datetime.now().strftime('%Y-%m-%d%H%M%S.%f')
        filename = '_'.join(exid[:2])
        file_path = f'{pickle_dir}/{filename}.{x}.eval.info.json'
        with open(file_path, 'w') as f:
            json.dump(info, f)
        print(file_path)
        i += 1
