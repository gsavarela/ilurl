'''Evaluation script for smart grid network

 TODO:
 ----
    * add directory to read rou.xml files
'''

__author__ = 'Guilherme Varela'
__date__ = '2019-09-24'

import multiprocessing as mp
from copy import deepcopy
from collections import defaultdict, OrderedDict
import re
import os
from glob import glob
import json
import argparse
import pdb

import dill

from ilurl.core.experiment import Experiment
from ilurl.envs.base import TrafficLightQLEnv
from ilurl.networks.base import Network

# TODO: Generalize for any parameter
ILURL_HOME = os.environ['ILURL_HOME']

EMISSION_PATH = \
    f'{ILURL_HOME}/data/emissions/'

NET_PATH = \
    f'{ILURL_HOME}/data/networks/'

TIMESTAMP_PROG = re.compile(r'[0-9]{8}\-[0-9]{8,}\.[0-9]+')


def get_arguments():
    parser = argparse.ArgumentParser(
        description="""
            This script runs a traffic light simulation based on saved
            Q-tables.
        """
    )

    # TODO: validate against existing networks
    parser.add_argument('dir_pickle', type=str, nargs='?',
                        default=f'{EMISSION_PATH}', help='Path to pickle')


    parser.add_argument('--cycles', '-c', dest='cycles', type=int,
                        default=100, nargs='?',
                        help=
                        '''Number of cycles for a single rollout of a
                        Q-table. This argument if provided takes the 
                        precedence over time parameter''')

    parser.add_argument('--num_processors', '-p', dest='num_processors',
                        type=int, default=1, nargs='?',
                        help='Number of synchronous num_processors')

    parser.add_argument('--render', '-r', dest='render', type=str2bool,
                        default=False, nargs='?',
                        help='Renders the simulation') 

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
    env1.Q = qtb
    env1.stop = True
    eval = Experiment(
        env1,
        dir_path=None,
        train=False,
    )
    result = eval.run(1, horizon)
    return result


def concat(evaluations, horizon):
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
    result = defaultdict(list)
    for evl in evaluations:
        id = evl.pop('id')
        if 'id' not in result:
            result['id'] = id
        elif result['id'] != id:
            raise ValueError(
                'Tried to concatenate evaluations from different experiments')
        for k, v in evl.items():
            if k != 'id':
                if isinstance(v, list):
                    result[k] = result[k] + v
                else:
                    result[k].append(v)
        result['horizon'] = horizon
    return result


def parse_all(paths):
    parsed_dict = defaultdict(dict)
    for path in paths:
        data = parse(path)
        # store argument condition
        if data is not None:
            if len(data) == 4:
                key = data[1:]
                key1 = None  # nested key
            elif len(data) == 6:
                key = tuple(list(data[1:3]) + [data[-1]])
                key1 = data[3:5]  # nested key
            else:
                raise ValueError(f'{data} not recognized')

            if key1 is None:
                parsed_dict[key]['env'] = path
            else:
                parsed_dict[key][key1] = path
    return parsed_dict


def parse(x):
    """Splits experiment string, parsing contents

    Params:
    -------
        * x: string
        Representing a source path,
        if x is not valid returns None,

        if x is an agent returns:
            source_dir, network_id, timestamp, ext

        if x is a Q-table returns:
            source_dir, network_id, timestamp, iter, cycle, ext

    Returns:
    -------
        * source_dir: string
            string representing source directory

        * network_id: string
            the network the experiment

        * timestamp: string
            representation of datetime the experimentation began

        * iter: integer
            iteration representing a history, i.e rollout number,
            if string encodes a Q-table

        * cycle: integer
            number of cycles within the history/rollout

    Usage:
    ------
    > x =  \
        'data/experiments/0x04/6030/' +
        'intersection_20200227-1131341582803094.6042109' +
        '.Q.1-8.pickle'
    > y = parse(x)
    > y[0]
    > 'data/experiments/0x04/6030/'
    > y[1:]
    > ('intersection', '20200227-1131341582803094.6042109', 1, 8, 'pickle')
    """

    *dirs, name = x.split('/')
    if not dirs:
        return None
    source_dir = '/'.join(dirs)

    result = TIMESTAMP_PROG.search(name)
    if result is None:
        return None
    timestamp = result.group()
    # everything that comes before
    start, finish = result.span()
    network_id = name[:start - 1]   # remove underscore
    ext = name[finish + 1:]
    if len(ext.split('.')) == 1:
        return source_dir, network_id, timestamp, ext
    else:
        q, code, ext = ext.split('.')

    if q != 'Q':
        return None

    iter, cycles = [int(c) for c in code.split('-')]

    return source_dir, network_id, timestamp, iter, cycles, ext


def sort_all(data):
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
    for exid, qtbs in data.items():
        result[exid]['env'] = qtbs['env']
        tmp = {k: v for k, v in qtbs.items() if k != 'env'}
        tmp = sort_tables(tmp.items())
        for trial, path in tmp.items():
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

    for exid, keys_paths in data.items():
        for key, path in keys_paths.items():
            # traffic light object
            if key == 'env':
                result[exid]['env'] = TrafficLightQLEnv.load(path)
            else:
                with open(path, 'rb') as f:
                    result[exid][key] = dill.load(f)
    return result


if __name__ == '__main__':
    args = get_arguments()

    dir_pickle = args.dir_pickle
    x = 'w' if args.switch else 'l'
    render = args.render
    cycles = args.cycles
    num_processors = args.num_processors

    if num_processors >= mp.cpu_count():
        num_processors = mp.cpu_count()-1
        print(f'Number of num_processors downgraded to {num_processors}')

    paths = glob(f"{dir_pickle}/*.pickle")
    if not any(paths):
        raise Exception("Environment pickles must be saved on root")

    # process data
    # converts paths into dictionary
    data = parse_all(paths)
    # sort experiments by instances and cycles
    data = sort_all(data)
    # converts paths into objects
    data = load_all(data)
    num_experiments = len(data)
    # one element will be the environment
    num_rollouts = sum([len(d) - 1 for d in data.values()])
    i = 1
    results = []
    for exid, qtbs in data.items():
        print(f"""
                experiment:\t{i}/{num_experiments}
                network_id:\t{exid[0]}
                timestamp:\t{exid[1]}
                rollouts:\t{len(qtbs) - 1}/{num_rollouts}
              """)

        env = qtbs.pop('env')
        cycle_time = getattr(env, 'cycle_time', 1)
        env_params = env.env_params
        sim_params = env.sim_params
        sim_params.render = render

        horizon = int((cycle_time * cycles) / sim_params.sim_step)
        ql_params = env.ql_params
        network = env.network

        def fn(qtb):
            ret = evaluate(
                env_params,
                sim_params,
                ql_params,
                network,
                horizon,
                qtb)
            return ret

        if num_processors > 1:
            pool = mp.Pool(num_processors)
            results = pool.map(fn, qtbs.values())
            pool.close()
        else:
            results = [fn(qtb) for qtb in qtbs.values()]
        info = concat(results, [horizon] * len(results))
        filename = '_'.join(exid[:2])
        file_path = f'{dir_pickle}/{filename}.eval.json'
        with open(file_path, 'w') as f:
            json.dump(info, f)
