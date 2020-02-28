'''Evaluation script for smart grid network

 TODO:
 ----
    * add directory to read rou.xml files
'''

__author__ = 'Guilherme Varela'
__date__ = '2019-09-24'

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

    parser.add_argument('--time', '-t', dest='time', type=int,
                        default=None, nargs='?',
                        help=
                        '''Simulation\'s real world time in seconds
                        for a single rollout of a Q-table. Is ignored
                        unless `cycles` parameter is not provided.''')

    parser.add_argument('--switch', '-w', dest='switch', type=str2bool,
                        default=False, nargs='?',
                        help=
                        '''Rollout demand distribution can be either
                        `lane` or `switch` defaults to lane''')

    parser.add_argument('--render', '-r', dest='render', type=str2bool,
                        default=False, nargs='?',
                        help='Renders the simulation')

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


def evaluate(env, code, q_tables, horizon, path, render=False):
    # load all networks with config -- else build
    results = []
    for i, qtb in enumerate(q_tables):
        #TODO: save network objects rather then routes
        env1 = TrafficLightQLEnv(env.env_params,
                                 env.sim_params,
                                 env.ql_params,
                                 env.network)
        env1.sim_params.render = render
        env1.stop = True
        num_iterations = 1
        eval = Experiment(
            env1,
            dir_path=path,
            train=False,
            q_tables=q_tables
        )

        print(f"Running evaluation {i + 1}")
        result = eval.run(num_iterations, horizon)
        results.append(result)

    # TODO: concatenate results along axis=1
    return results


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
                parsed_dict[key]['source'] = path
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


def sort_tables(qtbs):
    qtbs = sorted(qtbs, key=lambda x: x[0][1])
    qtbs = sorted(qtbs, key=lambda x: x[0][0])
    return OrderedDict(qtbs)


if __name__ == '__main__':
    args = get_arguments()

    dir_pickle = args.dir_pickle
    x = 'w' if args.switch else 'l'
    render = args.render
    time = args.time
    cycles = args.cycles

    paths = glob(f"{dir_pickle}/*.pickle")
    if not any(paths):
        raise Exception("Environment pickles must be saved on root")

    data = parse_all(paths)
    for exid, qtbs in data.items():
        #  for each experiment perform rowouts
        #  lexografical sort
        print(f"""
                network_id:\t{exid[0]}
                timestamp:\t{exid[1]}
                rollouts:\t{len(qtbs)}
              """)
        env = TrafficLightQLEnv.load(qtbs.pop('source'))
        qtbs = sort_tables(qtbs.items())
        results = \
            evaluate(env, x, qtbs, cycles, dir_pickle, render=render)

    # TODO: concatenate axis=1
    # TODO: save
