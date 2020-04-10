'''Evaluation: evaluates 1 experiment

   * loads one experiment file.
   * loads all Q-tables, from that experiment.
   * filter Q-tables from S to S steps.
   * for each table runs K experiments.
   
'''

__author__ = 'Guilherme Varela'
__date__ = '2020-03-04'

from datetime import datetime
import multiprocessing as mp
from collections import defaultdict, OrderedDict
import re
import os
from glob import glob
import json
import argparse

import dill

from flow.core.params import SumoParams, EnvParams
from ilurl.core.params import QLParams
from ilurl.core.experiment import Experiment
import ilurl.core.ql.dpq as ql

from ilurl.envs.base import TrafficLightEnv
from ilurl.networks.base import Network
from ilurl.utils import parse


ILURL_HOME = os.environ['ILURL_HOME']

NETWORKS_PATH = \
    f'{ILURL_HOME}/data/networks/'

def get_arguments():
    parser = argparse.ArgumentParser(
        description="""
            This script runs a traffic light simulation based on saved
            Q-tables.
        """
    )

    # TODO: validate against existing networks
    parser.add_argument('experiment_dir', type=str, nargs='?',
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


def evaluate(env_params, sim_params, programs, agent, network, horizon, qtb):
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
        dir_path=None,
        train=False,
    )
    result = exp.run(horizon)
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
        * qtb2path: dict of dicts
            dict with paths pointing to pickled cycles to Q-tables
            mappings

    """
    qtb2path = defaultdict(dict)
    for path in paths:
        nuple = parse(path)
        # store argument condition
        if nuple is not None:
            # this nuple encodes an env path
            if len(nuple) == 6:
                # this nuple encodes a q-table
                key = tuple(list(nuple[1:3]) + [nuple[-1]])
                key1 = nuple[3:5]  # nested key

                qtb2path[key][key1] = path
            else:
                raise ValueError(f'{nuple} not recognized')

    return qtb2path


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
            result[exid] = TrafficLightEnv.load(path_or_dict)
        elif isinstance(path_or_dict, dict):
            
            # q-table
            for key, path in path_or_dict.items():
                with open(path, 'rb') as f:
                    result[exid][key] = dill.load(f)
        else:
            raise ValueError(
                f'path_or_dict must be str, dict or None -- got {type(path_or_dict)}')
    return result


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
    exp_dir = args.experiment_dir
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
    # get train parameters
    pattern = f'{exp_dir}*.params.json'
    path = glob(pattern)[0]
    with open(path, 'r') as f:
        params = json.load(f)
    # Force render False
    params['sumo_args']['render'] = False


    # get train data
    pattern = f'{exp_dir}*.train.json'
    path = glob(pattern)[0]
    with open(path, 'r') as f:
        data = json.load(f)

    # build Q-tables pattern
    pattern = f'{exp_dir}*.Q.*'
    qtb2path = parse_all(glob(pattern))

    # remove paths
    qtb2path = filter_tables(qtb2path, skip, limit)
    # sort experiments by instances and cycles
    qtb2path = sort_all(qtb2path)
    # converts paths into objects
    qtb2obj = load_all(qtb2path)
    i = 1
    results = []
    for exid, qtbs in qtb2obj.items():

        filename = '_'.join(exid[:-1])
        # Load cycle time and TLS programs.
        network = Network(**params['network_args'])
        cycle_time, programs = tls_configs(network.network_id)
        ql_params = QLParams(**params['ql_args'])

        cls_agent = getattr(ql, ql_params.agent_id)
        agent = cls_agent(ql_params)
        env_params = EnvParams(**params['env_args'])
        sim_params = SumoParams(**params['sumo_args'])

        horizon = int((cycle_time * cycles) / sim_params.sim_step)

        def fn(id_qtb):
            qid, qtb = id_qtb
            ret = evaluate(env_params, sim_params, programs,
                           agent, network, horizon, qtb)
            return (qid, ret)

        _qtbs = list(qtbs.items()) * num_rollouts

        print(f"""
                experiment:\t{i}/{len(qtb2path)}
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

        keys = list(qtbs.keys())
        info['horizon'] = horizon 
        info['rollouts'] = [k[1] for k in keys]
        info['num_rollouts'] = num_rollouts
        info['limit'] = limit
        info['skip'] = skip
        info['processed_at'] = \
            datetime.now().strftime('%Y-%m-%d%H%M%S.%f')
        file_path = f'{exp_dir}{filename}.{x}.eval.info.json'
        with open(file_path, 'w') as f:
            json.dump(info, f)
        print(file_path)
        i += 1
