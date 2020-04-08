'''This scripts handles q-table manipulation'''
__author__ = 'Guilherme Varela'
__date__ = '2020-04-07'


from collections import defaultdict, OrderedDict
import re

import dill

TIMESTAMP_PROG = re.compile(r'[0-9]{8}\-[0-9]{8,}\.[0-9]+')

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

