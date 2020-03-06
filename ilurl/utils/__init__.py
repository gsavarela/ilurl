import pdb
import re

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

