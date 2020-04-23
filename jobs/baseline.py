"""Baseline:

    * uses train script for setting up experiment
    * has to have a command line `tl_type`: `actuated`, `static`


TODO: Include config for evaluation

"""
from pathlib import Path
from datetime import datetime
import sys
from os import environ
import json
import tempfile
import multiprocessing as mp
import time
import argparse

import configparser

from models.train import main as baseline
from ilurl.utils.decorators import processable, benchmarked
from ilurl.utils import str2bool

ILURL_PATH = Path(environ['ILURL_HOME'])

CONFIG_PATH = ILURL_PATH / 'config'

LOCK = mp.Lock()


def delay_baseline(*args, **kwargs):
    """delays execution by 1 sec.

        Parameters:
        -----------
        * fnc: function
            An anonymous function decorated by the user

        Returns:
        -------
        * fnc : function
            An anonymous function to be executed 1 sec. after
            calling
    """
    LOCK.acquire()
    try:
        time.sleep(1)
    finally:
        LOCK.release()
    return baseline(*args, **kwargs)


def get_arguments():
    parser = argparse.ArgumentParser(
        description="""
            User must provide the tls control type
        """
    )

    # TODO: validate against existing networks
    parser.add_argument('tls_type', type=str, nargs='?',
                        choices=('actuated', 'static', 'random'),  
                         help='Deterministic control type')
    flags = parser.parse_args()
    # BEWARE: clears all command line arguments
    sys.argv = [sys.argv[0]]
    return flags

def baseline_batch():
    flags = get_arguments()

    # Read script arguments from run.config file.
    run_config = configparser.ConfigParser()
    run_path = CONFIG_PATH / 'run.config'
    run_config.read(run_path)

    num_processors = int(run_config.get('run_args', 'num_processors'))
    num_runs = int(run_config.get('run_args', 'num_runs'))
    # TODO: seeds should be for evaluation
    seeds = json.loads(run_config.get("run_args", "train_seeds"))

    if len(seeds) != num_runs:
        raise configparser.Error('Number of seeds in run.config `seeds`'
                        ' must match the number of runs (`num_runs`) argument.')

    print('Arguments (baseline.py):')
    print('\tNumber of runs: {0}'.format(num_runs))
    print('\tNumber of processors: {0}'.format(num_processors))
    print('\tTrain seeds: {0}\n'.format(seeds))

    # Assess total number of processors.
    processors_total = mp.cpu_count()
    print(f'Total number of processors available: {processors_total}\n')

    # Adjust number of processors.
    if num_processors > processors_total:
        num_processors = processors_total
        print(f'Number of processors downgraded to {num_processors}\n')

    # Read train.py arguments from train.config file.
    # TODO: include an evaluation config
    baseline_config = configparser.ConfigParser()
    baseline_path = CONFIG_PATH / 'train.config'
    baseline_config.read(str(baseline_path))
    
    # Setup sumo-tls-type
    baseline_config.set('train_args', 'sumo-tls-type', flags.tls_type)
    baseline_config.set('train_args', 'experiment-save-agent', str(False))

    # Override train configurations with test's
    test_config = configparser.ConfigParser()
    test_path = CONFIG_PATH / 'test.config'
    test_config.read(test_path.as_posix())

    # is there another way
    horizon = int(test_config.get('test_args', 'cycles')) * 90
    emission = str2bool(test_config.get('test_args', 'emission'))
    switch = str2bool(test_config.get('test_args', 'switch'))
    seed_delta = int(test_config.get('test_args', 'seed_delta'))

    baseline_config.set('train_args', 'experiment-time', str(horizon))
    baseline_config.set('train_args', 'inflows-switch', str(switch))
    baseline_config.set('train_args', 'sumo-emission', str(emission))
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a config file for each train.py
        # with the respective seed. These config
        # files are stored in a temporary directory.
        tmp_path = Path(tmp_dir)
        baseline_configs = []
        for seed in seeds:

            cfg_path = tmp_path / f'{flags.tls_type}-{seed}.config'
            baseline_configs.append(cfg_path)

            # Setup train seed.
            baseline_config.set("train_args", "experiment-seed", str(seed + seed_delta))
            
            # Write temporary train config file.
            with cfg_path.open('w') as ft:
                baseline_config.write(ft)
        # Run.
        # TODO: option without pooling not working. why?
        # rvs: directories' names holding experiment data
        if num_processors > 1:
            pool = mp.Pool(num_processors)
            rvs = pool.map(delay_baseline, [[cfg] for cfg in baseline_configs])
            pool.close()
        else:
            rvs = []
            for cfg in baseline_configs:
                rvs.append(delay_baseline([cfg]))
        # Create a directory and move newly created files
        paths = [Path(f) for f in rvs]
        commons = [p.parent for p in paths]
        if len(set(commons)) > 1:
            raise ValueError(f'Directories {set(commons)} must have the same root')
        dirpath = commons[0]
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S.%f')
        batchpath = dirpath / timestamp
        if not batchpath.exists():
            batchpath.mkdir()

        # Move files
        for src in paths:
            dst = batchpath / src.parts[-1]
            src.replace(dst)
    sys.stdout.write(str(batchpath))
    return str(batchpath)

@processable
def baseline_job():
    return baseline_batch()

if __name__ == '__main__':
    # enable this in order to have a nice textual ouput
    baseline_batch()
    # baseline_job()

