from pathlib import Path
from datetime import datetime
import sys
from os import environ
import json
import tempfile
import argparse
import multiprocessing as mp
import time
from collections import defaultdict

import configparser

from ilurl.utils.decorators import processable
from models.rollouts import roll

ILURL_HOME = environ['ILURL_HOME']

CONFIG_PATH = Path(f'{ILURL_HOME}/config/')


def get_arguments():
    parser = argparse.ArgumentParser(
        description="""
            This scripts runs recursevely every experiment on path. It must receve a batch path.
        """
    )
    parser.add_argument('batch_dir', type=str, nargs='?',
                        help='''A directory which it\'s subdirectories are experiments''')


    parsed = parser.parse_args()
    sys.argv = [sys.argv[0]]
    return parsed

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
    result = defaultdict(list)
    for qtb in evaluations:
        exid = qtb.pop('id')
        qid = qtb.get('rollouts', 0)[0]
        # can either be a rollout from the prev
        # exid or a new experiment
        if exid not in result['id']:
            result['id'].append(exid)

        ex_idx = result['id'].index(exid)
        for k, v in qtb.items():
            append = isinstance(v, list) or isinstance(v, dict)
            # check if integer fields match
            # such as cycle, save_step, etc
            if not append:
                if k in result:
                    if result[k] != v:
                        raise ValueError(
                            f'key:\t{k}\t{result[k]} and {v} should match'
                        )
                else:
                    result[k] = v
            else:
                if ex_idx == len(result[k]):
                    result[k].append(defaultdict(list))
                result[k][ex_idx][qid].append(v)
    return result


def rollout_batch(test=False):
    # Read script arguments from run.config file.
    args = get_arguments()
    # clear command line arguments after parsing
    batch_path = Path(args.batch_dir)
    pattern = '*Q*.pickle'
    # for test this should get only the last pickle
    rollout_paths = [rp for rp in batch_path.rglob(pattern)]

    if test:
        # filter x path using latest create time
        def fn(x):
            return x.stat().st_ctime
        # selects only the latest rollouts
        parent_paths = [rp for rp in batch_path.iterdir() if rp.is_dir()]
        rollout_paths = [max([rp for rp in pp.glob(pattern)], key=fn)
                         for pp in parent_paths]
    run_config = configparser.ConfigParser()
    run_config.read(str(CONFIG_PATH / 'run.config'))

    num_processors = int(run_config.get('run_args', 'num_processors'))
    num_runs = int(run_config.get('run_args', 'num_runs'))
    train_seeds = json.loads(run_config.get("run_args", "train_seeds"))

    if len(train_seeds) != num_runs:
        raise configparser.Error('Number of seeds in run.config `train_seeds`'
                        'must match the number of runs (`num_runs`) argument.')

    # Assess total number of processors.
    processors_total = mp.cpu_count()
    print(f'Total number of processors available: {processors_total}\n')

    # Adjust number of processors.
    if num_processors > processors_total:
        num_processors = processors_total
        print(f'Number of processors downgraded to {num_processors}\n')

    # Read train.py arguments from train.config file.
    rollouts_config = configparser.ConfigParser()
    rollouts_config.read(str(CONFIG_PATH / 'rollouts.config'))
    num_rollouts = int(rollouts_config.get('rollouts_args', 'num-rollouts'))

    if test:
        # merges test and rollouts
        test_config = configparser.ConfigParser()
        test_config.read(str(CONFIG_PATH / 'test.config'))
        num_rollouts = 1

        cycles = test_config.get('test_args', 'cycles')
        emission = test_config.get('test_args', 'emission')
        switch = test_config.get('test_args', 'switch')
        seed_delta = int(test_config.get('test_args', 'seed_delta'))

        # overwrite defaults
        rollouts_config.set('rollouts_args', 'cycles', cycles)
        rollouts_config.set('rollouts_args', 'emission', emission)
        rollouts_config.set('rollouts_args', 'switch', switch)
        rollouts_config.set('rollouts_args', 'num-rollouts', repr(num_rollouts))

        # alocates the S seeds among M rollouts
        custom_configs = []
        base_seed = max(train_seeds)
        for rn, rp in enumerate(rollout_paths):
            custom_configs.append((str(rp), base_seed + rn + seed_delta))
        token = 'test'
    else:
        # number of processes vs layouts
        # seeds must be different from training
        custom_configs = []
        for rn, rp in enumerate(rollout_paths):
            base_seed = max(train_seeds) + num_rollouts * rn
            for rr in range(num_rollouts):
                seed = base_seed + rr + 1
                custom_configs.append((str(rp), seed))
        token = 'rollouts'

    print(f'''
    \tArguments (jobs.{token}.py):
    \t----------------------------
    \tNumber of runs: {num_runs}
    \tNumber of processors: {num_processors}
    \tTrain seeds: {train_seeds}
    \tNum. rollout files: {len(rollout_paths)}
    \tNum. rollout repetitions: {num_rollouts}
    \tNum. rollout total: {len(rollout_paths) * num_rollouts}\n\n''')

    with tempfile.TemporaryDirectory() as f:

        tmp_path = Path(f)
        # Create a config file for each rollout
        # with the respective seed. These config
        # files are stored in a temporary directory.
        rollouts_cfg_paths = []
        cfg_key = "rollouts_args"
        for cfg in custom_configs:
            rollout_path, seed = cfg

            # Setup custom rollout settings
            rollouts_config.set(cfg_key, "rollout-path", str(rollout_path))
            rollouts_config.set(cfg_key, "rollout-seed", str(seed))
            
            # Write temporary train config file.
            cfg_path = tmp_path / f'rollouts-{seed}.config'
            rollouts_cfg_paths.append(str(cfg_path))
            with cfg_path.open('w') as fw:
                rollouts_config.write(fw)

        # rvs: directories' names holding experiment data
        if num_processors > 1:
            pool = mp.Pool(num_processors)
            rvs = pool.map(roll, [[cfg] for cfg in rollouts_cfg_paths])
            pool.close()
        else:
            rvs = []
            for cfg in rollouts_cfg_paths:
                rvs.append(roll([cfg]))

    res = concat(rvs)
    res['num_rollouts'] = num_rollouts
    filepart = 'test' if test else 'eval'
    filename = f'{batch_path.parts[-1]}.l.{filepart}.info.json'
    target_path = batch_path / filename
    with target_path.open('w') as fj:
        json.dump(res, fj)

    sys.stdout.write(str(batch_path))
    return str(batch_path)

@processable
def rollout_job(test=False):
    return rollout_batch(test=test)

if __name__ == '__main__':
    rollout_job()
