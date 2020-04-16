from pathlib import Path
from datetime import datetime
import sys
import os
import json
import tempfile
import configparser
import multiprocessing as mp
import time

from models.train import main as train
from ilurl.utils.decorators import processable

ILURL_HOME = os.environ['ILURL_HOME']

CONFIG_PATH = \
    f'{ILURL_HOME}/config/'

LOCK = mp.Lock()

def delay_train(*args, **kwargs):
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
    return train(*args, **kwargs)


def train_batch():
    # Read script arguments from run.config file.
    run_config = configparser.ConfigParser()
    run_config.read(os.path.join(CONFIG_PATH, 'run.config'))

    num_processors = int(run_config.get('run_args', 'num_processors'))
    num_runs = int(run_config.get('run_args', 'num_runs'))
    train_seeds = json.loads(run_config.get("run_args","train_seeds"))

    if len(train_seeds) != num_runs:
        raise configparser.Error('Number of seeds in run.config `train_seeds`'
                        ' must match the number of runs (`num_runs`) argument.')

    print('Arguments (run_train.py):')
    print('\tNumber of runs: {0}'.format(num_runs))
    print('\tNumber of processors: {0}'.format(num_processors))
    print('\tTrain seeds: {0}\n'.format(train_seeds))

    # Assess total number of processors.
    processors_total = mp.cpu_count()
    print(f'Total number of processors available: {processors_total}\n')

    # Adjust number of processors.
    if num_processors > processors_total:
        num_processors = processors_total
        print(f'Number of processors downgraded to {num_processors}\n')

    # Read train.py arguments from train.config file.
    train_config = configparser.ConfigParser()
    train_config.read(os.path.join(CONFIG_PATH, 'train.config'))

    with tempfile.TemporaryDirectory() as tmp_dir:

        # Create a config file for each train.py
        # with the respective seed. These config
        # files are stored in a temporary directory.
        train_configs = []
        for seed in train_seeds:

            tmp_train_cfg_path = os.path.join(tmp_dir,
                                        'train-{0}.config'.format(seed))
            train_configs.append(tmp_train_cfg_path)

            # Setup train seed.
            train_config.set("train_args", "experiment-seed", str(seed))
            
            # Write temporary train config file.
            tmp_cfg_file = open(tmp_train_cfg_path, "w")

            train_config.write(tmp_cfg_file)
            tmp_cfg_file.close()

        # Run.
        # TODO: option without pooling not working. why?
        # rvs: directories' names holding experiment data
        if num_processors > 1:
            pool = mp.Pool(num_processors)
            rvs = pool.map(delay_train, [[cfg] for cfg in train_configs])
            pool.close()
        else:
            rvs = []
            for cfg in train_configs:
                rvs.append(delay_train([cfg]))
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
def train_job():
    return train_batch()

if __name__ == '__main__':
    # enable this in order to have a nice textual ouput
    train_batch()
    # train_job()

