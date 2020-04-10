import os
import configparser
import multiprocessing as mp

from models.train import main

ILURL_HOME = os.environ['ILURL_HOME']

SCRIPTS_PATH = \
    f'{ILURL_HOME}/scripts/'

if __name__ == '__main__':

    run_config = configparser.ConfigParser()
    run_config.read(os.path.join(SCRIPTS_PATH, 'run.config'))

    num_processors = int(run_config['run_args']['num_processors'])
    num_runs = int(run_config['run_args']['num_runs'])
    print('Arguments (run_train.py):')
    print('\tNumber of runs: {0}'.format(num_runs))
    print('\tNumber of processors: {0}\n'.format(num_processors))

    processors_total = mp.cpu_count()
    print(f'Total number of processors available: {processors_total}\n')

    if num_processors > processors_total:
        num_processors = processors_total
        print(f'Number of processors downgraded to {num_processors}\n')

    train_config_path = os.path.join(SCRIPTS_PATH, 'train.config')

    pool = mp.Pool(num_processors)
    runs_names = pool.map(main, [[train_config_path] for _ in range(num_runs)])
    pool.close()

    print(runs_names) 