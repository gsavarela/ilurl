import os
import json
import argparse
import pickle
import time

from flow.core.params import SumoParams, EnvParams

from ilurl.envs.base import TrafficLightEnv

from ilurl.core.params import QLParams
from ilurl.core.experiment import Experiment

from ilurl.networks.base import Network

ILURL_HOME = os.environ['ILURL_HOME']

# TODO: Put these as command line arguments.
EMISSION_PATH = \
    f'{ILURL_HOME}/data/emissions/intersection_20200323-1700241584982824.4755545'
#Q_FILE_NAME = 'intersection_20200316-1930221584387022.6974635.Q.1-48000.pickle'
PARAMS_FILE_NAME = 'intersection_20200323-1700241584982824.4755545.params.json'
RENDER = False
NUM_CYCLES = 20

# TODO: Load these parameters dynamically.
STEP = 1
CYCLE_TIME = 90

if __name__ == '__main__':

    # Load parameters.
    params_path = os.path.join(EMISSION_PATH, PARAMS_FILE_NAME)
    with open(params_path) as json_file:
        params = json.load(json_file)

    # Q_table = pickle.load(open("{0}/{1}".format(EMISSION_PATH, Q_FILE_NAME), "rb" ))

    sumo_args = params['sumo_args']
    sumo_args['emission_path'] = EMISSION_PATH
    sim_params = SumoParams(**sumo_args)
    
    env_params = EnvParams(**params['env_args'])

    horizon_t = int((CYCLE_TIME * NUM_CYCLES) / STEP)
    network_args = params['network_args']
    network_args['horizon'] = horizon_t
    network = Network(**network_args)

    # Agent.
    from ilurl.core.ql.dpq import DPQ

    ql_params = QLParams(**params['ql_args'])
    QL_agent = DPQ(ql_params)

    env = TrafficLightEnv(
        env_params=env_params,
        sim_params=sim_params,
        agent=QL_agent,
        network=network
    )

    # Setup Q-table.
    #env.Q = Q_table

    # Stop training.
    env.stop = True

    exp = Experiment(env=env,
                    dir_path=None,
                    train=False)

    print('Running evaluation...')

    info_dict = exp.run(horizon_t)

    # Save evaluation log.
    filename = \
            f"{env.network.name}.eval.json"

    info_path = os.path.join(EMISSION_PATH, filename)
    with open(info_path, 'w') as fj:
        json.dump(info_dict, fj)