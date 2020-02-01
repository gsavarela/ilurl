"""Provides baseline for networks"""
__author__ = 'Guilherme Varela'
__date__ = '2020-01-24'

import os
import json
import dill

from ilurl.envs.base import TrafficLightQLEnv #, QL_PARAMS
from ilurl.core.experiment import Experiment

from ilurl.networks.base import Network

# TODO: Generalize for any parameter
ROOT_PATH = os.environ['ILURL_HOME']
EXPERIMENTS_PATH = f'{ROOT_PATH}/data/experiments/0x01/'
NETWORK_PATH = f'{ROOT_PATH}/data/networks/intersection/'
# run analysis/evaluate to get those files
POLICIES = (
        '4545/intersection_20200123-2319551579821595.08625.Q.6.pickle',
        '5040/intersection_20200123-2327251579822045.260267.Q.9.pickle',
        '5436/intersection_20200123-2328431579822123.730189.Q.6.pickle',
        '6030/intersection_20200123-2324481579821888.7447438.Q.4.pickle')

if __name__ == '__main__':
    # TODO: parametrize
    network_id = 'intersection'
    code = 'w'
    horizon = 9000
    inflows_type = 'switch'

    # TODO: test db path
    path = f'{NETWORK_PATH}intersection.test.{horizon}.{code}.rou.xml' 
    if os.path.isfile(path):
        # loading
        print('loading', path)
        network = Network.load(network_id, path)
    else:
        network = Network.make(
            network_id, horizon, inflows_type, 1, 'test'
        )

    for policy_name in POLICIES:
        print(f'process: {policy_name}')

        policy_dir = policy_name.split('/')[0]
        policy_path = f'{EXPERIMENTS_PATH}{policy_name}'
        with open(policy_path, mode='rb') as f:
            policy = dill.load(f)
       
        env_path = '.'.join(policy_name.split('.')[:2])
        env_path = f'{EXPERIMENTS_PATH}{env_path}.pickle'
        env = TrafficLightQLEnv.load(env_path)
        network.name = env.network.name

        env = TrafficLightQLEnv(
            env_params=env.env_params,
            sim_params=env.sim_params,
            ql_params=env.ql_params,
            network=network
        )
        # always generate emissions
        emission_path = f'{EXPERIMENTS_PATH}{policy_dir}'
        env.sim_params.emission_path = emission_path
        # prevent environment from learning
        env.stop = True

         # dir_path controlss the path of pickle 
         # we don't want it pickled
        exp = Experiment(env=env, dir_path=None, train=True)
        info_dict = exp.run(1, horizon)
        filename = \
             f"{network.name}.test.{horizon}.{code}.info.json"

        info_path = os.path.join(emission_path, filename)
        with open(info_path, 'w') as fj:
            json.dump(info_dict, fj)

