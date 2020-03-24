import os
import json
import argparse
import pickle

from flow.core.params import SumoParams, EnvParams

from ilurl.envs.base import TrafficLightEnv

from ilurl.core.params import QLParams
from ilurl.core.experiment import Experiment

from ilurl.networks.base import Network

ILURL_HOME = os.environ['ILURL_HOME']

EMISSION_PATH = \
    f'{ILURL_HOME}/data/emissions'

# TODO: Put these as command line arguments.
EXPERIMENT = 'intersection_20200324-1318121585055892.4707007'
Q_TABLE_NUMBER = '2300'

def get_arguments():
    parser = argparse.ArgumentParser(
        description="""
            This script evaluates a traffic light system.
        """
    )

    parser.add_argument('--number-cycles', '-c', dest='num_cycles', type=int,
                        default=300, nargs='?',
                        help='Number of cycles to perform evaluation.')

    parser.add_argument('--sumo-render', '-r', dest='render', type=str2bool,
                        default=False, nargs='?',
                        help='If true renders the simulation.')

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

def print_arguments(args):

    print('Arguments:')
    print('\tExperiment number of cycles: {0}'.format(args.num_cycles))
    print('\tSUMO render: {0}\n'.format(args.render))

def setup_programs(programs_json):
    programs = {}
    for tls_id in programs_json.keys():
        programs[tls_id] = {int(action): programs_json[tls_id][action]
                                for action in programs_json[tls_id].keys()}
    return programs


if __name__ == '__main__':

    args = get_arguments()
    print_arguments(args)

    # Load parameters.
    params_file = '{0}/{1}.params.json'.format(EXPERIMENT,
                                               EXPERIMENT)
    params_path = os.path.join(EMISSION_PATH, params_file)
    with open(params_path) as json_file:
        params = json.load(json_file)
    
    # Load Q-table.
    q_table_file = '{0}/{1}.Q.1-{2}.pickle'.format(EXPERIMENT,
                                                   EXPERIMENT,
                                                   Q_TABLE_NUMBER)
    q_table_path = os.path.join(EMISSION_PATH, q_table_file)
    Q_table = pickle.load(open(q_table_path, "rb" ))

    path = '{0}/{1}'.format(EMISSION_PATH, EXPERIMENT)

    sumo_args = params['sumo_args']
    sumo_args['emission_path'] = path
    sumo_args['render'] = args.render
    sim_params = SumoParams(**sumo_args)
    
    env_params = EnvParams(**params['env_args'])

    sim_step = params['sumo_args']['sim_step']
    cycle_time = params['env_args']['additional_params']['cycle_time']
    horizon_t = int((cycle_time * args.num_cycles) / sim_step)

    network_args = params['network_args']
    network_args['horizon'] = horizon_t
    network = Network(**network_args)

    # Agent.
    from ilurl.core.ql.dpq import DPQ

    ql_params = QLParams(**params['ql_args'])
    QL_agent = DPQ(ql_params)

    programs = setup_programs(params['programs'])
    env = TrafficLightEnv(
        env_params=env_params,
        sim_params=sim_params,
        agent=QL_agent,
        network=network,
        TLS_programs=programs
    )

    # Setup Q-table.
    env.Q = Q_table

    # Stop training.
    env.stop = True

    exp = Experiment(env=env,
                    dir_path=None,
                    train=False)

    print('Running evaluation...')

    info_dict = exp.run(horizon_t)

    # Save evaluation log.
    file_path = '{0}/{1}.eval.json'.format(EXPERIMENT,
                                           env.network.name)
    info_path = os.path.join(EMISSION_PATH, file_path)
    print(info_path)
    with open(info_path, 'w') as fj:
        json.dump(info_dict, fj)