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

# TODO: Put these as command line arguments.
EMISSION_PATH = \
    f'{ILURL_HOME}/data/emissions/intersection_20200324-0141281585014088.4119484'
Q_FILE_NAME = 'intersection_20200324-0141281585014088.4119484.Q.1-300.pickle'
PARAMS_FILE_NAME = 'intersection_20200324-0141281585014088.4119484.params.json'

# TODO: Load cycle time dynamically.
CYCLE_TIME = 90

def get_arguments():
    parser = argparse.ArgumentParser(
        description="""
            This script evaluates a traffic light system.
        """
    )

    parser.add_argument('--number-cycles', '-c', dest='num_cycles', type=int,
                        default=300, nargs='?',
                        help='Nu,ber of cycles to perform evaluation')

    parser.add_argument('--sumo-render', '-r', dest='render', type=str2bool,
                        default=False, nargs='?',
                        help='Renders the simulation')

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


if __name__ == '__main__':

    args = get_arguments()
    print_arguments(args)

    # Load parameters.
    params_path = os.path.join(EMISSION_PATH, PARAMS_FILE_NAME)
    with open(params_path) as json_file:
        params = json.load(json_file)
    
    Q_table = pickle.load(open("{0}/{1}".format(EMISSION_PATH, Q_FILE_NAME), "rb" ))

    sumo_args = params['sumo_args']
    sumo_args['emission_path'] = EMISSION_PATH
    sumo_args['render'] = args.render
    sim_params = SumoParams(**sumo_args)
    
    env_params = EnvParams(**params['env_args'])

    sim_step = params['sumo_args']['sim_step']
    horizon_t = int((CYCLE_TIME * args.num_cycles) / sim_step)
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
    env.Q = Q_table

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