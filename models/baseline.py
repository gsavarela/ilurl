"""Provides baseline for scenarios"""
__author__ = 'Guilherme Varela'
__date__ = '2020-01-08'

import os
import json
import argparse
import math

from flow.core.params import SumoParams, EnvParams, InFlows

from flow.envs.loop.loop_accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from ilurl.core.experiment import Experiment

from ilurl.scenarios.base import BaseScenario, get_edges, get_routes

# TODO: Generalize for any parameter
ILURL_HOME = os.environ['ILURL_HOME']

EMISSION_PATH = \
    f'{ILURL_HOME}/data/emissions/'


def get_arguments():
    parser = argparse.ArgumentParser(
        description="""
            This script runs a traffic light simulation based on
            custom environment with with presets saved on data/networks
        """
    )

    # TODO: validate against existing networks
    parser.add_argument('scenario', type=str, nargs='?',
                        help='Network to be simulated')


    parser.add_argument('--experiment-time', '-t', dest='time', type=int,
                        default=360, nargs='?', help='Simulation\'s real world time in seconds')


    parser.add_argument('--experiment-iterations', '-i', dest='num_iterations', type=int,
                        default=1, nargs='?', help='Number of times to repeat the experiment')

    parser.add_argument('--sumo-render', '-r', dest='render', type=str2bool,
                        default=False, nargs='?', help='Renders the simulation')

    parser.add_argument('--sumo-print', '-p',
                        dest='print', type=str2bool, default=False, nargs='?',
                        help='Prints warning from simulation')
    
    parser.add_argument('--sumo-step', '-s',
                        dest='step', type=float, default=0.1, nargs='?',
                        help='Simulation\'s step size which is a fraction from horizon')

    parser.add_argument('--sumo-emission', '-e',
                        dest='emission', type=str2bool, default=False, nargs='?',
                       help='Saves emission data from simulation on /data/emissions')


    parser.add_argument('--inflow-switch', '-W', dest='switch',
                        type=str2bool, default=False, nargs='?',
                        help='''Assign higher probability of spawning a vehicle every other hour on opposite sides''')

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

if __name__ == '__main__':
    args = get_arguments()
    
    sumo_args = {
        'render': args.render,
        'print_warnings': args.print,
        'sim_step': args.step,
        'restart_instance': True
    }
    if args.emission:
        sumo_args['emission_path'] = EMISSION_PATH

    sim_params = SumoParams(**sumo_args)

    env_params = EnvParams(evaluate=True,
                           additional_params=ADDITIONAL_ENV_PARAMS)

    # inflows = make_inflows(args.scenario, args.time) if args.switch else None
    inflows_type = 'switch' if args.switch else 'lane'
    scenario = BaseScenario(
        network_id=args.scenario,
        horizon=args.time,
        inflows_type=inflows_type
    )


    env = AccelEnv(
        env_params=env_params,
        sim_params=sim_params,
        scenario=scenario
    )

    exp = Experiment(env=env)

    import time
    start = time.time()
    info_dict = exp.run(args.num_iterations, int(args.time / args.step))
    # save info dict
    # save pickle environment
    # TODO: save with running parameters
    # general process information
    x = 'l' if inflows_type == 'lane' else 'w'
    filename = \
        f"{env.scenario.name}.{args.time}.{x}.info.json"

    info_path = os.path.join(EMISSION_PATH, filename)
    with open(info_path, 'w') as fj:
        json.dump(info_dict, fj)

    if hasattr(env, 'dump'):
        env.dump(EMISSION_PATH)

    print(f'Elapsed time {time.time() - start}')
