"""Provides baseline for scenarios"""
__author__ = 'Guilherme Varela'
__date__ = '2020-01-08'

import os
import json
import argparse
import math

from flow.core.params import SumoParams, EnvParams
from flow.envs.loop.loop_accel import ADDITIONAL_ENV_PARAMS
from flow.core.params import InFlows

from ilurl.envs.base import TrafficLightQLEnv, QL_PARAMS
from ilurl.envs.base import ADDITIONAL_TLS_PARAMS

from ilurl.core.params import QLParams
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
    parser.add_argument('scenario', type=str, nargs='?', default='intersection',
                        help='Network to be simulated')


    parser.add_argument('--experiment-time', '-t', dest='time', type=int,
                        default=360, nargs='?', help='Simulation\'s real world time in seconds')


    parser.add_argument('--experiment-iterations', '-i', dest='num_iterations', type=int,
                        default=1, nargs='?',
                        help='Number of times to repeat the experiment')


    parser.add_argument('--experiment-pickle', '-p', dest='pickle', type=str2bool,
                        default=1, nargs='?',
                        help='Pickle the environment allowing to reproduce')

    parser.add_argument('--sumo-render', '-r', dest='render', type=str2bool,
                        default=False, nargs='?',
                        help='Renders the simulation')

    parser.add_argument('--sumo-step', '-s',
                        dest='step', type=float, default=0.1, nargs='?',
                        help='Simulation\'s step size which is a fraction from horizon')

    parser.add_argument('--sumo-emission', '-e',
                        dest='emission', type=str2bool, default=False, nargs='?',
                       help='Saves emission data from simulation on /data/emissions')


    parser.add_argument('--tls-short', '-S', dest='short_phase',
                        type=int, default=45, nargs='?',
                        help='Short phase length in seconds of the cycle')

    
    parser.add_argument('--tls-long', '-L', dest='long_phase',
                        type=int, default=False, nargs='?',
                        help='Long phase length in seconds of the cycle')


    parser.add_argument('--inflows-switch', '-W', dest='switch',
                        type=str2bool, default=False, nargs='?',
                        help='''Assign higher probability of spawning a vehicle every other hour on opposite sides''')

    return parser.parse_args()


def make_inflows(network_id, horizon):
    inflows = InFlows()
    edges = get_edges(network_id)
    switch = 3600   # switches flow every 3600 seconds
    for eid in get_routes(network_id):
        # use edges distribution to filter routes
        edge = [e for e in edges if e['id'] == eid][0]
        # TODO: get edges that are opposite and intersecting
        num_lanes = edge['numLanes'] if 'numLanes' in edge else 1
        prob0 = 0.2    # default emission prob (veh/s)
        num_flows = max(math.ceil(horizon / switch), 1)
        for hr in range(num_flows):
            step = min(horizon - hr * switch, switch)
            # switches in accordance to the number of lanes
            prob = prob0 - 0.1 if (hr + num_lanes) % 2 == 1 else prob0
            inflows.add(
                eid,
                'human',
                probability=prob,
                depart_lane='best',
                depart_speed='random',
                name=f'flow_{eid}',
                begin=1 + hr * switch,
                end=step + hr * switch
            )

    return inflows

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
        'print_warnings': False,
        'sim_step': args.step,
        'restart_instance': True
    }
    if args.emission:
        sumo_args['emission_path'] = EMISSION_PATH

    sim_params = SumoParams(**sumo_args)

    additional_params = {}
    additional_params.update(ADDITIONAL_ENV_PARAMS)
    additional_params.update(ADDITIONAL_TLS_PARAMS)
    additional_params['long_cycle_time'] = args.long_phase
    additional_params['short_cycle_time'] = args.short_phase

    print(args.long_phase, args.short_phase)
    env_params = EnvParams(evaluate=True,
                           additional_params=additional_params)

    inflows = make_inflows(args.scenario, args.time) if args.switch else None
    scenario = BaseScenario(
        network_id=args.scenario,
        horizon=args.time,
        inflows=inflows
    )

    ql_params = QLParams(epsilon=0.10, alpha=0.05,
                         states=('speed', 'count'),
                         rewards={'type': 'weighted_average',
                                  'costs': None},
                         num_traffic_lights=1, c=10,
                         choice_type='ucb')

    env = TrafficLightQLEnv(
        env_params=env_params,
        sim_params=sim_params,
        ql_params=ql_params,
        scenario=scenario
    )
    exp = Experiment(env=env)

    import time
    start = time.time()
    info_dict = exp.run(args.num_iterations, int(args.time / args.step))
    if args.pickle:
        # save info dict
        # save pickle environment
        # TODO: save with running parameters
        path = f'{EMISSION_PATH}/{args.long_phase}{args.short_phase}'
        if not os.path.isdir(path):
            os.mkdir(path)
        # general process information
        filename = \
            "{0}.info.json".format(env.scenario.name)

        info_path = os.path.join(path, filename)
        with open(info_path, 'w') as fj:
            json.dump(info_dict, fj)

        if hasattr(env, 'dump'):
            env.dump(path)

    print(f'Elapsed time {time.time() - start}')
