"""This script loads a template from data"""

__author__ = 'Guilherme Varela'
__date__ = '2019-10-08'

import time
import os

from flow.controllers import GridRouter
from flow.core.params import (EnvParams, InFlows, InitialConfig, NetParams,
                              SumoCarFollowingParams, SumoParams,
                              TrafficLightParams, VehicleParams)
from flow.envs.green_wave_env import ADDITIONAL_ENV_PARAMS
from flow.envs import TestEnv
from flow.scenarios.grid import SimpleGridScenario
from flow.scenarios import Scenario
from ilu.benchmarks.grid import grid_example
from ilu.core.experiment import Experiment
from ilu.core.params import QLParams
from ilu.envs.agents import TrafficLightQLGridEnv

EMISSION_PATH = '/Users/gsavarela/sumo_data/'
HORIZON = 1500
NUM_ITERATIONS = 5
SHORT_CYCLE_TIME = 31
LONG_CYCLE_TIME = 45
SWITCH_TIME = 6


def gen_edges(col_num, row_num):
    """Generate the names of the outer edges in the grid network.

    Parameters
    ----------
    col_num : int
        number of columns in the grid
    row_num : int
        number of rows in the grid

    Returns
    -------
    list of str
        names of all the outer edges
    """
    edges = []

    # build the left and then the right edges
    for i in range(col_num):
        edges += ['left' + str(row_num) + '_' + str(i)]
        edges += ['right' + '0' + '_' + str(i)]

    # build the bottom and then top edges
    for i in range(row_num):
        edges += ['bot' + str(i) + '_' + '0']
        edges += ['top' + str(i) + '_' + str(col_num)]

    return edges


def get_flow_params(col_num, row_num, additional_net_params):
    """Define the network and initial params in the presence of inflows.

    Parameters
    ----------
    col_num : int
        number of columns in the grid
    row_num : int
        number of rows in the grid
    additional_net_params : dict
        network-specific parameters that are unique to the grid

    Returns
    -------
    flow.core.params.InitialConfig
        parameters specifying the initial configuration of vehicles in the
        network
    flow.core.params.NetParams
        network-specific parameters used to generate the scenario
    """
    initial = InitialConfig(spacing='custom',
                            lanes_distribution=float('inf'),
                            shuffle=True)

    inflow = InFlows()
    outer_edges = gen_edges(col_num, row_num)
    for i in range(len(outer_edges)):
        inflow.add(veh_type='human',
                   edge=outer_edges[i],
                   probability=0.25,
                   departLane='free',
                   departSpeed=20)

    net = NetParams(inflows=inflow,
                    no_internal_links=False,
                    additional_params=additional_net_params)

    return initial, net


def get_non_flow_params(enter_speed, add_net_params):
    """Define the network and initial params in the absence of inflows.

    Note that when a vehicle leaves a network in this case, it is immediately
    returns to the start of the row/column it was traversing, and in the same
    direction as it was before.

    Parameters
    ----------
    enter_speed : float
        initial speed of vehicles as they enter the network.
    add_net_params: dict
        additional network-specific parameters (unique to the grid)

    Returns
    -------
    flow.core.params.InitialConfig
        parameters specifying the initial configuration of vehicles in the
        network
    flow.core.params.NetParams
        network-specific parameters used to generate the scenario
    """
    additional_init_params = {'enter_speed': enter_speed}
    initial = InitialConfig(spacing='custom',
                            additional_params=additional_init_params)

    net = NetParams(additional_params=add_net_params)

    return initial, net


def network_example(render=None,
                    use_inflows=False,
                    additional_env_params=None,
                    emission_path=None,
                    sim_step=0.1):
    """
    Perform a the simulation on a predefined network

    Parameters
    ----------
    render: bool, optional
        specifies whether to use the gui during execution
    use_inflows : bool, optional
        set to True if you would like to run the experiment with inflows of
        vehicles from the edges, and False otherwise

    Returns
    -------
    exp: flow.core.experiment.Experiment
        A non-rl experiment demonstrating the performance of human-driven
        vehicles and balanced traffic lights on a grid.
    """
    v_enter = 10
    # inner_length = 300
    # long_length = 500
    # short_length = 300
    # n_rows = 2
    # # n_columns = 3
    # n_columns = 2
    # num_cars_left = 20
    # num_cars_right = 20
    # num_cars_top = 20
    # num_cars_bot = 20
    # tot_cars = (num_cars_left + num_cars_right) * n_columns \
    #     + (num_cars_top + num_cars_bot) * n_rows

    # grid_array = {
    #     "short_length": short_length,
    #     "inner_length": inner_length,
    #     "long_length": long_length,
    #     "row_num": n_rows,
    #     "col_num": n_columns,
    #     "cars_left": num_cars_left,
    #     "cars_right": num_cars_right,
    #     "cars_top": num_cars_top,
    #     "cars_bot": num_cars_bot
    # }

    # net_params = NetParams(
    #     template=
    # )
    # if render is None:
    #     sim_params = SumoParams(sim_step=sim_step,
    #                             render=False,
    #                             print_warnings=False,
    #                             emission_path=emission_path)

    # else:
    #     sim_params = SumoParams(sim_step=sim_step,
    #                             render=render,
    #                             print_warnings=False,
    #                             emission_path=emission_path)

    # if render is not None:
    #     sim_params.render = render

    # vehicles = VehicleParams()
    # vehicles.add(
    #     veh_id="human",
    #     routing_controller=(GridRouter, {}),
    #     car_following_params=SumoCarFollowingParams(
    #         min_gap=2.5,
    #         decel=7.5,  # avoid collisions at emergency stops
    #     ),
    #     num_vehicles=tot_cars)

    # if additional_env_params is None:
    #     additional_env_params = ADDITIONAL_ENV_PARAMS.copy()
    #     additional_env_params[
    #         'short_cycle_time'] = SHORT_CYCLE_TIME
    #     additional_env_params[
    #         'long_cycle_time'] = LONG_CYCLE_TIME

    # additional_env_params.update({
    #     # minimum switch time for each traffic light (in seconds)
    #     "switch_time": SWITCH_TIME,
    #     # whether the traffic lights should be actuated by sumo or RL
    #     # options are "controlled" and "actuated"
    #     "tl_type": "controlled",
    #     # determines whether the action space is meant to be discrete or continuous
    #     "discrete": True,
    # })

    # env_params = EnvParams(horizon=HORIZON,
    #                        additional_params=additional_env_params)

    # additional_net_params = {
    #     "grid_array": grid_array,
    #     "speed_limit": 35,
    #     "horizontal_lanes": 1,
    #     "vertical_lanes": 1
    # }

    # if use_inflows:
    #     initial_config, net_params = get_flow_params(
    #         col_num=n_columns,
    #         row_num=n_rows,
    #         additional_net_params=additional_net_params)
    # else:
    #     initial_config, net_params = get_non_flow_params(
    #         enter_speed=v_enter, add_net_params=additional_net_params)

    # TODO: template should be an input variable
    # assumption project gets run from root
    import os
    template = f'{os.getcwd()}/data/networks/intersection.net.xml'
    net_params = NetParams(
        template=template
    )
    env_params = EnvParams()
    initial_config = InitialConfig()
    sim_params = SumoParams(render=True, sim_step=0.1)
    vehicles = VehicleParams()
    vehicles.add("Human", num_vehicles=10)
    scenario = Scenario(
        name="intersection",
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config,
        traffic_lights=TrafficLightParams(baseline=True))

    env = TestEnv(
        env_params=env_params,
        sim_params=sim_params,
        scenario=scenario)

    exp = Experiment(env)
    # ql_params = QLParams(epsilon=0.10, alpha=0.05,
    #                      states=('flow', 'queue'),
    #                      rewards={'type': 'score', 'costs': None},
    #                      num_traffic_lights=n_columns * n_rows,
    #                      c=10,
    #                      choice_type='ucb')
    # env = TrafficLightQLGridEnv(env_params, sim_params, ql_params, scenario)

    return Experiment(env)


if __name__ == "__main__":
    exp = network_example()

    exp.run(NUM_ITERATIONS, HORIZON)
    # # import the experiment variable
    # import os
    # import json
    # # print('running grid_intersection')
    # # start = time.time()
    # # exp = grid_example(
    # #     short_cycle_time=SHORT_CYCLE_TIME,
    # #     long_cycle_time=LONG_CYCLE_TIME,
    # #     switch_time=SWITCH_TIME,
    # #     render=False,
    # #     emission_path=None)

    # # grid_dict = exp.run(NUM_ITERATIONS, HORIZON)

    # print('running smart_grid')
    # start = time.time()
    # exp = smart_grid_example(render=False, emission_path=None)
    # # de-serialize data
    # # env = TrafficLightQLGridEnv.load(pickle_path)
    # # run for a set number of rollouts / time steps
    # info_dict = exp.run(NUM_ITERATIONS, HORIZON)
    # print(time.time() - start)
    # # serialize data
    # # UNCOMMENT to serialize
    # exp.env.dump(os.getcwd())
    # infoname = '{}.info.json'.format(exp.env.scenario.name)
    # with open(infoname, 'w') as f:
    #     json.dump(info_dict, f)
