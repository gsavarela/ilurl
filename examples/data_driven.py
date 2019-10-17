"""This script loads a template from data"""

__author__ = 'Guilherme Varela'
__date__ = '2019-10-08'

import time
import os

from flow.controllers import GridRouter
from flow.core.params import (EnvParams, InFlows, InitialConfig, NetParams,
                              SumoCarFollowingParams, SumoParams,
                              TrafficLightParams, VehicleParams)
from flow.envs.loop.loop_accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.scenarios.grid import SimpleGridScenario
from flow.scenarios import Scenario
from ilurl.benchmarks.grid import grid_example
from ilurl.core.experiment import Experiment
from ilurl.core.params import QLParams
from ilurl.envs.green_wave_env import TrafficLightQLGridEnv

EMISSION_PATH = '/Users/gsavarela/sumo_data/'
HORIZON = 1500
NUM_ITERATIONS = 5
SHORT_CYCLE_TIME = 31
LONG_CYCLE_TIME = 45
SWITCH_TIME = 6


def get_flow_params(flow_sources, additional_net_params):
    """Define the network and initial params in the presence of inflows.

    Parameters
    ----------
    flow_sources: list of strings
        ids from the edges in which the vehicles come from
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
    for i in range(len(flow_sources)):
        inflow.add(veh_type='human',
                   edge=flow_sources[i],
                   probability=0.25,
                   depart_lane='free',
                   depart_speed=20)

    net = NetParams(inflows=inflow,
                    template=f'{os.getcwd()}/data/networks/intersection.net.xml',
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

    net = NetParams(
        template= f'{os.getcwd()}/data/networks/intersection.net.xml',
        additional_params=add_net_params)

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
    tot_cars = 160
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

    if render is None:
        sim_params = SumoParams(sim_step=sim_step,
                                render=False,
                                print_warnings=False,
                                emission_path=emission_path)

    else:
        sim_params = SumoParams(sim_step=sim_step,
                                render=render,
                                print_warnings=False,
                                emission_path=emission_path)

    if render is not None:
        sim_params.render = render

    vehicles = VehicleParams()
    vehicles.add(
        veh_id="human",
        routing_controller=(GridRouter, {}),
        car_following_params=SumoCarFollowingParams(
            min_gap=2.5,
            decel=7.5,  # avoid collisions at emergency stops
        ),
        num_vehicles=tot_cars)

    env_params = EnvParams(horizon=HORIZON,
                           additional_params=ADDITIONAL_ENV_PARAMS)



    tl_logic = TrafficLightParams(baseline=False)
    # phases = [{
    #     "duration": "31",
    #     "minDur": "8",
    #     "maxDur": "45",
    #     "state": "GrGrGrGrGrGr"
    # }, {
    #     "duration": "6",
    #     "minDur": "3",
    #     "maxDur": "6",
    #     "state": "yryryryryryr"
    # }, {
    #     "duration": "31",
    #     "minDur": "8",
    #     "maxDur": "45",
    #     "state": "rGrGrGrGrGrG"
    # }, {
    #     "duration": "6",
    #     "minDur": "3",
    #     "maxDur": "6",
    #     "state": "ryryryryryry"
    # }]
    # # Junction ids
    # tl_logic.add("247123161", phases=phases, programID=1)
    # tl_logic.add("247123374", phases=phases, programID=1)
    # tl_logic.add("center2", phases=phases, programID=1, tls_type="actuated")

    # Define flow
    # lookup ids
    additional_net_params = {
        'template': f'{os.getcwd()}/data/networks/intersection.net.xml',
        "speed_limit": 35
    }

    edge_ids = [
        "309265401#0",
        "-306967025#2",
        "96864982#0",
        "309265398#0"
    ]
    if use_inflows:
        initial_config, net_params = get_flow_params(
            edge_ids,
            additional_net_params=additional_net_params)
    else:
        initial_config, net_params = get_non_flow_params(
            enter_speed=v_enter, add_net_params=additional_net_params)

    # TODO: template should be an input variable
    # assumption project gets run from root
    scenario = Scenario(
        name="intersection",
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config,
        traffic_lights=tl_logic)

    env = AccelEnv(
        env_params=env_params,
        sim_params=sim_params,
        scenario=scenario)

    exp = Experiment(env)

    return Experiment(env)


if __name__ == "__main__":
    exp = network_example(
        render=True,
        use_inflows=True
    )

    exp.run(NUM_ITERATIONS, HORIZON)
