"""This script loads a template from data"""

__author__ = 'Guilherme Varela'
__date__ = '2019-10-08'

import time
import os

import pandas as pd
from flow.controllers import GridRouter
from flow.core.params import (EnvParams, InFlows, InitialConfig, NetParams,
                              SumoCarFollowingParams, SumoParams,
                              TrafficLightParams, VehicleParams)

from flow.envs.loop.loop_accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.scenarios import Scenario

from ilurl.core.experiment import Experiment
from ilurl.loaders.induction_loops import get_induction_loops
from ilurl.loaders.induction_loops import groupby_induction_loops

EMISSION_PATH = '/Users/gsavarela/Work/py/ilu/ilurl/data/emissions/'
SIM_HOURS = 24
HORIZON = SIM_HOURS * 3600 * 10
NUM_ITERATIONS = 1

# feed SOURCES to InitialConfig
# on edges distribution
# EDGES_DISTRIBUTION = ["309265401#0", "-306967025#2", "96864982#0",  "-238059324#1"]
EDGES_DISTRIBUTION = ["309265401#0"]


# This dictionary maps ID_LOOPS (espiras)
# graph edges
LOOP_TO_EDGE = {"3:9": "309265401#0"}

SOURCES = EDGES_DISTRIBUTION

SINKS = [
    "-309265401#2",
    "306967025#0",
    "238059324#0",
]

EDGES = ["212788159_0", "247123161_0", "247123161_1", "247123161_3",
         "247123161_14", "247123161_4", "247123161_5", "247123161_6",
         "247123161_15", "247123161_7", "247123161_8", "247123161_10",
         "247123161_16", "247123161_11", "247123161_12", "247123161_13",
         "247123161_17", "247123367_0", "247123367_1", "247123374_0",
         "247123374_1", "247123374_3", "247123374_9", "247123374_4",
         "247123374_5", "247123374_6", "247123374_7", "247123449_0",
         "247123449_2", "247123449_1", "247123464_0", "3928875116_0"]


class SampleIntersectionScenario(Scenario):

    def specify_routes(self, net_params):
        rts = {
            "309265401#0": [(["309265401#0", "238059328#0", "306967025#0"], 0.8),
                            (["309265401#0", "238059324#0"], 0.10),
                            (["309265401#0", "238059328#0",
                              "309265399#0", "96864982#1",
                              "392619842", "238059324#0"], 0.10)],
            "-306967025#2": ["-306967025#2", "-238059328#2", "-309265401#2"],
            # "96864982#0": ["96864982#0", "96864982#1", "392619842", "238059324#0"],
            # "-238059324#1": [(["-238059324#1", "-309265401#2"], 0.5), (["-238059324#1", "238059328#0", "306967025#0"], 0.5)],
            # "309265398#0": [(["309265398#0", "306967025#0"], 0.33)#, (["309265399#0", "96864982#1", "392619842", "-309265401#2"], 0.33), (["309265399#0", "96864982#1", "392619842", "238059324#0"], 0.34)]
        }
        return rts

    def specify_edge_starts(self):
        sts = [
            ("309265401#0", 77.4), ("238059328#0", 81.22),
            ("306967025#0", 131.99), ("-306967025#2", 131.99),
            ("-238059328#2", 81.22), ("-309265401#2", 77.4),
            ("96864982#0", 46.05), ("96864982#1", 82.63),
            ("392619842", 22.22), ("238059324#0", 418.00),
            ("-238059324#1", 418.00), ("309265398#0", 117.82),
            ("309265399#0", 104.56)
        ]

        return sts


def get_flow_params(additional_net_params, df=None):
    """Define the network and initial params in the presence of inflows.

    Parameters
    ----------
    * additional_net_params : dict
        network-specific parameters that are unique to the grid

    * df : pandas.DataFrame

    Returns
    -------
    flow.core.params.InitialConfig
        The initial configuration of vehicles in the network
    flow.core.params.NetParams
        Network-specific parameters used to generate the scenario
    """

    initial = InitialConfig(edges_distribution=EDGES_DISTRIBUTION)

    inflow = InFlows()
    for edge_id in EDGES_DISTRIBUTION:
        if df is None:
            vehs = (1500, 1600, 1700)
            for i, vehs_per_hour in enumerate(vehs):
                flow_name = f'static_{i:02d}'
                print(i, vehs_per_hour)
                inflow.add(name=flow_name,
                           veh_type='human',
                           edge=edge_id,
                           depart_lane='best',
                           depart_speed=20,
                           vehs_per_hour=vehs_per_hour,
                           begin=i * 3600 + 1,
                           end=(i + 1) * 3600)

        else:
            # TODO: Read start from DataFrame
            start = 1
            for idx, count in df.iterrows():
                # Data is given every 15 minutes
                dt, loop_id = idx
                if dt.hour == SIM_HOURS:
                    break
                vehs_per_hour = count['Count']
                print(dt.hour, vehs_per_hour)
                flow_name = f'loop_{loop_id:s}_{dt.hour:02d}'
                inflow.add(name=flow_name,
                           veh_type='human',
                           edge=LOOP_TO_EDGE[loop_id],
                           depart_lane='best',
                           depart_speed=20,
                           vehs_per_hour=vehs_per_hour,
                           begin=start,
                           end=start + 3599)
                start += 3600


    net = NetParams(inflows=inflow,
                    template=f'{os.getcwd()}/data/networks/sample_intersection/intersection.net.xml',
                    additional_params=additional_net_params)

    return initial, net



def network_example(render=None,
                    use_induction_loops=False,
                    additional_env_params=None,
                    emission_path=None,
                    sim_step=0.1):
    """
    Perform a the simulation on a predefined network

    Parameters
    ----------
    render: bool, optional
        specifies whether to use the gui during execution

    use_induction_loops : bool, optional
        set to True if you would like to run the experiment with sensor data use False to choose a fixed traffic demand

    Returns
    -------
    exp: flow.core.experiment.Experiment
        A non-rl experiment demonstrating the performance of human-driven
        vehicles and balanced traffic lights on a grid.

    Update
    ------
    2019-12-09: Add restart_instance;
    Should prevent the following warning:
    WARNING: Inflows will cause computational performance to
    significantly decrease after large number of rollouts. In
    order to avoid this, set SumoParams(restart_instance=True).
    """
    if render is None:
        sim_params = SumoParams(sim_step=sim_step,
                                render=False,
                                print_warnings=False,
                                emission_path=emission_path,
                                restart_instance=True)

    else:
        sim_params = SumoParams(sim_step=sim_step,
                                render=render,
                                print_warnings=False,
                                emission_path=emission_path,
                                restart_instance=True)

    vehicles = VehicleParams()
    vehicles.add(
        veh_id="human",
        routing_controller=(GridRouter, {}),
        car_following_params=SumoCarFollowingParams(
            min_gap=2.5,
            decel=7.5,  # avoid collisions at emergency stops
        )
    )

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
        "speed_limit": 35
    }

    if use_induction_loops:
        df = get_induction_loops(('3:9',), workdays=True)
        df = groupby_induction_loops(df, width=5)
        df['edge_id'] = EDGES_DISTRIBUTION[0]

        initial_config, net_params = get_flow_params(additional_net_params, df)
    else:
        initial_config, net_params = get_flow_params(additional_net_params)

    # TODO: template should be an input variable
    # assumption project gets run from root
    scenario = SampleIntersectionScenario(
        name="sample_intersection",
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config,
        traffic_lights=tl_logic)

    print(f'scenario: {scenario.name}')
           
    env = AccelEnv(
        env_params=env_params,
        sim_params=sim_params,
        scenario=scenario)

    exp = Experiment(env)

    return Experiment(env)


if __name__ == "__main__":
    import time
    import datetime
    start = time.time()
    exp = network_example(
        render=False,
        use_induction_loops=True,
        emission_path=EMISSION_PATH
    )

    exp.run(NUM_ITERATIONS, HORIZON, convert_to_csv=True)
    elapsed = datetime.timedelta(seconds=time.time() - start)
    print(f'total running time: {str(elapsed):s}')
