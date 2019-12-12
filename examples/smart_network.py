"""This script integrates a Open Street Map section, induction loop data and
    tabular q-learning

"""

__author__ = 'Guilherme Varela'
__date__ = '2019-12-10'

import time
# import os

import pandas as pd
from flow.controllers import GridRouter
from flow.core.params import (EnvParams, InFlows, SumoParams,
                              SumoCarFollowingParams, VehicleParams,
                              TrafficLightParams)


from ilurl.envs.tls import TrafficLightQLEnv
from ilurl.envs.green_wave_env import ADDITIONAL_ENV_PARAMS

from ilurl.core.params import QLParams
from ilurl.core.experiment import Experiment
from ilurl.scenarios.intersection import (IntersectionScenario,
                                          SOURCES)

from ilurl.loaders.induction_loops import get_induction_loops
from ilurl.loaders.induction_loops import groupby_induction_loops

EMISSION_PATH = '/Users/gsavarela/Work/py/ilu/ilurl/data/emissions/'
SIM_HOURS = 3
HORIZON = SIM_HOURS * 3600 * 10
NUM_ITERATIONS = 1
SHORT_CYCLE_TIME = 15
LONG_CYCLE_TIME = 60
SWITCH_TIME = 6

# This dictionary maps ID_LOOPS (espiras)
# graph edges
LOOP_TO_EDGE = {"3:9": "309265401#0"}


def build_flow_params(df=None):
    """Define the network and initial params in the presence of inflows.

    Parameters
    ----------
    * df : pandas.DataFrame

    Returns
    -------
    flow.core.params.InFlows
        Cars distributions to be placed at sources
    """

    inflows = InFlows()
    for edge_id in SOURCES:
        if df is None:
            vehs = (324, 200, 2764.25, 1352.75)
            for i, vehs_per_hour in enumerate(vehs):
                flow_name = f'static_{i:02d}'
                print(i, vehs_per_hour)
                inflows.add(name=flow_name,
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
                inflows.add(name=flow_name,
                            veh_type='human',
                            edge=LOOP_TO_EDGE[loop_id],
                            depart_lane='best',
                            depart_speed=20,
                            vehs_per_hour=vehs_per_hour,
                            begin=start,
                            end=start + 3599)
                start += 3600

    return inflows



def network_example(render=None,
                    use_induction_loops=False,
                    additional_env_params=None,
                    emission_path=None,
                    sim_step=0.1):
    """
    Perform a the simulation on a predefined network

    Parameters
    ----------
    * render: bool, optional
        specifies whether to use the gui during execution

    * use_induction_loops : bool, optional
        set to True if you would like to run the experiment with sensor
        data use False to choose a fixed traffic demand

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

    additional_env_params = ADDITIONAL_ENV_PARAMS.copy()
    additional_env_params.update({
        # minimum switch time for each traffic light (in seconds)
        "switch_time": SWITCH_TIME,
        # whether the traffic lights should be actuated by sumo or RL
        # options are "controlled" and "actuated"
        "tl_type": "controlled",
        # determines whether the action space is meant to be discrete or continuous
        "discrete": True,
        "short_cycle_time": SHORT_CYCLE_TIME,
        "long_cycle_time": LONG_CYCLE_TIME
    })
    env_params = EnvParams(horizon=HORIZON,
                           additional_params=additional_env_params)



    tl_logic = TrafficLightParams(baseline=False)

    phases = [{
        "duration": "39",
        "state": "GGgrrrrGGGrrr"
    }, {
        "duration": "6",
        "state": "yyyrrrryyyrrr"
    }, {
        "duration": "39",
        "state": "rrrGGggrrrGGg"
    }, {
        "duration": "6",
        "state": "rrryyyyrrryyy"
    }]
    # Junction ids
    # tl_logic.add("GS_247123161", phases=phases, programID=1)
    # this thing here is also bound by the scenario
    tl_logic.add("GS_247123161", programID=0)
    # tl_logic.add("247123374", phases=phases, programID=1)
    # tl_logic.add("center2", phases=phases, programID=1, tls_type="actuated")


    if use_induction_loops:
        df = get_induction_loops(('3:9',), workdays=True)
        df = groupby_induction_loops(df, width=5)
        df['edge_id'] = SOURCES[0]

        inflows = build_flow_params(df)
    else:
        inflows = build_flow_params()

    scenario = IntersectionScenario(
        name="intersection",
        vehicles=vehicles,
        traffic_lights=tl_logic,
        inflows=inflows)


    ql_params = QLParams(epsilon=0.10, alpha=0.05,
                         states=('speed', 'count'),
                         rewards={'type': 'weighted_average',
                                  'costs': None},
                         num_traffic_lights=1, c=10,
                         choice_type='ucb')

    env = TrafficLightQLEnv(
        env_params,
        sim_params,
        ql_params,
        scenario
    )

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
