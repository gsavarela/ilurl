"""Grid example."""
import json

from flow.controllers import GridRouter
from flow.core.experiment import Experiment
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams
from flow.core.params import TrafficLightParams
from flow.core.params import SumoCarFollowingParams
from flow.core.params import InFlows

from flow.scenarios.grid import SimpleGridScenario
from ilu.envs.traffic_lights import TrafficLightQLGridEnv, ADDITIONAL_QL_ENV_PARAMS


EMISSION_PATH = '/Users/gsavarela/sumo_data/'
HORIZON = 1500
NUM_ITERATIONS = 5


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
    initial = InitialConfig(
        spacing='custom', lanes_distribution=float('inf'), shuffle=True)

    inflow = InFlows()
    outer_edges = gen_edges(col_num, row_num)
    for i in range(len(outer_edges)):
        inflow.add(
            veh_type='human',
            edge=outer_edges[i],
            probability=0.25,
            departLane='free',
            departSpeed=20)

    net = NetParams(
        inflows=inflow,
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
    initial = InitialConfig(
        spacing='custom', additional_params=additional_init_params)
    net = NetParams(
        no_internal_links=False, additional_params=add_net_params)

    return initial, net


def grid_example(render=None, use_inflows=False):
    """
    Perform a simulation of vehicles on a grid.

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
    inner_length = 300
    long_length = 500
    short_length = 300
    n_rows = 2
    # n_columns = 3
    n_columns = 2
    num_cars_left = 20
    num_cars_right = 20
    num_cars_top = 20
    num_cars_bot = 20
    tot_cars = (num_cars_left + num_cars_right) * n_columns \
        + (num_cars_top + num_cars_bot) * n_rows

    grid_array = {
        "short_length": short_length,
        "inner_length": inner_length,
        "long_length": long_length,
        "row_num": n_rows,
        "col_num": n_columns,
        "cars_left": num_cars_left,
        "cars_right": num_cars_right,
        "cars_top": num_cars_top,
        "cars_bot": num_cars_bot
    }

    sim_params = SumoParams(sim_step=0.1, render=False, print_warnings=False)

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

    additional_env_params = ADDITIONAL_QL_ENV_PARAMS.copy()
    additional_env_params.update({
        # minimum switch time for each traffic light (in seconds)
        "switch_time": 15.0,
        # whether the traffic lights should be actuated by sumo or RL
        # options are "controlled" and "actuated"
        "tl_type": "controlled",
        # determines whether the action space is meant to be discrete or continuous
        "discrete": True
    })

    env_params = EnvParams(horizon=HORIZON,
                           additional_params=additional_env_params)

    # tl_logic = TrafficLightParams(baseline=False)
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
    # tl_logic.add("center0", phases=phases, programID=1)
    # tl_logic.add("center1", phases=phases, programID=1)
    # tl_logic.add("center2", phases=phases, programID=1, tls_type="actuated")

    additional_net_params = {
        "grid_array": grid_array,
        "speed_limit": 35,
        "horizontal_lanes": 1,
        "vertical_lanes": 1
    }

    if use_inflows:
        initial_config, net_params = get_flow_params(
            col_num=n_columns,
            row_num=n_rows,
            additional_net_params=additional_net_params)
    else:
        initial_config, net_params = get_non_flow_params(
            enter_speed=v_enter,
            add_net_params=additional_net_params)

    scenario = SimpleGridScenario(
        name="smart-grid",
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config,
        traffic_lights=TrafficLightParams(baseline=False))

    #env = AccelEnv(env_params, sim_params, scenario)
    env = TrafficLightQLGridEnv(env_params, sim_params, scenario)

    return Experiment(env), env



if __name__ == "__main__":
    # import the experiment variable
    exp, env = grid_example()

    # run for a set number of rollouts / time steps
    data = exp.run(
        NUM_ITERATIONS,
        HORIZON,
        rl_actions=env.eps_greedy
    )
    dump_filename = \
        "{0}.dump".format(env.scenario.name)

    dump_path = "{}{}".format(EMISSION_PATH, dump_filename)

    data['velocities'] = {i: list(v) for i, v in enumerate(data['velocities'])}
    with open(dump_path, 'w') as fp:
        json.dump(data, fp)

