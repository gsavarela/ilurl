"""This module acts as a wrapper for scenarios generated from network data"""
__author__ = 'Guilherme Varela'
__date__ = '2020-01-10'

import os

# Network related parameters
from flow.core.params import InitialConfig, TrafficLightParams
from flow.core.params import VehicleParams, SumoCarFollowingParams
from flow.controllers.routing_controllers import GridRouter

from flow.scenarios.base_scenario import Scenario

from ilurl.core.params import InFlows, NetParams
from ilurl.loaders.vtypes import get_vehicle_types
from ilurl.loaders.nets import (get_routes, get_edges, get_path, get_tls,
                                get_logic, get_connections, get_nodes,
                                get_types)

class BaseScenario(Scenario):
    """This class leverages on specs created by SUMO"""

    @classmethod
    def make(cls, network_id, horizon, demand_type, num_reps, label=None):
        """Builds a new scenario from rou.xml file -- the resulting
        vehicle trips will be almost-deterministic use it for validation
        
        Params:
        ------
        *   network_id: string
            identification of net.xml file, ex: `intersection`
        *   horizon: integer
            maximum emission time in seconds
        *   demand_type: string
            a demand distribution e.g `lane`
        *   num: integer

        Returns:
        -------
        *   scenario(s): ilurl.scenario.BaseScenario or list
            n = 0  attempts to load one scenario,
            n > 0  attempts to load n+1 scenarios returning a list
        """

        scenarios = []
        for path in range(num_reps):
            net_params = NetParams.from_template(
                network_id, horizon, demand_type, label=label
            )
            scenarios.append(
                BaseScenario(
                    network_id,
                    horizon,
                    net_params,
                    vehicles=VehicleParams())
            )

        ret = scenarios[0] if num_reps == 1 else scenarios
        return ret

    @classmethod
    def load(cls, network_id, route_path):
        """Attempts to load a new scenario from rou.xml and 
        vtypes.add.xml -- if it fails will call `make`
        the resulting vehicle trips will be stochastic use 
        it for training

        Params:
        ------
        *   network_id: string
            identification of net.xml file, ex: `intersection`
        *   horizon: integer
            maximum emission time in seconds
        *   demand_type: string
            string
        *   num: integer

        Returns:
        -------
        *   scenario(s): ilurl.scenario.BaseScenario or list
            n = 0  attempts to load one scenario,
            n > 0  attempts to load n+1 scenarios returning a list
        """

        net = get_path(network_id, 'net')
        vtype = get_vehicle_types()

        net_params = NetParams(
            template={
                'net': net,
                'vtype': vtype,
                'rou': [route_path]
            }
        )

        horizon = int(route_path.split('.')[-4])

        scenario = BaseScenario(
            network_id,
            horizon,
            net_params,
            vehicles=VehicleParams()
        )
        return scenario

    def __init__(self,
                 network_id,
                 horizon=360,
                 net_params=None,
                 vehicles=None,
                 demand_type='lane',
                 initial_config=None,
                 traffic_lights=None):


        """Builds a new scenario from inflows -- the resulting
        vehicle trips will be stochastic use it for training"""
        self.network_id = network_id

        if initial_config is None:
            initial_config = InitialConfig(
                edges_distribution=tuple(get_routes(network_id).keys())
            )

        if net_params is None:
            #TODO: check vtype
            if vehicles is None:
                # vtypes_path = get_vehicle_types()
                vehicles = VehicleParams()
                vehicles.add(
                    veh_id="human",
                    routing_controller=(GridRouter, {}),
                    car_following_params=SumoCarFollowingParams(
                        min_gap=2.5,
                        decel=7.5,  # avoid collisions at emergency stops
                    ),
                )

            inflows = InFlows(network_id, horizon, demand_type)
            net_params = NetParams(inflows,
                                   template=get_path(network_id, 'net'))

        if traffic_lights is None:
            prog_list = get_logic(network_id)
            if prog_list:
                traffic_lights = TrafficLightParams(baseline=False)
                for prog in prog_list:
                    prog_id = prog.pop('id')
                    prog['tls_type'] = prog.pop('type')
                    prog['programID'] = int(prog.pop('programID')) + 1
                    traffic_lights.add(prog_id, **prog)
            else:
                traffic_lights = TrafficLightParams(baseline=False)

        super(BaseScenario, self).__init__(
                 network_id,
                 vehicles,
                 net_params,
                 initial_config=initial_config,
                 traffic_lights=traffic_lights
        )

        self.nodes = self.specify_nodes(net_params)
        self.edges = self.specify_edges(net_params)
        self.connections = self.specify_connections(net_params)
        self.types = self.specify_types(net_params)

    def specify_nodes(self, net_params):
        return get_nodes(self.network_id)

    def specify_edges(self, net_params):
        return get_edges(self.network_id)

    def specify_connections(self, net_params):
        return get_connections(self.network_id)

    def specify_routes(self, net_params):
        return get_routes(self.network_id)

    def specify_types(self, net_params):
        return get_types(self.network_id)
