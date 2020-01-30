"""This module acts as a wrapper for scenarios generated from network data"""
__author__ = 'Guilherme Varela'
__date__ = '2020-01-10'

import os
import math
import xml.etree.ElementTree as ET

# InFlows
from flow.core.params import InFlows

# Network related parameters
from flow.core.params import NetParams, InitialConfig, TrafficLightParams
from flow.core.params import VehicleParams, SumoCarFollowingParams
from flow.controllers.routing_controllers import GridRouter

from flow.scenarios.base_scenario import Scenario

from ilurl.loaders.routes import inflows2route
from ilurl.loaders.vtypes import get_vehicle_types

ILURL_HOME = os.environ['ILURL_HOME']

DIR = \
    f'{ILURL_HOME}/data/networks/'


def get_path(network_id, file_type):
    return \
        os.path.join(DIR, f'{network_id}/{network_id}.{file_type}.xml')


def get_generic_element(network_id, target, file_type='net',
                        ignore=None, key=None, child_key=None):
    """Parses the {network_id}.{file_type}.xml in search for target

    Usage:
    -----
    > # Returns a list of dicts representing the nodes
    > elements = get_generic_element('grid', 'junctions')
    """
    # Parse xml recover target elements
    file_path = get_path(network_id, file_type)
    elements = []

    if os.path.isfile(file_path):
        root = ET.parse(file_path).getroot()
        for elem in root.findall(target):
            if ignore not in elem.attrib:
                if key in elem.attrib:
                    elements.append(elem.attrib[key])
                else:
                    elements.append(elem.attrib)

                if child_key is not None:
                    elements[-1][f'{child_key}s'] = \
                        [chlem.attrib for chlem in elem.findall(child_key)]

    return elements


def get_routes(network_id):
    """Get routes as specified on Scenario

        routes must contain length and speed (max.)
        but those attributes belong to the lanes.

        parameters:
        ----------
            * network_id: string
            path data/networks/{network_id}/{network_id}.net.xml

        returns:
        -------
            * routes: list of dictionaries
            as specified at flow.scenarios.py

        specs:
        ------

        routes : dict
            A variable whose keys are the starting edge of a specific route, and
            whose values are the list of edges a vehicle is meant to traverse
            starting from that edge. These are only applied at the start of a
            simulation; vehicles are allowed to reroute within the environment
            immediately afterwards.

        reference:
        ----------
        flow.scenarios.base_scenario
    """
    # Parse xml to recover all generated routes
    routes = get_generic_element(network_id, 'vehicle/route',
                                 file_type='rou', key='edges')

    
    # unique routes as array of arrays
    routes = [rou.split(' ') for rou in set(routes)]

    # starting edges
    keys = {rou[0] for rou in routes}

    # match routes to it's starting edges
    routes = {k: [r for r in routes if k == r[0]] for k in keys}

    # convert to equipropable array of tuples: (routes, probability)
    routes = {k: [(r, 1 / len(rou)) for r in rou] for k, rou in routes.items()}

    return routes


def get_edges(network_id):
    """Get edges as specified on Scenario

        edges must contain length and speed (max.)
        but those attributes belong to the lanes.

        parameters:
        ----------
            * network_id: string
            path data/networks/{network_id}/{network_id}.net.xml

        returns:
        -------
            * edges: list of dictionaries
            as specified at flow.scenarios.py

        specs:
        ------
    edges : list of dict or None
        edges that are assigned to the scenario via the `specify_edges` method.
        This include the shape, position, and properties of all edges in the
        network. These properties include the following mandatory properties:

        * **id**: name of the edge
        * **from**: name of the node the edge starts from
        * **to**: the name of the node the edges ends at
        * **length**: length of the edge

        In addition, either the following properties need to be specifically
        defined or a **type** variable property must be defined with equivalent
        attributes in `self.types`:

        * **numLanes**: the number of lanes on the edge
        * **speed**: the speed limit for vehicles on the edge

        Moreover, the following attributes may optionally be available:

        * **shape**: the positions of intermediary nodes used to define the
          shape of an edge. If no shape is specified, then the edge will appear
          as a straight line.

        Note that, if the scenario is meant to generate the network from an
        OpenStreetMap or template file, this variable is set to None

        reference:
        ----------
        flow.scenarios.base_scenario
    """
    edges = get_generic_element(
        network_id, 'edge', ignore='function', child_key='lane')

    for e in edges:
        e['speed'] = max([float(lane['speed']) for lane in e['lanes']])
        e['length'] = max([float(lane['length']) for lane in e['lanes']])
        e['numLanes'] = len(e['lanes'])
        del e['lanes']
    return edges


def get_tls(network_id):
    """Queries the traffic light installed over network"""

    tls_nodes = [n for n in get_nodes(network_id)
                 if n['type'] == 'traffic_light']
    return tls_nodes

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

        if demand_type == 'lane':
            initial_config = InitialConfig(
                edges_distribution=tuple(get_routes(network_id).keys())
            )
            inflows = make_lane(network_id, horizon, initial_config)

        elif demand_type == 'switch':
            inflows = make_switch(network_id, horizon)
        else:
            raise ValueError(f'Unknown demand_type {demand_type}')

        # checks if route exists -- returning the path
        paths = inflows2route(
            network_id,
            inflows,
            get_routes(network_id),
            get_edges(network_id),
            distribution=demand_type,
            num_reps=num_reps,
            label=label
        )

        net = get_path(network_id, 'net')
        vtype = get_vehicle_types()
        scenarios = []
        for path in paths:
            net_params = NetParams(
                template={
                    'net': net,
                    'vtype': vtype,
                    'rou': [path]
                }
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
                 inflows_type='lane',
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
            if inflows_type == 'lane':
                inflows = make_lane(network_id, horizon, initial_config)
            elif inflows_type == 'switch':
                inflows = make_switch(network_id, horizon)

            else:
                raise ValueError(f'Unknown inflows_type {inflows_type}')


            net_params = NetParams(inflows,
                                   template=get_path(network_id, 'net'))

        if traffic_lights is None:
            prog_list = get_generic_element(network_id, 'tlLogic',
                                            child_key='phase')
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
        return get_generic_element(self.network_id, 'junction')

    def specify_edges(self, net_params):
        return get_edges(self.network_id)

    def specify_connections(self, net_params):
        return get_generic_element(self.network_id, 'connection')

    def specify_routes(self, net_params):
        return get_routes(self.network_id)

    def specify_types(self, net_params):
        return get_generic_element(self.network_id, 'type')


def make_lane(network_id, horizon, initial_config):
    inflows = InFlows()
    edges = get_edges(network_id)
    for eid in get_routes(network_id):
        # use edges distribution to filter routes
        if eid in initial_config.edges_distribution:
            edge = [e for e in edges if e['id'] == eid][0]

            num_lanes = edge['numLanes'] if 'numLanes' in edge else 1
            inflows.add(
                eid,
                'human',
                probability=0.2 * num_lanes,
                depart_lane='best',
                depart_speed='random',
                name=f'flow_{eid}',
                begin=1,
                end=horizon
            )

    return inflows

def make_switch(network_id, horizon):
    inflows = InFlows()
    edges = get_edges(network_id)
    switch = 900   # switches flow every 900 seconds
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
            prob = prob0 + 0.2 * num_lanes if (hr + num_lanes) % 2 == 1 else prob0
            print(f'{eid} {prob}')
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

