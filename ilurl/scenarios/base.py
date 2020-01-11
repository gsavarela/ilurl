"""This module acts as a wrapper for scenarios generated from network data"""
__author__ = 'Guilherme Varela'
__date__ = '2020-01-10'

import os
import xml.etree.ElementTree as ET
from collections import defaultdict

# Vehicle definition stuff
from flow.controllers import GridRouter
from flow.core.params import SumoCarFollowingParams, VehicleParams
# InFlows
from flow.core.params import InFlows
# Network related parameters
from flow.core.params import NetParams, InitialConfig, TrafficLightParams

from flow.scenarios import Scenario


ILURL_HOME = os.environ['ILURL_HOME']

DIR = \
    f'{ILURL_HOME}/data/networks/'


def get_path(network_id, file_type):
    return \
        os.path.join(DIR, f'{network_id}/{network_id}.{file_type}.xml')


def get_tl_logic(network_id):
    # Parse xml to recover all programs
    tls_path = get_path(network_id, 'net')
    prog_list = []

    if os.path.isfile(tls_path):
        root = ET.parse(tls_path).getroot()
        for prog in root.findall('tlLogic'):
            prog_list.append(prog.attrib)
            prog_list[-1]['phases'] = \
                [phase.attrib for phase in prog.findall('phase')]

    return prog_list


def get_generic_element(network_id, target,
                        file_type='net', ignore=None, key=None):
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

    return elements


def get_routes(network_id):
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


class BaseScenario(Scenario):
    """This class leverages on specs created by SUMO"""

    def __init__(self,
                 network_id,
                 horizon=360,
                 inflows=None,
                 vehicles=None,
                 net_params=None,
                 initial_config=None,
                 traffic_lights=None):

        self.network_id = network_id
        #TODO: check vtype
        if vehicles is None:
            vehicles = VehicleParams()
            vehicles.add(
                veh_id="human",
                routing_controller=(GridRouter, {}),
                car_following_params=SumoCarFollowingParams(
                    min_gap=2.5,
                    decel=7.5,  # avoid collisions at emergency stops
                ),
            )

        if net_params is None:
            if not inflows:
                inflows = InFlows()
                for edge in get_routes(network_id):
                    inflows.add(
                        edge,
                        'human',
                        probability=0.2,
                        depart_lane='best',
                        depart_speed='random',
                        name=f'flow_{edge}',
                        begin=1,
                        end=0.9 * horizon
                    )
            net_params = NetParams(
                inflows,
                template=get_path(network_id, 'net')
            )

        if initial_config is None:
            initial_config = InitialConfig(
                edges_distribution=get_routes(network_id).keys()
            )

        if traffic_lights is None:
            prog_list = get_tl_logic(network_id)
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
        return get_generic_element(self.network_id, 'edge', ignore='function')

    def specify_connections(self, net_params):
        return get_generic_element(self.network_id, 'connection')

    def specify_routes(self, net_params):
        return get_routes(self.network_id)

    def specify_types(self, net_params):
        return get_generic_element(self.network_id, 'type')



if __name__ == '__main__':
    print(get_routes('intersection'))

