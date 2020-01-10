"""This module defines a Flow scenario from network data"""

__author__ = 'Guilherme Varela'
__date__ = '2020-01-10'
import os
import xml.etree.ElementTree as ET
from collections import defaultdict

from flow.core.params import NetParams, InitialConfig, TrafficLightParams
from flow.scenarios import Scenario

DIR = \
    '/Users/gsavarela/Work/py/ilu/ilurl/data/networks/'


def get_routes():
    # Parse xml to recover all generated routes
    rou_path = f'{DIR}/intersection/intersection.rou.xml'
    root = ET.parse(rou_path).getroot()
    route_list = []
    for it in root.findall('vehicle/route'):
        route_list.append(it.attrib['edges'])

    # Convert routes into a dictionary
    route_dict = defaultdict(list)
    for rou in set(route_list):
        rou = rou.split(' ')
        key = rou[0]
        route_dict[key].append(rou)

    # If there is more than one route starting from an edge
    # then specify the probability -- equiprobable
    specify_routes_dict = {}
    for start, routes in route_dict.items():
        n = len(routes)
        specify_routes_dict[start] = \
            [(rou, 1 / n) for rou in routes]
    return specify_routes_dict


class IntersectionScenario(Scenario):
    """This class leverages on specs created by SUMO"""

    def __init__(self,
                 name,
                 vehicles,
                 net_params=None,
                 initial_config=None,
                 traffic_lights=None):

        if net_params is None:
            net_params = NetParams(
                template={
                    'net': os.path.join(DIR, 'intersection/intersection.net.xml'),
                },
            )

        if initial_config is None:
            initial_config = InitialConfig(
                edges_distribution=get_routes().keys()
            )

        if traffic_lights is None:
            # TODO: collect from generated data
            traffic_lights = TrafficLightParams()

        super(IntersectionScenario, self).__init__(
                 name,
                 vehicles,
                 net_params,
                 initial_config=initial_config,
                 traffic_lights=traffic_lights
        )

    def specify_routes(self, net_params):
        # TODO: Parse route file and extract the routes
        return get_routes()


if __name__ == '__main__':
    print(get_routes())
