"""This module defines a Flow scenario from network data"""

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

TEMPLATE_PATH = \
    os.path.join(DIR, 'grid/grid.net.xml')


def get_routes():
    # Parse xml to recover all generated routes
    rou_path = f'{DIR}/grid/grid.rou.xml'
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


class GridScenario(Scenario):
    """This class leverages on specs created by SUMO"""

    def __init__(self,
                 name,
                 horizon=360,
                 vehicles=None,
                 net_params=None,
                 initial_config=None,
                 traffic_lights=None):

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
            inflows = InFlows()
            for edge in get_routes():
                inflows.add(
                    edge,
                    'human',
                    probability=0.05,
                    depart_lane='best',
                    depart_speed='random',
                    name=f'flow_{edge}',
                    begin=1,
                    end=0.9 * horizon
                )

            net_params = NetParams(
                inflows,
                template=TEMPLATE_PATH
            )

        if initial_config is None:
            initial_config = InitialConfig(
                edges_distribution=get_routes().keys()
            )

        if traffic_lights is None:
            prog_list = get_tl_logic()
            if prog_list:
                traffic_lights = TrafficLightParams(baseline=False)
                for prog in prog_list:
                    prog_id = prog.pop('id')
                    prog['tls_type'] = prog.pop('type')
                    prog['programID'] = int(prog.pop('programID')) + 1
                    traffic_lights.add(prog_id, **prog)
            else:
                traffic_lights = TrafficLightParams(baseline=False)

        super(GridScenario, self).__init__(
                 name,
                 vehicles,
                 net_params,
                 initial_config=initial_config,
                 traffic_lights=traffic_lights
        )

    def specify_routes(self, net_params):
        return get_routes()


def get_tl_logic():
    # Parse xml to recover all programs
    tls_path = f'{DIR}/grid/grid.net.xml'
    prog_list = []

    if os.path.isfile(tls_path):
        root = ET.parse(tls_path).getroot()
        for prog in root.findall('tlLogic'):
            prog_list.append(prog.attrib)
            prog_list[-1]['phases'] = \
                [phase.attrib for phase in prog.findall('phase')]

    return prog_list

if __name__ == '__main__':
    #print(get_routes())
    print(get_tl_logic())

