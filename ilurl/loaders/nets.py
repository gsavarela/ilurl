"""Low level XML queries to the networks data"""

__author__ = 'Guilherme Varela'
__date__ = '2020-01-30'
import os
from operator import itemgetter
from collections import OrderedDict
import json
import xml.etree.ElementTree as ET

ILURL_HOME = os.environ['ILURL_HOME']

DIR = \
    f'{ILURL_HOME}/data/networks/'

def get_path(network_id, file_type):
    rel_path = f'{network_id}/'
    filename = f'{network_id}.{file_type}.xml'
    return \
        os.path.join(DIR, rel_path, filename)


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
    """Get routes as specified on Network

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
        flow.networks.base
    """
    # Parse xml to recover all generated routes
    routes = get_generic_element(network_id, 'vehicle/route',
                                 file_type='rou', key='edges')

   
    # unique routes as array of arrays
    routes = [rou.split(' ') for rou in set(routes)]

    # starting edges
    keys = {rou[0] for rou in routes}

    # match routes to it's starting edges
    routes = {k: sorted([r for r in routes if k == r[0]])
              for k in sorted(keys)}

    # convert to equipropable array of tuples:
    # (routes, probability)
    routes = OrderedDict({
        k: [(r, 1 / len(rou)) for r in rou]
        for k, rou in routes.items()
    })
    return routes


def get_nodes(network_id):
    return get_generic_element(network_id, 'junction')

def get_connections(network_id):
    return get_generic_element(network_id, 'connection')

def get_types(network_id):
    return get_generic_element(network_id, 'type')

def get_edges(network_id):
    """Get edges as specified on Network

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
        attributes in `.types`:

        * **numLanes**: the number of lanes on the edge
        * **speed**: the speed limit for vehicles on the edge

        Moreover, the following attributes may optionally be available:

        * **shape**: the positions of intermediary nodes used to define the
          shape of an edge. If no shape is specified, then the edge will appear
          as a straight line.

        Note that, if the scenario is meant to generate the network from an
        OpenStreetMap or template file, this variable is set to None

        Reference:
        ----------
        flow.networks.base
    """
    edges = get_generic_element(
        network_id, 'edge', ignore='function', child_key='lane')

    for e in sorted(edges, key=itemgetter('id')):
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


def get_logic(network_id):
    res = get_generic_element(network_id, 'tlLogic', child_key='phase')
    res = sorted(res, key=itemgetter('id'))
    return res


def get_tls_custom(network_id, baseline=False):
    """ Loads TLS settings (cycle time and programs) from tls_config.json file.

        Parameters
        ----------
        network_name : string
        network id

        Return
        ----------
        cycle_time: int
        the cycle time for the TLS system

        programs: dict
        the programs (timings) for the TLS system
        defines the actions that the agent can pick

    """
    tls_config_file = f'{DIR}/{network_id}/tls_config.json'

    if not os.path.isfile(tls_config_file):
        raise FileNotFoundError("tls_config.json file not provided "
                "for network {0}.".format(network.network_id))

    with open(tls_config_file, 'r') as f:
        cfgs = json.load(f)

    # cfgs = cfgs['actuated'] if baseline else cfgs['rl']
    if 'cycle_time' not in cfgs:
        raise KeyError(f'Missing `cycle_time` key in tls_config.json')
    else:
        # Setup cycle time.
        cycle_time = cfgs.pop('cycle_time')

    # Setup programs.
    if baseline:
        programs = cfgs
    else:
        programs = {}
        for tls_id, data in cfgs.items():

            # Setup actions (programs) for given TLS.
            programs[tls_id] = \
                {int(action): data[action] for action in data.keys()}

    return cycle_time, programs

