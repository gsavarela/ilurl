"""This module handles low-level route routines

    PROBLEM:
    -------
    Experiments with inflows generate such a wide variety of returns that
    Komogorov Smirnov tests assign that they are different.

    SOLUTION:
    --------
    Make create per vehicle trips or static route files as supported by
    http://sumo.dlr.de/xsd/routes_file.xsd

"""

__author__ = 'Guilherme Varela'
__date__ = '20200117'

import time
import os
import glob
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString
from collections import defaultdict

from numpy import arange
from numpy import random

from ilurl.loaders.nets import get_routes, get_edges


XML_PATH = f"{os.environ['ILURL_HOME']}/data/networks/"

def inflows_path(network_id, horizon, distribution='lane', n=0):
    path = f'{XML_PATH}{network_id}/{network_id}'

    if distribution not in ('lane', 'switch'):
        raise ValueError(f'distribution not implemented {distribution}')
    else:
        x = 'l' if distribution == 'lane' else 'w'

    path = f'{path}.{n}.{horizon}.{x}.rou.xml'

    return path


def inflows_paths(network_id, horizon, distribution='lane'):
    path = f'{XML_PATH}{network_id}/{network_id}'

    if distribution not in ('lane', 'switch'):
        raise ValueError(f'distribution not implemented {distribution}')
    else:
        x = 'l' if distribution == 'lane' else 'w'

    paths = glob.glob(f'{path}.[0-9].{horizon}.{x}.rou.xml')

    return paths


def inflows_dump(network_id, inflows,
                 distribution='lane', label=None):
    """

    EXAMPLE:
    -------
    > vim data/networks/intersection/intersection.w.rou.xml
      <routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
           <vehicle depart="5" departSpeed="27.76" id="0001" type="human">
                   <route edges="309265401 238059328" />
           </vehicle>
           <vehicle depart="12" departSpeed="5.27" id="0002" type="human">
                   <route edges="309265401 238059328" />
           </vehicle>
     </routes>
    """
    # load network data
    routes = get_routes(network_id)
    edges = get_edges(network_id)
    # sort by begin
    inflows_sorted = sorted(inflows.get(), key=lambda d: d['begin'])
    horizon = max([int(ii['end']) for ii in inflows_sorted])
    paths = []

    root = ET.Element(
        'routes',
        attrib={
            'xmlns:xsi':
            "http://www.w3.org/2001/XMLSchema-instance",
            'xsi:noNamespaceSchemaLocation':
            "http://sumo.dlr.de/xsd/routes_file.xsd"}
    )

    path = inflows_path(network_id, horizon, distribution, label)
    vehicles = [] # trips elements
    veh_id = 1

    # random.seed(0)
    # an array of dictionaries, keys are edgeids
    # values are tuple (routeids, probabilities)
    edges2routes = defaultdict(list)
    routes2edges = {}
    # first define route elements
    for i, inflows in enumerate(inflows_sorted):
        edge_id = inflows['edge']
        if not edges2routes[edge_id]:
            for j, edges_probs in enumerate(routes[edge_id]):
                route_id = f'route{edge_id}_{j}'
                ET.SubElement(
                    root,
                    'route',
                    attrib={
                        'id': route_id,
                        'edges': ' '.join(edges_probs[0]),
                    }
                )
                edges2routes[edge_id].append((route_id, edges_probs[1]))
                routes2edges[route_id] = ' '.join(edges_probs[0])

    #  define trips indexed by route_ids
    #  array of trips attributes
    trips = []
    for i, inflows in enumerate(inflows_sorted):
        start = int(inflows['begin'])
        finish = int(inflows['end'])
        # emission probability from this edge
        prob = inflows['probability']

        edge_id = inflows['edge']

        route_ids, route_ps = zip(*edges2routes[edge_id])
        edge = [e for e in edges if e['id'] == edge_id][0]

        if inflows['departSpeed'] == 'random':
            max_speed = edge['speed']
        else:
            raise NotImplementedError

        for depart in range(start, finish):
 
            if random.random() < prob:
                # emit a vehicle
                idx = random.choice(len(route_ids), p=route_ps)
                route_id = route_ids[idx]

                speed = round(float(max_speed) * random.random(), 2)
                
                trips.append({
                    'type': inflows['vtype'],
                    'depart': depart,
                    'departSpeed': f'{speed:0.2f}',  # todo: check
                    'departPos': '0.0',    # TODO: check if it should set edge starts
                    'route': route_id,
                })

    # sort trips by depart times
    trips = sorted(trips, key=lambda x: x['depart'])
    for nt, trip in enumerate(trips):
        trip['id'] = f'{nt:05d}'
        trip['depart'] = f"{trip['depart']:0.1f}"

        route_id = trip.pop('route')
        vehicle = ET.SubElement(
            root,
            'vehicle',
            attrib=trip
        )
        route = ET.SubElement(
            vehicle,
            'route',
            attrib={
                'route': route_id,
                'edges': routes2edges[route_id]
                    }
        )
    dom = parseString(ET.tostring(root))
    etree = ET.ElementTree(ET.fromstring(dom.toprettyxml()))
    etree.write(path)
    paths.append(path)

    return paths
