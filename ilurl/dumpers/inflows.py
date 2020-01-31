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
    vehicles = []
    veh_id = 1

    random.seed(0)
    for i, inflow in enumerate(inflows_sorted):
        start = int(inflow['begin'])
        finish = int(inflow['end'])
        prob = inflow['probability']

        flw_routes, flw_probs = zip(*routes[inflow['edge']])
        flw_indexes = arange(len(flw_routes))

        edge = [e
                for e in edges if e['id'] == inflow['edge']][0]

        if inflow['departSpeed'] == 'random':
            max_speed = edge['speed']
        else:
            raise NotImplementedError

        for depart in range(start, finish):
 
            if random.random() < prob:
                # emit a vehicle
                idx = random.choice(flw_indexes, p=flw_probs)
                rou = flw_routes[idx]

                speed = float(max_speed) * random.random()
                
                vehicle = ET.SubElement(
                    root,
                    'vehicle',
                    attrib={
                        'id': f'{veh_id:04d}',
                        'type': inflow['vtype'],
                        'depart': str(depart),
                        'departSpeed': f'{speed:0.2f}',  # TODO: check
                        'departPos': '0.0', # TODO: check if it should set edge starts
                    })

                route = ET.SubElement(
                    vehicle,
                    'route',
                    attrib={
                        'edges': ' '.join(rou)
                    }
                )
                veh_id += 1

    dom = parseString(ET.tostring(root))
    etree = ET.ElementTree(ET.fromstring(dom.toprettyxml()))
    etree.write(path)
    paths.append(path)

    return paths
