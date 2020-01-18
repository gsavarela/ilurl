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

import os
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString

from numpy import arange
from numpy.random import rand, choice

XML_PATH = f"{os.environ['ILURL_HOME']}/data/networks/"

def is_route(network_id, horizon, distribution='lane'):
    """Verifies if there's a route config with those specs
    """
    p = _get_path(network_id, horizon, distribution)
    if os.path.isfile(p):
        return p
    else:
        return None

def _get_path(network_id, horizon, distribution='lane'):
    path = f'{XML_PATH}{network_id}/{network_id}'

    if distribution not in ('lane', 'switch'):
        raise ValueError(f'distribution not implemented {distribution}')
    else:
        x = 'l' if distribution == 'lane' else 'w'

    path = f'{path}.{horizon}.{x}.rou.xml'

    return path

def inflows2route(network_id, inflows,
                  routes, edges, distribution='lane'):
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
    # path = f'{XML_PATH}{network_id}/{network_id}'
    root = ET.Element(
        'routes',
        attrib={
            'xmlns:xsi':
            "http://www.w3.org/2001/XMLSchema-instance",
            'xsi:noNamespaceSchemaLocation':
            "http://sumo.dlr.de/xsd/routes_file.xsd"}
    )
    # if distribuition not in ('lane', 'switch'):
    #     raise ValueError(f'distribution not implemented {distribution}')
    # else:
    #     x = 'l' if distribution == 'lane' else 'w'

    _inflows = sorted(inflows.get(), key=lambda d: d['begin'])
    horizon = max([int(ii['end']) for ii in _inflows])
    path = _get_path(network_id, horizon, distribution)
    vehicles = []
    veh_id = 1
    for inflow in _inflows:
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
 
            if rand() < prob:
                # emit a vehicle
                idx = choice(flw_indexes, p=flw_probs)
                rou = flw_routes[idx]

                speed = float(max_speed) * rand()
                
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
    return path
