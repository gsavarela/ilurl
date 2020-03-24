"""This module acts as a wrapper for networks generated from network data"""
__author__ = 'Guilherme Varela'
__date__ = '2020-01-10'

import os
import operator as op
from itertools import groupby
import pdb

# Network related parameters
from flow.core.params import InitialConfig, TrafficLightParams
from flow.core.params import VehicleParams, SumoCarFollowingParams
from flow.controllers.routing_controllers import GridRouter

from flow.networks.base import Network as FlowNetwork

from ilurl.core.params import InFlows, NetParams
from ilurl.loaders.nets import (get_routes, get_edges, get_path,
                                get_logic, get_connections, get_nodes,
                                get_types)

class Network(FlowNetwork):
    """This class leverages on specs created by SUMO"""

    @classmethod
    def make(cls, network_id, horizon, demand_type, num_reps,
             label=None, initial_config=None):
        """Builds a new network from rou.xml file -- the resulting
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
        *   network(s): ilurl.network.Network or list
            n = 0  attempts to load one network,
            n > 0  attempts to load n+1 networks returning a list
        """

        networks = []

        for nr in range(num_reps):
            label1 = f'{nr}.{label}' if label and num_reps > 1 else nr
            net_params = NetParams.from_template(
                network_id, horizon, demand_type, label=label1,
                initial_config=initial_config
            )

            networks.append(
                Network(
                    network_id,
                    horizon,
                    net_params,
                    initial_config=initial_config,
                    vehicles=VehicleParams()
                )
            )

        ret = networks[0] if num_reps == 1 else networks
        return ret

    @classmethod
    def load(cls, network_id, route_path):
        """Attempts to load a new network from rou.xml and 
        vtypes.add.xml -- if it fails will call `make`
        the resulting vehicle trips will be stochastic use 
        it for training

        Params:
        ------
        *   network_id: string
            identification of net.xml file, ex: `intersection`
        *   horizon: integer
            latest depart time
        *   demand_type: string
            string
        *   label: string
            e.g `eval, `train` or `test`
        Returns:
        -------
        *   network(s): ilurl.network.Network or list
            n = 0  attempts to load one network,
            n > 0  attempts to load n+1 networks returning a list
        """
        net_params = NetParams.load(network_id, route_path)

        horizon = int(route_path.split('.')[-4])

        network = Network(
            network_id,
            horizon,
            net_params,
            vehicles=VehicleParams()
        )
        return network

    def __init__(self,
                 network_id,
                 horizon=360,
                 net_params=None,
                 vehicles=None,
                 demand_type='lane',
                 insertion_probability=0.1,
                 initial_config=None,
                 traffic_lights=None):


        """Builds a new network from inflows -- the resulting
        vehicle trips will be stochastic use it for training"""
        self.network_id = network_id

        if initial_config is None:
            initial_config = InitialConfig(
                edges_distribution=tuple(get_routes(network_id).keys())
            )

        if net_params is None:
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

            inflows = InFlows(network_id,
                              horizon,
                              demand_type,
                              insertion_probability=insertion_probability,
                              initial_config=initial_config)

            net_params = NetParams(inflows,
                                   template=get_path(network_id, 'net'))

        if traffic_lights is None:
            # Converts a static program into a reinforcement learning
            # program.
            programs = get_logic(network_id)
            if programs:
                traffic_lights = TrafficLightParams(baseline=False)
                for prog in programs:
                    prog_id = prog.pop('id')
                    prog['tls_type'] = prog.pop('type')
                    prog['programID'] = int(prog.pop('programID')) + 1
                    traffic_lights.add(prog_id, **prog)
            else:
                traffic_lights = TrafficLightParams(baseline=False)

        super(Network, self).__init__(
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
        """Connections bind edges' lanes to one another at junctions
         
            DEF:
            ----
            definitions follow the standard
            *Name   :Type
                Description

            *from   :edge id (string)
                The ID of the incoming edge at which the connection
                begins
            *to     :edge id (string)
                The ID of the outgoing edge at which the connection ends
            *fromLane   :index (unsigned int)
                The lane of the incoming edge at which the connection
                begins
            *toLane     :index (unsigned int)
                The lane of the outgoing edge at which the connection ends
            *via    :lane id (string)
                The id of the lane to use to pass this connection across the junction
            *tl     :traffic light id (string
                The id of the traffic light that controls this connection; the attribute is missing if the connection is not controlled by a traffic light
            *linkIndex  :index (unsigned int
                The index of the signal responsible for the connection within the traffic light; the attribute is missing if the connection is not controlled by a traffic light
            *dir:enum
                ("s" = straight, "t" = turn, "l" = left, "r" = right, "L" = partially left, R = partially right, "invalid" = no direction
            The direction of the connection
            *state:enum
                ("-" = dead end, "=" = equal, "m" = minor link, "M" = major link, traffic light only: "O" = controller off, "o" = yellow flashing, "y" = yellow minor link, "Y" = yellow major link, "r" = red, "g" = green minor, "G" green major
            The state of the connection

            REF:
            ----
            http://sumo.sourceforge.net/userdoc/Networks/SUMO_Road_Networks.html
        """
        return get_connections(self.network_id)


    def specify_routes(self, net_params):
        return get_routes(self.network_id)

    def specify_types(self, net_params):
        return get_types(self.network_id)

    @property
    def approaches(self):
        """Returns the incoming approaches for a traffic light junction

        Params:
        ------
        * nodeid: string
            a valid nodeid in self.nodes

        Returns:
        ------
        * approaches: dict<string, list<string>>
            list of mappings from node_id -> incoming edge ids
        
        Usage:
        -----
         > network.approaches
         > {'247123161': ['-238059324', '-238059328', '309265401', '383432312']}
        DEF:
        ---
        A roadway meeting at an intersection is referred to as an approach. At any general intersection, there are two kinds of approaches: incoming approaches and outgoing approaches. An incoming approach is one on which cars can enter the intersection.

        REF:
        ---
            * Wei et al., 2019
            http://arxiv.org/abs/1904.08117
        """
        
        if not hasattr(self, '_cached_approaches'):
            # define approaches
            self._cached_approaches = {
                nid: [e['id']
                          for e in self.edges if e['to'] == nid]
                for nid in self.tls_ids
            }
        return self._cached_approaches

    @property
    def phases(self):
        """Returns a nodeid x sets of non conflicting movement patterns.
            The sets are index by integers and the moviment patterns are
            expressed as lists of approaches. We consider only incoming
            approaches to be controlled by phases.
            
        Returns:
        ------
        * phases: dict<string,dict<int, dict<string, obj>>>
            keys: nodeid, phase_id, 'states', 'components'
            

        Usage:
        -----
        > network.states
        > {'gneJ2':
            ['GGGgrrrrGGGgrrrr', 'yyygrrrryyygrrrr', 'rrrGrrrrrrrGrrrr',
            'rrryrrrrrrryrrrr', 'rrrrGGGgrrrrGGGg', 'rrrryyygrrrryyyg',
            'rrrrrrrGrrrrrrrG', 'rrrrrrryrrrrrrry']}

        > network.phases
        > {'gneJ2':
            {0: {'components':
                    [('-gneE8', [0, 1, 2]), ('gneE12', [0, 1, 2])],
                    'states': ['GGGgrrrrGGGgrrrr']
                },
             1: {'components':
                     [('-gneE8', [2]), ('gneE12', [2])],
                  'states': ['yyygrrrryyygrrrr', 'rrrGrrrrrrrGrrrr',
                             'rrryrrrrrrryrrrr']
                },
             2: {'components':
                     [('gneE7', [0, 1, 2]), ('-gneE10', [0, 1, 2])],
                 'states': ['rrrrGGGgrrrrGGGg']
                 },
             3: {'components':
                     [('gneE7', [2]), ('-gneE10', [2])], 
                 'states': ['rrrryyygrrrryyyg', 'rrrrrrrGrrrrrrrG',
                            'rrrrrrryrrrrrrry']
                }
             }
           }
        DEF:
        ---
        A phase is a combination of movement signals which are
        non-conflicting. The relation from states to phases is
        such that phases "disregards" yellow configurations 
        usually num_phases = num_states / 2

        REF:
        ---
        Wei et al., 2019
        http://arxiv.org/abs/1904.08117
        """

        # TODO: include duration
        if not hasattr(self, '_cached_phases'):
            self._cached_phases = {}

            def fn(x, n):
                return x.get('tl') == n and 'linkIndex' in x

            for nid in self.tls_ids:
                # green and yellow are considered to be one phase
                self._cached_phases[nid] = {}
                connections = [c for c in self.connections if fn(c, nid)]
                states = self.states[nid]
                links = {
                    int(cn['linkIndex']):
                        (cn['from'], int(cn['fromLane']))
                    for cn in connections if 'linkIndex' in cn
                }
                i = 0
                components = {}
                for state in states:
                    # components: linkIndex, 0-1, edge_id, lane
                    components = {
                        (lnk,) + edge_lane
                        for lnk, edge_lane in links.items()
                        if state[lnk] in ('G','g')
                    }
                    # adds components if they don't exist
                    if components:
                        found = False
                        # sort by link, edge_id
                        components = \
                            sorted(components, key=op.itemgetter(0, 1))

                        # groups lanes by edge_ids and states
                        components = \
                            [(k, list({l[-1] for l in g}))
                             for k, g in groupby(components,
                                                 key=op.itemgetter(1))]                         
                        for j in range(0, i + 1):
                            if j in self._cached_phases[nid]:
                                # same edge_id and lanes
                                _component =  \
                                    self._cached_phases[nid][j]['components']
                                found = \
                                    components == _component

                                if found:
                                    self._cached_phases[nid][j]['states'].append(state)
                        if not found:
                            self._cached_phases[nid][i] = \
                                {'components': components,
                                 'states': [state]}
                            i += 1
                    else:
                        # states only `r` and `y`
                        self._cached_phases[nid][i-1]['states'].append(state)
        return self._cached_phases

    @property
    def states(self):
        """states wrt to programID = 1 for traffic light nodes

        Returns:
        -------
            * states: dict<string, list<string>>

        Usage:
        ------
        > network.states
        > {'247123161': ['GGrrrGGrrr', 'yyrrryyrrr', 'rrGGGrrGGG', 'rryyyrryyy']}O

        REF:
        ----
            http://sumo.sourceforge.net/userdoc/Simulation/Traffic_Lights.html
        """
        if not hasattr(self, '_cached_states'):
            configs = self.traffic_lights.get_properties()

            def fn(x):
                return x['type'] == 'static' and x['programID'] == 1

            self._cached_states = {
                 nid: [p['state']
                       for p in configs[nid]['phases']
                       if fn(configs[nid])]
                 for nid in self.tls_ids}
        return self._cached_states

    @property
    def durations(self):
        """Gives the times or durations in seconds for each of the states
        on the default programID = 1

        Returns:
        -------
            * durations: list<int>
                a list of integer representing the time in seconds, each
                state is alloted on the default static program.

        Usage:
        ------
        > network.durations
        > {'247123161': [39, 6, 39, 6]}

        REF:
        ----
            http://sumo.sourceforge.net/userdoc/Simulation/Traffic_Lights.html
        """
        if not hasattr(self, '_cached_durations'):
            configs = self.traffic_lights.get_properties()

            def fn(x):
                return x['type'] == 'static' and x['programID'] == 1

            self._cached_durations = {
                 nid: [int(p['duration']) for p in configs[nid]['phases']
                       if fn(configs[nid])]
                 for nid in self.tls_ids}
        return self._cached_durations

    @property
    def tls_ids(self):
        """List of nodes which are also traffic light signals

        Returns
        -------
            * nodeids: list<string>
        Usage:
        -----
        # intersection
        > network.tls_ids
        ['247123161']

        """
        if not hasattr(self, '_cached_tls_ids'):
            self.cached_tls_ids = \
                [n['id'] for n in self.nodes if n['type'] == 'traffic_light']
        return self.cached_tls_ids
    
