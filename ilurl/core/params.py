"""Objects that define the various meta-parameters of an experiment."""
__author__ = 'Guilherme Varela'
__date__ = '2020-01-30'
import math

import flow.core.params as flow_params

from collections import namedtuple
from ilurl.core.ql.reward import REWARD_TYPES
from ilurl.core.ql.choice import CHOICE_TYPES

from ilurl.dumpers.inflows import inflows_dump
from ilurl.loaders.nets import get_edges, get_routes, get_path
from ilurl.loaders.vtypes import get_vehicle_types

# number of phases is fixed
NUM_PHASES = 2
STATE_FEATURES = ('speed', 'count', 'flow', 'queue')
''' Bounds : namedtuple
        provide the settings to describe discrete variables ( e.g actions ). Or
        create discrete categorizations from continous variables ( e.g states)

    * rank: int
        Number of variable dimensions

    * depth: int
        Number of categories

'''
Bounds = namedtuple('Bounds', 'rank depth')
''' Rewards : namedtuple
        Settings needed to perform reward computation

    * type: string
        A reward computation type in REWARD_TYPES

    * costs: tuple or None
        A tuple having size states depth representing
        the cost for each speed category. Larger is worse.

'''
Rewards = namedtuple('Rewards', 'type costs')

ADDITIONAL_PARAMS = {
    # every `switch`seconds concentrate flow in one direction
     "switch": 900
}

class QLParams:
    """Base Q-learning parameters

      Provides also common behaviour functions
      for environments and rewards (e.g categorize_state,
      split_state)
    """

    def __init__(
            self,
            epsilon=3e-2,
            alpha=5e-2,
            gamma=0.95,
            c=2,
            initial_value=0,
            max_speed=35,
            rewards={
                'type': 'weighted_average',
                'costs': None
            },
            num_traffic_lights=1,
            states=('speed', 'count'),
            actions=('fast_slow_green', ),
            choice_type='eps-greedy'
    ):
        """Instantiate base traffic light.

        PARAMETERS
        ----------
        * epsilon: is the chance to adopt a random action instead of
                  a greedy action [1].

        * alpha: is the learning rate the weight given to new
                 knowledge [1].

        * gamma: is the discount rate for value function [1].

        * c: upper confidence bound (ucb) exploration constant.
        * rewards: namedtuple
                    see above
        * states: namedtuple
            Create discrete categorizations from continous variables
            ( e.g states )
            SEE Bounds above
            * rank: int
                Number of variable dimensions

            * depth: int
                Number of categories
        * actions: namedtuple
            SEE Bounds above
            * rank: int
                Number of variable dimensions

            * depth: int
                Number of categories
                    see above

        REFERENCES:
        ----------
            [1] Sutton et Barto, Reinforcement Learning 2nd Ed 2018

        """
        kwargs = locals()
        if alpha <= 0 or alpha >= 1:
            raise ValueError('''The ineq 0 < alpha < 1 must hold.
                    got alpha = {}'''.format(alpha))

        if gamma <= 0 or gamma > 1:
            raise ValueError('''The ineq 0 < gamma <= 1 must hold.
                    got gamma = {}'''.format(gamma))

        if epsilon <= 0 or epsilon >= 1:
            raise ValueError('''The ineq 0 < epsilon < 1 must hold.
                    got epsilon = {}'''.format(epsilon))

        if choice_type not in CHOICE_TYPES:
            raise ValueError(
                f'''Choice type should be in {CHOICE_TYPES} got {choice_type}'''
            )


        for attr, value in kwargs.items():
            if attr not in ('self', 'states', 'actions', 'rewards'):
                setattr(self, attr, value)

        if 'states' in kwargs:
            states_tuple = kwargs['states']
            for name in states_tuple:
                if name not in STATE_FEATURES:
                    raise ValueError(f'''
                        {name} must be in {STATE_FEATURES}
                    ''')
            self.set_states(states_tuple)

        if 'actions' in kwargs:
            actions_tuple = kwargs['actions']
            self.set_actions(actions_tuple)

        rewards = kwargs['rewards']
        if 'type' not in rewards:
            raise ValueError('''``type` must be provided in reward types''')

        elif rewards['type'] not in REWARD_TYPES:
            raise ValueError(f'''
                Rewards must be in {REWARD_TYPES} got {rewards['type']}
            ''')
        elif rewards['type'] == 'costs':
            if rewards['costs'] is None:
                raise ValueError(
                    '''Cost must not be None each state depth (tier)
                      must have a cost got {} {} '''.format(
                        self.state.depth, len(rewards['costs'])))
            elif len(rewards['costs']) != self.states.depth:
                raise ValueError('''Cost each state depth (tier)
                      must have a cost got {} {} '''.format(
                    self.state.depth, len(rewards['costs'])))
        self.set_rewards(rewards['type'], rewards['costs'])

    def set_states(self, states_tuple):
        self.states_labels = states_tuple
        rank = self.num_traffic_lights * len(states_tuple) * NUM_PHASES
        depth = 3
        self.states = Bounds(rank, depth)

    def set_actions(self, actions_tuple):
        self.actions_labels = actions_tuple
        rank = self.num_traffic_lights * len(actions_tuple)
        depth = 2
        self.actions = Bounds(rank, depth)

    def set_rewards(self, type, costs):
        self.rewards = Rewards(type, costs)

    def categorize_space(self, observation_space):

        labels = list(self.states_labels)
        categorized_space = []
        # first loop is for intersections
        for inters_space in observation_space:
            # second loop is for phases
            categorized_intersections = []
            for phase_space in inters_space:
                # third loop is for variables
                categorized_phases = []
                for i, label in enumerate(labels):
                    val = phase_space[i]
                    category = getattr(self, f'_categorize_{label}')(val)
                    categorized_phases.append(category)
                categorized_intersections.append(categorized_phases)
            categorized_space.append(categorized_intersections)
            
        
        return categorized_space

    def split_space(self, observation_space):
        """Splits different variables into tuple
        
        Params:
            * observation_space: list of lists
            nested 2 level list such that;
            The second level represents it's phases; e.g
            north-south and east-west. And the last level represents
            the variables withing labels e.g `speed` and `count`.

        Returns:
        -------
            * flatten space
            
        Example:
        -------
        > observation_space = [[13.3, 2.7], [15.7, 1.9]]
        > splits = split_space(observation_space)
        > splits
        > [[13.3, 15.7], [2.7, 1.9]]

        """

        num_intersections = len(observation_space)
        num_phases = 2
        num_labels = len(self.states_labels)

        splits = []
        for inters_space in observation_space:
            for label in range(num_labels):
                components = []
                for comp in range(num_phases):
                    components.append(inters_space[comp][label])
                splits.append(components)

        return splits

    def flatten_space(self, observation_space):
        """Linearizes hierarchial state representation
        
        Params:
            * observation_space: list of lists
            nested 2 level list such that;
            The second level represents it's phases; e.g
            north-south and east-west. And the last level represents
            the variables withing labels e.g `speed` and `count`.

        Returns:
        -------
            * flattened_space: a list
            
        Example:
        -------
        > observation_space = [[[13.3, 2.7], [15.7, 1.9]]]
        > splits = split_space(observation_space)
        > splits
        > [[13.3, 15.7], [2.7, 1.9]]

        """
        flattened = [obs_value for inters in observation_space
                     for phases in inters for obs_value in phases]

        return flattened

    def _categorize_speed(self, speed):
        """Converts a float speed into a category
        
           Segregates into 3 categories estimated
           from environment analysis/hist
        """
        # intersection
        # if speed >= 2.2:  # hightest 25%
        #     return 2
        # elif speed <= 1.88:  # lowest 25%
        #     return 0
        # else:
        #     return 1

        # intersection
        if speed >= 2.15:  # hightest 25%
            return 2
        elif speed <= 2.06:  # lowest 25%
            return 0
        else:
            return 1

    def _categorize_count(self, count):
        """Converts a int count into a category
        
           Segregates into 3 categories estimated
           from environment analysis/hist
        """
        if count >= 35.21:    # highest 25%
            return 2
        elif count <= 33.25:  # lowest 25%
            return 0
        else:
            return 1

    def _categorize_flow(self, flow_per_cycle):
        """Converts float flow into a category

            UPDATES:
            -------
            2017-09-27 histogram analysis sugest the following
            breakdowns for the quantiles of 20% and 75% for
            the varable flow_per_cyle
        """
        if flow_per_cycle > .5067: # Top 25% data-points
            return 2

        if flow_per_cycle > .2784:  # Average 20% -- 75%  data-points
            return 1
        return 0  # Bottom 20% data-points

    def _categorize_queue(self, queue_per_cycle):
        """Converts float queue into a category

            UPDATES:
            -------
            2017-09-27 histogram analysis sugest the following
            breakdowns for the quantiles of 20% and 75% for
            the varable queue_per_cyle
        """
        if queue_per_cycle > .2002:  # Top 25% data-points
            return 2
        if queue_per_cycle > .1042:  # Average 20% -- 75%  data-points
            return 1
        return 0  # Bottom 20% data-points


class InFlows(flow_params.InFlows):
    """InFlow: plus load & dump functionality"""

    @classmethod
    def make(cls, network_id, horizon, demand_type, label):

        inflows = cls(network_id, horizon, demand_type)
        # checks if route exists -- returning the path
        path = inflows_dump(
            network_id,
            inflows,
            distribution=demand_type,
            label=label
        )
        
        return path

    def __init__(self, network_id, horizon, demand_type,
                 additional_params=ADDITIONAL_PARAMS):

        super(InFlows, self).__init__()

        edges = get_edges(network_id)
        # an array of kwargs
        params = []
        for eid in get_routes(network_id):
            # use edges distribution to filter routes
            # if eid in initial_config.edges_distribution:
            edge = [e for e in edges if e['id'] == eid][0]

            num_lanes = edge['numLanes'] if 'numLanes' in edge else 1

            probability = 0.2
            args = (eid, 'human')
            if demand_type == 'lane':
                kwargs = {
                    'probability': round(probability * num_lanes, 2),
                    'depart_lane': 'best',
                    'depart_speed': 'random',
                    'name': f'lane_{eid}',
                    'begin': 1,
                    'end': horizon
                }

                params.append((args, kwargs))
            elif demand_type == 'switch':
                switch = additional_params['switch']
                num_flows = max(math.ceil(horizon / switch), 1)
                for hr in range(num_flows):
                    step = min(horizon - hr * switch, switch)
                    # switches in accordance to the number of lanes
                    if (hr + num_lanes) % 2 == 1:
                        probability = probability + 0.2 * num_lanes

                    kwargs = {
                        'probability': round(probability, 2),
                        'depart_lane': 'best',
                        'depart_speed': 'random',
                        'name': f'switch_{eid}',
                        'begin': 1 + hr * switch,
                        'end': step + hr * switch
                    }
                    probability = 0.2

                    params.append((args, kwargs))
            else:
                raise ValueError(f'Unknown demand_type {demand_type}')
        
        # Sort params flows will be consecutive
        params = sorted(params, key=lambda x: x[1]['end'])
        params = sorted(params, key=lambda x: x[1]['begin'])
        for args, kwargs in params:
            
            self.add(*args, **kwargs)
            
class NetParams(flow_params.NetParams):
    """Extends NetParams to work with saved templates"""

    @classmethod
    def from_template(cls, network_id, horizon, demand_type, label=None):
        """Factory method based on {network_id} layout + configs
        
        Params:
        -------
        *   network_id: string
            standard {network_id}.net.xml file, ex: `intersection`
            see data/networks for a list
        *   horizon: integer
            latest depart time
        *   demand_type: string
            string
        *   label: string
            e.g `eval, `train` or `test`

        Returns:
        -------
        *   ilurl.core.params.NetParams
            network parameters SEE parent
        """
        net_path = get_path(network_id, 'net')
        # TODO: test if exists first!
        rou_path = InFlows.make(network_id,
                                horizon, demand_type, label=label)
        vtype_path = get_vehicle_types()

        return cls(
            template={
                'net': net_path,
                'vtype': vtype_path,
                'rou': rou_path
            }
        )

    @classmethod
    def load(cls, network_id, route_path):
        """Loads paremeters from net {network_id} and
            routes from {route_path}
        
        Params:
        -------
        *   network_id: string
            standard {network_id}.net.xml file, ex: `intersection`
            see data/networks for a list
        *   route_path: string
            valid path on disk for a *.rou.xml file

        Returns:
        -------
        *   ilurl.core.params.NetParams
            network parameters SEE parent
        """
        net_path = get_path(network_id, 'net')
        vtype_path = get_vehicle_types()

        return cls(
            template={
                'net': net_path,
                'vtype': vtype_path,
                'rou': [route_path]
            }
        )
