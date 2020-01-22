"""Objects that define the various meta-parameters of an experiment."""
from collections import namedtuple
from ilurl.core.ql.reward import REWARD_TYPES
from ilurl.core.ql.choice import CHOICE_TYPES

STATE_FEATURES = ('speed', 'count', 'flow', 'queue')
ACTIONS = ('fast_slow_green', )
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
            num_traffic_lights=4,
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
        rank = self.num_traffic_lights * len(states_tuple)
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

        labels = list(self.states_labels) * self.num_traffic_lights
        return tuple([
            getattr(self, f'_categorize_{name}')(value)
            for name, value in zip(labels, observation_space)
        ])

    def split_space(self, observation_space):
        """Splits different variables into tuple"""
        num_labels = len(self.states_labels)
        ss = [observation_space[ll::num_labels]
              for ll in range(num_labels)]
        return tuple(ss)

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
