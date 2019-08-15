"""Objects that define the various meta-parameters of an experiment."""
from collections import namedtuple

REWARD_TYPES = ('weighted_average', 'costs')
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
            initial_value=0,
            max_speed=35,
            rewards={
                'type': 'weighted_average',
                'costs': None
            },
            states={
                'rank': 8,
                'depth': 3,
            },
            actions={
                'rank': 4,
                'depth': 2
            },
    ):
        """Instantiate base traffic light.

        PARAMETERS
        ----------
        * epsilon: is the chance to adopt a random action instead of
                  a greedy action [1].

        * alpha: is the learning rate the weight given to new
                 knowledge [1].

        * gamma: is the discount rate for value function [1].

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
        for attr, value in kwargs.items():
            if attr not in ('self', 'states', 'actions', 'rewards'):
                setattr(self, attr, value)

        if 'states' in kwargs:
            states_dict = kwargs['states']
            self.set_states(states_dict['rank'], states_dict['depth'])

        if 'actions' in kwargs:
            actions_dict = kwargs['actions']
            self.set_actions(actions_dict['rank'], actions_dict['depth'])

        if 'type' not in rewards:
            raise ValueError('''``type` must be provided in reward types''')

        elif rewards['type'] not in REWARD_TYPES:
            raise ValueError('''rewards must be in {} got {}'''.format(
                REWARD_TYPES, rewards['type']))
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

    def set_states(self, rank, depth):
        self.states = Bounds(rank, depth)

    def set_actions(self, rank, depth):
        self.actions = Bounds(rank, depth)

    def set_rewards(self, type, costs):
        self.rewards = Rewards(type, costs)

    def categorize_space(self, observation_space):
        return tuple([
            self._categorize_count(val) if i %
            2 == 1 else self._categorize_speed(val)
            for i, val in enumerate(observation_space)
        ])

    def split_space(self, observation_space):
        return observation_space[::2], \
                observation_space[1::2]

    def _categorize_speed(self, speed):
        """Converts a float speed into a category"""
        if speed >= .66 * self.max_speed:
            return 2
        elif speed <= .25 * self.max_speed:
            return 0
        else:
            return 1

    def _categorize_count(self, count):
        """Converts a int count into a category"""
        if count >= 6:
            return 2
        elif count == 0:
            return 0
        else:
            return 1
