"""Objects that define the various meta-parameters of an experiment."""
import pdb
from collections import namedtuple

REWARD_TYPES = ('average_speed', 'cost')
Bounds = namedtuple('Bounds', 'rank depth')


class QLParams:
    """Base Q-learning parameters"""

    def __init__(
            self,
            epsilon=3e-2,
            alpha=5e-2,
            gamma=0.95,
            initial_value=0,
            reward_type='average_speed',
            cost_medium=0.5,
            cost_low=0.75,
            states={
                'rank': 8,
                'depth': 3
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

        * reward_type: type of reward function to use, see
                        ilu/core/rewards for details.
        * cost_medium: cost for the number of vehicles on the medium
                       speed tier. (reward_type='cost')
        * cost_low: cost for the number of vehicles on the the low
                    speed tier. (reward_type='cost')

        REFERENCES:
        ----------
            [1] Sutton et Barto, Reinforcement Learning 2nd Ed 2018

        """
        if reward_type not in REWARD_TYPES:
            raise ValueError('''reward_type must be in {} got {}'''.format(
                REWARD_TYPES, reward_type))
        elif reward_type == 'cost':
            if cost_medium > cost_low or cost_low > 1 or cost_medium <= 0:
                raise ValueError(
                    '''The ineq 0 < cost_medium < cost_low < 1 must hold.
                       got cost_medium = {} and cost_low = {}'''.format(
                        cost_medium, cost_low))
        # self.reward_type = reward_type
        # self.cost_medium = cost_medium
        if alpha <= 0 or alpha >= 1:
            raise ValueError('''The ineq 0 < alpha < 1 must hold.
                    got alpha = {}'''.format(alpha))

        if gamma <= 0 or gamma >= 1:
            raise ValueError('''The ineq 0 < gamma < 1 must hold.
                    got gamma = {}'''.format(gamma))

        if epsilon <= 0 or epsilon >= 1:
            raise ValueError('''The ineq 0 < epsilon < 1 must hold.
                    got epsilon = {}'''.format(epsilon))
        for attr, value in locals().items():
            if attr not in ('self', 'states', 'actions'):
                setattr(self, attr, value)

        if 'states' in locals():
            states_dict = locals()['states']
            self.set_states(states_dict['rank'], states_dict['depth'])

        if 'actions' in locals():
            actions_dict = locals()['actions']
            self.set_actions(actions_dict['rank'], actions_dict['depth'])

    def set_states(self, rank, depth):
        self.states = Bounds(rank, depth)

    def set_actions(self, rank, depth):
        self.actions = Bounds(rank, depth)