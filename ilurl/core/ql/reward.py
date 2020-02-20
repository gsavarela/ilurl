"""This modules implement some custom rewards"""

__author__ = "Guilherme Varela"
__date__ = "2019-07-26"
import pdb
import numpy as np

REWARD_TYPES = ('fix', 'weighted_average', 'score', 'target_velocity')


class RewardCalculator(object):
    def __init__(self, env_params, ql_params):
        self.type = ql_params.rewards.type
        if self.type == 'target_velocity': 
            if 'target_velocity' not in env_params.additional_params:
                raise ValueError('''
                    target_velocity must be provided on env_params
                    ''')

            else:
                # target mean velocity
                self.target_velocity = \
                env_params.additional_params['target_velocity']
        self.costs = ql_params.rewards.costs
        self.categorize = lambda x: ql_params.categorize_space(x)
        self.split = lambda x: ql_params.split_space(x)
        self.labels = ql_params.states_labels

    def calculate(self, observation_space):
        if self.type in ('fix', ):
            # makes as compution based on discrete levels
            speeds, counts = self.split(self.categorize(observation_space))
            return reward_fix(speeds, counts, self.costs)

        elif self.type in ('weighted_average', ):
            # weighted average
            speeds, counts = self.split(observation_space)
            if counts is not None:
                K = sum(counts)
                if K == 0.0:
                    return 0.0
                return sum([s * c for s, c in zip(speeds, counts)]) / K

        elif self.type in ('target_velocity',):
            # get this target velocity from environment
            speeds, counts = self.split(observation_space)
            if sum(counts) <= 0.0:
                return 0

            max_cost = np.array([self.target_velocity] * len(speeds))
            return -np.maximum(max_cost - speeds, 0).dot(counts)

        elif self.type in ('score', ):
            # scores some are either negative in case of queues
            # or positive in the case of flow or velocity
            data = self.split(observation_space)
            s = 0
            for i, label in enumerate(self.labels):
                if label in ('flow', 'speed',):
                    s += sum(data[i])
                if label in ('queue', ):
                    s -= sum(data[i])
            return s
        else:
            raise NotImplementedError


def reward_fix(speeds, counts, costs):
    """Constant reward of 1000 if
        (a) there are no vehicles ( dispatched all )
        (b) all cars are moving at their fastest

    PARAMETERS
    ----------
    * speeds: tuple
        categorical vehicle speeds

    * counts: tuple
        categorical vehicle number

    * costs: tuple
        cost for being at speed level i.g
        cost[0] cost for speeds=0
        cost[k] cost for speed=k ...
    """

    N = len(speeds)
    R = len(counts)
    weights = [0] * R
    # weights are proportional to the speed levels
    for i, s in enumerate(speeds):
        if counts[i] == 0:  # no vehicles present
            # either all have been dispached or none
            weights[-1] += 1
        else:
            weights[s] += 1

    # weights are then normalized to sum at most one
    weights = [w / N for w in weights]

    k = 1
    for c, w in zip(costs, weights):
        k -= c * w

    return 1000 * k
