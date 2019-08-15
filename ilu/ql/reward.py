"""This modules implement some custom rewards"""

__author__ = "Guilherme Varela"
__date__ = "2019-07-26"
import numpy as np


class RewardCalculator(object):
    def __init__(self, ql_params):
        self.type = ql_params.rewards.type
        self.costs = ql_params.rewards.costs
        self.categorize = lambda x: ql_params.categorize_space(x)
        self.split = lambda x: ql_params.split_space(x)

    def calculate(self, observation_space):
        if self.type in ('weighted_average', ):
            speeds, counts = self.split(observation_space)
            if counts is not None:
                K = sum(counts)
                if K == 0.0:
                    return 0.0
                return sum([s * c for s, c in zip(speeds, counts)]) / K
        elif self.type in ('costs', ):
            speeds, counts = self.split(self.categorize(observation_space))
            return reward_costs(speeds, counts, self.costs)
        else:
            raise NotImplementedError


def reward_costs(speeds, counts, costs):
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
