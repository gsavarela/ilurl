"""The module helps define the q learning dictionary"""
__author__ = "Guilherme Varela"
__date__ = "2019-07-25"

from numpy.random import choice, rand


def choice_eps_greedy(actions_values, epsilon):
    """Takes a single action using an epsilon greedy policy.

        See Chapter 2 of [1]

    REFERENCES
    ----------
    [1] Sutton et Barto, Reinforcement Learning 2nd Ed 2018

    PARAMETERS
    ----------
    actions_values : list of nested tuples
        each element of the list is a tuple containing
        action : tuple[self.num_traffic_lights]
        value : q estimate for the state and action

    RETURNS
    -------
    float
        discounted value for state and action pair
    """
    if rand() <= epsilon:
        # Take a random action
        idx = choice(len(actions_values))
        action_value = actions_values[idx]

    else:
        # greedy action
        action_value = max(actions_values, key=lambda x: x[1])

    # Take action A observe R and S'
    A = action_value[0]
    return A


def choice_optimistic(actions_values):
    """Takes a single action using an optimistic values policy.

        See section 2.6 of [1]

    REFERENCES
    ----------
    [1] Sutton et Barto, Reinforcement Learning 2nd Ed 2018

    PARAMETERS
    ----------
    actions_values : list of nested tuples
        each element of the list is a tuple containing
        action : tuple[self.num_traffic_lights]
        value : q estimate for the state and action

    RETURNS
    -------
    float
        discounted value for state and action pair
    """

    # direction are the current values for traffic lights
    action_value = max(actions_values, key=lambda x: x[1])

    # Take action A observe R and S'
    A = action_value[0]
    return A

