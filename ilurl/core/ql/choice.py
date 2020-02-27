"""The module helps define the q learning dictionary"""
__author__ = "Guilherme Varela"
__date__ = "2019-07-25"


from numpy import argmax, sqrt, log, isnan
from numpy.random import choice, rand

CHOICE_TYPES = ('eps-greedy', 'optimistic', 'ucb')

def all_eq(values):
    # returns True if every element of values is the same
    return all(isnan(values)) or max(values) - min(values) < 1e-6

def choice_eps_greedy(actions, values, epsilon):
    """Takes a single action using an epsilon greedy policy.

        See Chapter 2 of [1]

    REFERENCES
    ----------
    [1] Sutton et Barto, Reinforcement Learning 2nd Ed 2018

    PARAMETERS
    ----------
    * actions : list of actions
        Each element is an action the agent can take

    * values : list of float
        Each value is a q estimate for the state and action

    * epsilon : a chance of taking a random action

    RETURNS
    -------
    action
        An element from the actions element array
    """
    # greedy action -- handles the case where
    # all values are zero: or are equal.
    if rand() <= epsilon or all_eq(values):
        # Take a random action
        idx = choice(len(values))
    else:
        idx = argmax(values)

    # Take action A observe R and S'
    action = actions[idx]
    return action


def choice_optimistic(actions_values):
    """Takes a single action using an optimistic values policy.

        See section 2.6 of [1]

    references:
    ----------
    [1] Sutton et Barto, Reinforcement Learning 2nd Ed 2018

    parameters:
    ----------
    actions_values : list of nested tuples
        each element of the list is a tuple containing
        action : tuple[self.num_traffic_lights]
        value : q estimate for the state and action

    returns:
    -------
    float
        discounted value for state and action pair
    """

    if all_eq([v for _, v in actions_values]):
        idx = choice(len(actions_values))
        action = [a for a, _ in actions_values][idx]
    else:
        # direction are the current values for traffic lights
        action_value = max(actions_values, key=lambda x: x[1])

        # Take action A observe R and S'
        action = action_value[0]
    return action


def choice_ucb(actions_values, c, decision_counter, actions_counter):
    """Performs upper confidence bound for actions

        See section 2.7 of [1]

    references:
    ----------
    [1] Sutton et Barto, Reinforcement Learning 2nd Ed 2018

    parameters:
    ----------
    * actions_values : list of nested tuples
        each element of the list is a tuple containing
        action : tuple[self.num_traffic_lights]
        value : q estimate for the state and action

    * c: float (c > 0.0)
        Constant that regulates exploration (largest more
        exploring.

    * decision_counter: int
        The total number of decision actions taken e.g for
        synchronous agents: step_counter % cycle_time + 1

    * actions_counter: dict
        keys are actions values are integer
        The number of times a given action a has been selected
        len(actions_values) == len(actions_counter)

    returns:
    -------
    * action
        Returns an action
    """

    c1 = log(decision_counter)

    ucb_actions_values = \
    [(a, v + c * sqrt(c1 / actions_counter[a]))
     for a, v in actions_values]


    if all_eq([v for _, v in ucb_actions_values]):
        idx = choice(len(ucb_actions_values))
        action = [a for a, _ in actions_values][idx]

    else:
        # direction are the current values for traffic lights
        action_value = max(ucb_actions_values, key=lambda x: x[1])

        # Take action A observe R and S'
        action = action_value[0]
    return action
