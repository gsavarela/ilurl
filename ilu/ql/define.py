"""The module helps define the q learning dictionary"""

__author__ = "Guilherme Varela"
__date__ = "2019-07-25"
from itertools import product as prod


def dpq_tls(state_rank, state_dim,
            action_rank, action_dim, initial_value=0):
    """Prepares a dynamic programming Q-learning table
    for a traffic light agent

    PARAMETERS
    ----------
    **_rank: int
    See catspace bellow

    **_dim: int
    See catspace bellow

    *initial_value: int
    See dpq bellow

    RETURNS
    -------
    * q: dictionary
    Where q is a nested dictionary where the outer keys
    are states and the inner keys are the actions

    """
    state_space = catspace(state_rank, state_dim)
    action_space = catspace(action_rank, action_dim)
    return dpq(state_space, action_space,
               initial_value=initial_value)


def dpq(states, actions, initial_value=0):
    """dynamic programming definition for Q-learning

    This implementation returns a table like, nested
    dict of states and actions -- where the inner values
    q[s][a] reflect the best current estimates of expected
    rewards for taking action a on state s.

    See [1] chapter 3 for details.

    PARAMETERS
    ----------
    * states: list, tuple or any other enumerable
    Where each state in the collection is representated by
    any other immutable type i.g int or tuple.

    * actions: list, tuple or any other enumerable
    Where each action in the collection is representated by
    any other immutable type i.g int or tuple.

    * initial_value: numeric
    Initial a priori guess for the Q function

    RETURNS
    -------
    * q: dictionary
    Where q is a nested dictionary where the outer keys
    are states and the inner keys are the actions

    REFERENCE
    ---------

    [1] Sutton et Barto, Reinforcement Learning 2nd Ed 2018
    """
    return {
            s: {
                a: initial_value
                for a in actions
            }
            for s in states
    }


def catspace(rank, dim):
    """Makes a categorical space of the discrete
    combinations of rank each holding dim elements

    USAGE
    -----
    > catspace(3, 2)
    > [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
       (0, 0, 1), (1, 0, 1), (0, 1, 0), (1, 1, 0)]

    PARAMETERS
    ----------
    * rank: int
    The length of the tuples ( width )

    * dim: int
    The number of elements in each position of
    the tuple ( depth )

    RETURNS
    -------
    * space: list of tuples
    Each tuple is of length rank, and each
    tuple element is an integer in 0...dim
    """
    return [
        tuple(e)
        for e in prod(range(dim), repeat=rank)]
