"""The module updates q value function"""

__author__ = "Guilherme Varela"
__date__ = "2019-07-25"


def dpq_update(gamma, alpha, q,
               state, action, reward, next_state):
    """Applies Q-Learning"""

    # compute Q* = max{Q(S',a), a}
    # i.e choose the best action for the state
    # and follow that policy thereafter
    q_max = max(q[next_state].values())
    delta = gamma * q_max - q[state][action]
    q[state][action] += alpha * (reward + delta)

    return q
