"""This modules implement some custom rewards"""

__author__ = "Guilherme Varela"
__date__ = "2019-07-26"


def reward_fixed_apply(env):
    """Äpplies the reward fixed to environment
    """
    p = env.cost_medium
    q = env.cost_low
    state = env.get_state()
    n, n1, n0 = env.num_traffic_lights, 0, 0
    for speed in state[0::2]:
        n1 = n1 + 1 if speed == 1 else n1
        n0 = n0 + 1 if speed == 0 else n0

    return reward_fixed(1000, p, q, n1 / n, n0 / n)


def reward_fixed(K, p, q, ratio_medium, ratio_low):
    """Promotes the maxium reward of K for each step

    PARAMETERS
    ----------
    * K: int or float
        Maximum reward happens when either:
        ratio_medium == 0 and ratio_low == 0, or
        there are the number of vehicle on simulation
        is zero

    * p: float (0 < p < q)
        Reward loss for excess ratio_medium

    * q: float (p < q <= 1)
        Reward loss for excess ratio_low

    * ratio_medium: float ( 0 <= ratio_medium < 1)
        Ratio of the number of medium speed vehicles
        and vehicle total

    * ratio_low: float ( 0 <= ratio_low < 1)
        Ratio of the number of low speed vehicles
        and vehicle total
    """
    return K * (1 - p * ratio_medium - q * ratio_low)
