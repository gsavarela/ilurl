'''
            Traffic Light Environments

    Extends the flow's green wave environmenets
'''
__author__ = "Guilherme Varela"

import numpy as np
from itertools import product as prod
from numpy.random import rand, choice

from gym.spaces.discrete import Discrete
from gym.spaces.box import Box
from gym.spaces.tuple_space import Tuple

from flow.core import rewards
from flow.envs.green_wave_env import TrafficLightGridEnv, ADDITIONAL_ENV_PARAMS


ADDITIONAL_QL_PARAMS = {
        'epsilon': 5e-2,
        'alpha': 5e-2,
        'gamma': 0.999
}
ADDITIONAL_QL_ENV_PARAMS = {
    **ADDITIONAL_ENV_PARAMS,
    **ADDITIONAL_QL_PARAMS
}


class TrafficLightQLGridEnv(TrafficLightGridEnv):
    """Environment used to train traffic lights.

    Required from env_params:

    * switch_time: minimum time a light must be constant before
      it switches (in seconds). Earlier RL commands are ignored.
    * tl_type: whether the traffic lights should be actuated by sumo or RL,
      options are respectively "actuated" and "controlled"
    * discrete: determines whether the action space is meant to be discrete or
      continuous

    Q-Learning  parameters:

    * epsilon: [1]  small positive number representing the change of the agent
               taking a random action.
    * alpha: [1]  positive number between 0 and 1 representing the update rate.
    * gamma: [1]  positive number between 0 and 1 representing the discount
             rate for the rewards.

   References:
    [1] Sutton et Barto, Reinforcement Learning 2nd Ed 2018

    States
        An observation is the distance of each vehicle to its intersection, a
        number uniquely identifying which edge the vehicle is on, and the speed
        of the vehicle.

    Actions
        The action space consist of a list of float variables ranging from 0-1
        specifying whether a traffic light is supposed to switch or not. The
        actions are sent to the traffic light in the grid from left to right
        and then top to bottom.

    Rewards
        The reward is the negative per vehicle delay minus a penalty for
        switching traffic lights

    Termination
        A rollout is terminated once the time horizon is reached.

    Additional
        Vehicles are rerouted to the start of their original routes once they
        reach the end of the network in order to ensure a constant number of
        vehicles.
    """
    def __init__(self,  env_params, sim_params, scenario, simulator='traci'):

        for p, val in ADDITIONAL_QL_PARAMS.items():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))
            else:
                # dynamicaly set attributes for Q-learning attributes alpha,
                # gamma, epsilon
                setattr(self, p, val)

        super(TrafficLightQLGridEnv, self).__init__(env_params,
                                                    sim_params,
                                                    scenario,
                                                    simulator=simulator)

        # Q learning stuff
        r = self.num_traffic_lights
        self.Q = {
            tuple(state):
                {
                    tuple(action): 0 for action in prod([0, 1], repeat=r)
                }
            for state in prod([0, 1], repeat=r)
        }

    @property
    def action_space(self):
        """See class definition."""
        return Discrete(2 ** self.num_traffic_lights)

    @property
    def observation_space(self):
        """See class definition."""
        speed = Box(
            low=0,
            high=1,
            shape=(self.initial_vehicles.num_vehicles,),
            dtype=np.float32)

        traffic_lights = Box(
            low=0.,
            high=1,
            shape=(2 * self.rows * self.cols,),
            dtype=np.float32)

        return Tuple((speed, traffic_lights))

    def eps_greedy(self, S):
        """Applies Q-Learning using an epsilon greedy policy"""
        S = self._tuplefy(S)[0]

        # direction are the current values for traffic lights
        actions_values = list(self.Q[S].items())

        if rand() <= self.epsilon:
            # Take a random action
            idx = choice(len(actions_values))
            action_value = actions_values[idx]

        else:
            # greedy action
            action_value = max(actions_values, key=lambda x: x[1])

        # Take action A observe R and S'
        A = action_value[0]
        return A

    def q_update(self, S, A, R, Sprime):
        """Applies Q-Learning using an epsilon greedy policy"""

        S, A, Sprime = self._tuplefy(S, A, Sprime)

        # compute Q* = max{Q(S',a), a}
        Qstar = max(self.Q[Sprime].items(), key=lambda x: x[1])[1]
        self.Q[S][A] += self.alpha * (R + self.gamma * Qstar - self.Q[S][A])

    def get_state(self):
        """See class definition."""
        return self.direction

    def _apply_rl_actions(self, rl_actions):
        """Q-Learning

        Algorithm as in Sutton et Barto, 2018 [1]
        for a single agent controlling all traffic
        light.

        """

        # check if the action space is discrete
        S, A = self.get_state(), rl_actions

        #  _apply_rl_actions -- actions have to be on integer format
        idx = self._integerfy(rl_actions)

        super(TrafficLightQLGridEnv, self)._apply_rl_actions(idx)

        # place q-learning here
        R = self.compute_reward(rl_actions)

        self.q_update(S, A, R, self.get_state())

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        return rewards.average_velocity(self, fail=False)

    def _integerfy(self, action: tuple):
        """"Converts an action in tuple form to an integer"""
        # defines a generator on the reverse of the action
        # the super class defines actions oposite as ours
        gen_act = enumerate(action[::-1])

        # defines PowerOf2
        def po2(k, n):
            return int(k * pow(2, n))

        return sum([po2(k, n) for n, k in gen_act])

    def _tuplefy(self, *args):
        """"Converts a numpy.ndarray to tuple"""
        ret = []
        for arg in args:
            if not isinstance(arg, tuple):
                arg = tuple(int(f) for f in arg)
            ret.append(arg)
        return tuple(ret)

