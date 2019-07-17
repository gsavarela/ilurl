'''
            Traffic Light Environments

    Extends the flow's green wave environmenets
'''
__author__ = "Guilherme Varela"

from collections import defaultdict
from itertools import product as prod

import numpy as np

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

    This is a single TFLQLAgent controlling a variable number of
    traffic lights (TFL) with discrete features defined as such:

    1. One TFLQLAgent controlling k = 1, 2, ..., K TFL

    2. The actions for the agent is for each of the
       K-TFL is to:
        2.1 0 - keep state.
        2.2 1 - switch.
        2.3 For each switch action the TFL must be open at
            least for min_switch_time.
        2.4 A switch action might no be taken while a traffic
            light is on yellow state.


    3. Each k-TFL can only observe it's subjacent edges -
        meaning the state is described by the cars locally
        available on the neighborhood of K-TFL.

    4. The state Sk for each of the K-TFL can be represented by
       a tuple such that Sk = (vk, nk) where:
        4.1 vk is the mean speed over all adjacent edges.
        4.2 nk is the total number of all adjacent edges.

    5. The state S is a set describes all the possible configurations
        such that S = (S1, S2 ..., SK) for ease of implementation
        the S representation is flattened such that:

            S = (v1, n1, v2, n2  ..vK, nK)

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

        # neighbouring maps neightborhood edges
        self._init_traffic_light_to_edges()

        # Q learning stuff
        self.num_features = 2   # every state is composed of 2 features (velocity, number)
        self.feature_depth = 3  # every feature is composed of 3 dimensions (low, medium, high)
        self.action_depth = 2   # keep on current state or skip to the next
        self._init_Q()

    def eps_greedy(self, S):
        """Applies Q-Learning using an epsilon greedy policy"""
        S = tuple(S)


        # direction are the current values for traffic lights
        actions_values = list(self.Q[S].items())
        actions_values = self._tuple_filter(actions_values)

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

    def _init_traffic_light_to_edges(self):
        # map traffic light to edges
        self.traffic_light_to_edges = defaultdict(list)
        for n in range(self.num_traffic_lights):
            i = int(n / self.grid_array["col_num"])  # row counter
            j = n - i * self.grid_array["col_num"]   # column counter

            for s in ('left', 'right'):
                self.traffic_light_to_edges[n].append("{}{}_{}".format(s, i, j))

            for jj in range(j, j + 2):
                for s in ('bottom', 'top'):
                    self.traffic_light_to_edges[n].append("{}{}_{}".format(s, i, jj))

            for s in ('left', 'right'):
                self.traffic_light_to_edges[n].append("{}{}_{}".format(s, i + 1, j))

    def _init_Q(self):
        # map traffic light to edges
        self.traffic_light_to_edges = defaultdict(list)
        for n in range(self.num_traffic_lights):
            i = int(n / self.grid_array["col_num"])  # row counter
            j = n - i * self.grid_array["col_num"]   # column counter

            for s in ('left', 'right'):
                self.traffic_light_to_edges[n].append("{}{}_{}".format(s, i, j))

            for jj in range(j, j + 2):
                for s in ('bottom', 'top'):
                    self.traffic_light_to_edges[n].append("{}{}_{}".format(s, i, jj))

            for s in ('left', 'right'):
                self.traffic_light_to_edges[n].append("{}{}_{}".format(s, i + 1, j))

        rs = self.num_features * self.num_traffic_lights
        ra = self.num_traffic_lights

        self.Q = {
            tuple(s):
                {
                    tuple(a): 0
                    for a in prod(range(self.action_depth), repeat=ra)
                }
            for s in prod(range(self.feature_depth), repeat=rs)
        }


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

        Sprime = self.get_state()
        self.q_update(S, A, R, Sprime)
        self._log(S, A, R, Sprime)


    def get_state(self):
        speed_in_edge = [
            (self.k.vehicle.get_speed(veh_id),
             self.k.vehicle.get_edge(veh_id))
            for veh_id in self.k.vehicle.get_ids()
        ]

        speed = [0] * self.num_traffic_lights
        counts = [0] * self.num_traffic_lights

        for i, edges in enumerate(self.traffic_light_to_edges.values()):
            for s, e in speed_in_edge:
                if e in edges:
                    if counts[i] == 0:
                        speed[i] = s
                        counts[i] = 1
                    else:
                        speed[i] = (speed[i] * counts[i] + s) / (counts[i] + 1)
                        counts[i] += 1

        max_speed = self.k.scenario.max_speed()
        for i, v in enumerate(speed):
            if v >= .66 * max_speed:
                speed[i] = 2
            elif v <= .33 * max_speed:
                speed[i] = 0
            else:
                speed[i] = 1

        max_count = len(speed_in_edge)
        for i, c in enumerate(counts):
            if c >= .66 * (max_count / self.num_traffic_lights):
                counts[i] = 2
            elif c <= .33 * (max_count / self.num_traffic_lights):
                counts[i] = 0
            else:
                counts[i] = 1


        return tuple(x for sc in zip(speed, counts) for x in sc)

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
                arg = tuple(int(f) for f in arg[0])
            ret.append(arg)
        return tuple(ret)

    def _tuple_toggle(self, tpl:tuple) -> tuple:
        """Negates the sign of the binary-tuple"""
        return tuple([int(not(bin(b))) for b in tpl])

    def _tuple_apply_mask(self, tpl:tuple, msk:tuple):
        """Negates the sign of the binary-tuple"""
        return tuple([t * m for t, m in zip(tpl, msk)])

    def _tuple_filter(self, list_of_tuples :list) -> list:
        """filters a list of tuples based on a mask"""
        mask = tuple(
            np.bitwise_or(
                np.bitwise_not(self.last_change.astype(bool)),
                np.bitwise_not(self.last_change < self.min_switch_time)
            ).flatten()
            .astype(int)
        )

        ret = []
        for action, value in list_of_tuples:
            filt = False
            for a, m in zip(action, mask):
                if m == 0 and a == 1:
                    filt = True
                    break
            if not filt:
                ret.append((action, value))
        return ret

    def _log(self, S, A, R, Sprime):
        if not hasattr(self, 'dump'):
            self.dump = defaultdict(list)

        self.dump['t'].append(self.step_counter)
        self.dump['S'].append(str(S))
        self.dump['A'].append(str(A))
        self.dump['R'].append(R)
        self.dump['Sprime'].append(str(Sprime))







