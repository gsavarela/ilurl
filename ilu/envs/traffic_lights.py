'''
            Traffic Light Environments

    Extends the flow's green wave environmenets
'''
__author__ = "Guilherme Varela"

from collections import defaultdict
from itertools import product as prod
from itertools import groupby


import numpy as np

from numpy.random import rand, choice

from gym.spaces.discrete import Discrete
from gym.spaces.box import Box
from gym.spaces.tuple_space import Tuple

from flow.core import rewards
from flow.envs.green_wave_env import TrafficLightGridEnv, ADDITIONAL_ENV_PARAMS


ADDITIONAL_QL_PARAMS = {
        # epsilon is the chance to adopt a random action instead of
        # a greedy action ( SEE Sutton & Barto 2018 2ND edition )
        'epsilon': 5e-2,
        # alpha is the learning rate the weight given to new knowledge
        'alpha': 5e-2,
        # gamma is the discount rate for value function
        'gamma': 0.999,
        # min_duration_time is the time a given traffic light has to stay
        # at the same configuration: min_duration_time >= min_switch_time
        'min_duration_time': 10
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


        super(TrafficLightQLGridEnv, self).__init__(env_params,
                                                    sim_params,
                                                    scenario,
                                                    simulator=simulator)

        for p, val in ADDITIONAL_QL_PARAMS.items():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))
            else:
                # dynamicaly set attributes for Q-learning attributes 
                setattr(self, p, val)

        # Check constrains on minimum duration
        if self.min_switch_time > self.min_duration_time:
            raise ValueError(
                'Minimun duration time must be greater than minimum switch time')

        # duration measures the amount of time the current
        # configuration has been going on
        self.duration = [0] * self.num_traffic_lights

        # keeps the internal value of sim step
        self.sim_step = sim_params.sim_step

        # Q learning stuff
        self.num_features = 2   # every state is composed of 2 features (velocity, number)
        self.feature_depth = 3  # every feature is composed of 3 dimensions (low, medium, high)
        self.action_depth = 2   # keep on current state or skip to the next
        self._init_traffic_light_to_edges()
        self._init_Q(max_speed=self.k.scenario.max_speed())
        self.default_action = tuple([0] * self.num_traffic_lights)

        # neighbouring maps neightborhood edges
        self._init_traffic_light_to_edges()

    def rl_actions(self, state):
        S = tuple(state)

        # direction are the current values for traffic lights
        actions_values = list(self.Q[S].items())
        actions_values = self._action_value_filter(actions_values)

        if self.use_epsilon:
            return self._eps_greedy_choice(actions_values)
        else:
            return self._optimistic_choice(actions_values)

    def _eps_greedy_choice(self, actions_values):
        """Takes a single action using an epsilon greedy policy.

            See Chapter 2 of [1]

        References
        ----------
        [1] Sutton et Barto, Reinforcement Learning 2nd Ed 2018

        Parameters
        ----------
        actions_values : list of nested tuples
            each element of the list is a tuple containing
            action : tuple[self.num_traffic_lights]
            value : q estimate for the state and action

        Returns
        -------
        float
            discounted value for state and action pair
        """
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

    def _optimistic_choice(self, actions_values):
        """Takes a single action using an optimistic values policy.

            See section 2.6 of [1]

        References
        ----------
        [1] Sutton et Barto, Reinforcement Learning 2nd Ed 2018

        Parameters
        ----------
        actions_values : list of nested tuples
            each element of the list is a tuple containing
            action : tuple[self.num_traffic_lights]
            value : q estimate for the state and action

        Returns
        -------
        float
            discounted value for state and action pair
        """

        # direction are the current values for traffic lights
        action_value = max(actions_values, key=lambda x: x[1])

        # Take action A observe R and S'
        A = action_value[0]
        return A


    def q_update(self, S, A, R, Sprime):
        """Applies Q-Learning using an epsilon greedy policy"""

        # compute Q* = max{Q(S',a), a}
        Qstar = max(self.Q[Sprime].values())
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
                for s in ('bot', 'top'):
                    self.traffic_light_to_edges[n].append("{}{}_{}".format(s, i, jj))

            for s in ('left', 'right'):
                self.traffic_light_to_edges[n].append("{}{}_{}".format(s, i + 1, j))

    def _init_Q(self, max_speed=None):
        if max_speed is None:
            # use epsilon greed criteria
            self.use_epsilon = True
            Q0 = 0
        else:
            # use optimistic values
            self.use_epsilon = False
            Q0 = max_speed

        rs = self.num_features * self.num_traffic_lights
        ra = self.num_traffic_lights

        self.Q = {
            tuple(s):
                {
                    tuple(a): Q0
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
        if rl_actions is None:
            rl_actions = self.rl_actions(self.get_state())

        # check if the action space is discrete
        S, A = self.get_state(), rl_actions

        #  _apply_rl_actions -- actions have to be on integer format
        idx = self._action_to_index(rl_actions)

        super(TrafficLightQLGridEnv, self)._apply_rl_actions(idx)

        # place q-learning here
        R = self.compute_reward(rl_actions)

        for i, a in enumerate(A):
            if a == 1:
                self.duration[i] = 0.0
            else:
                self.duration[i] += self.sim_step

        Sprime = self.get_state()
        self.q_update(S, A, R, Sprime)
        self._log(S, A, R, Sprime)


    def get_state(self):
        """See class definition."""

        # query api and get the speeds and edges for each vehicle
        speeds_edges_list = [
            (self.k.vehicle.get_speed(veh_id),
             self.k.vehicle.get_edge(veh_id))
            for veh_id in self.k.vehicle.get_ids()
        ]

        # group by edges
        edges_dict = dict()
        for edge, edge_group in groupby(speeds_edges_list, key=lambda x: x[1]):
            edge_speeds_list = [s for s, _ in edge_group]
            edges_dict[edge] = (
                sum(edge_speeds_list),
                len(edge_speeds_list)
            )

        # aggregate
        data_dict = dict()
        for i, edges_list in self.traffic_light_to_edges.items():
            speed_tuple, count_tuple = zip(*[
                speed_tuple
                for edge_id, speed_tuple in edges_dict.items() if edge_id in edges_list
            ])
            data_dict[i] = (sum(speed_tuple) / sum(count_tuple), sum(count_tuple))

        # categorize
        ret = []
        max_speed = self.k.scenario.max_speed()
        max_count = len(speeds_edges_list)
        for s, c in data_dict.values():
            if s >= .66 * max_speed:
                s = 2
            elif s <= .25 * max_speed:
                s = 0
            else:
                s = 1
            ret.append(s)

            if c >= .66 * max_count:
                c = 2
            elif c <= .15 * max_count:
                c = 0
            else:
                c = 1
            ret.append(c)

        return tuple(ret)

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        return rewards.average_velocity(self, fail=False)

    def _action_to_index(self, action: tuple):
        """"Converts an action in tuple form to an integer"""
        # defines a generator on the reverse of the action
        # the super class defines actions oposite as ours
        gen_act = enumerate(action[::-1])

        # defines PowerOf2
        def po2(k, pwr):
            return int(k * pow(2, pwr))

        return sum([po2(k, n) for n, k in gen_act])

    def _action_value_filter(self, actions_values: list) -> list:
        """filters a list of tuples based on a mask"""

        # avaliates duration: 1 lets the pattern to pass otherwise
        filter_mask = [
            1 if d <= self.min_duration_time else 0 for d in self.duration
        ]

        ffn = lambda x: self._apply_mask_actions(x, filter_mask)

        ret = [
            (action, value)
            for action, value in actions_values if ffn(action)
        ]

        return ret

    def _apply_mask_actions(self, action, filter_mask):
        return not any([
            a * m for a, m in zip(action, filter_mask)
        ])

    def _log(self, S, A, R, Sprime):
        if not hasattr(self, 'dump'):
            self.dump = defaultdict(list)

        self.dump['t'].append(self.step_counter * self.sim_step)
        self.dump['S'].append(str(S))
        self.dump['A'].append(str(A))
        self.dump['R'].append(R)
        self.dump['Sprime'].append(str(Sprime))
        
