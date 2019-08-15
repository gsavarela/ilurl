'''
            Traffic Light Environments

    Extends the flow's green wave environmenets
'''
__author__ = "Guilherme Varela"
from collections import defaultdict

import numpy as np

from flow.core import rewards
from flow.envs.green_wave_env import ADDITIONAL_ENV_PARAMS, TrafficLightGridEnv
from ilu.ql.dpq import DPQ
from ilu.ql.reward import RewardCalculator
from ilu.utils.decorators import logger
from ilu.utils.serialize import Serializer

# action rank is the number of dimensions for the
# representation of one tls action
TLS_ACTION_RANK = 1
# action depth is size of each of dimensions
# ( binary )
TLS_ACTION_RANK_DEPTH = 2
# action rank is the number of dimensions for the
# representation of one tls action
TLS_STATE_RANK = 2
# action depth is size of each of dimensions
# ( binary )
TLS_STATE_RANK_DEPTH = 3

QL_PARAMS = {
    # epsilon is the chance to adopt a random action instead of
    # a greedy action - if is None then adopt optimistic values
    # see class definitino for details
    'epsilon': None,
    # alpha is the learning rate the weight given to new knowledge
    'alpha': 5e-2,
    # gamma is the discount rate for value function
    'gamma': 0.999,
}
ADDITIONAL_TLS_PARAMS = {
    # short_cycle_time is the time it takes for a direction to enter
    # the yellow state e.g
    # GrGr -> yryr or rGrG -> yGyG
    'short_cycle_time': 36,
    # slow_time same effect as short_cycle_time but taking longer
    'long_cycle_time': 36,
    # switch_time is the time it takes to exit the yellow state e.g
    # yryr -> rGrG or ryry -> GrGr
    'switch_time': 5,
    # use only incoming edges to account for observation states
    # None means use both incoming and outgoing
    'filter_incoming_edges': None,
}


class TrafficLightQLGridEnv(TrafficLightGridEnv, Serializer):
    """Environment used to train traffic lights.

    This is a single TFLQLAgent controlling a variable number of
    traffic lights (TFL) with discrete features defined as such:

    1. One TFLQLAgent controlling k = 1, 2, ..., K TFL

    2. The actions for the agent is for each of the
       K-TFL is to:
        2.1 0 - short green (direction 0), Ex:
            (10s Green, 5s Yellow, 25s Red)
        2.2 1 - long green (direction 0), Ex:
            (25s Green, 5s Yellow, 10s Red)
        2.3 short green (direction 0) implies long
            green (direction 1) and vice-versa.

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

    PARAMETERS
    ----------

    TrafficLightGridEnv
    -------------------

    * switch_time: minimum time a light must be constant before it
                    switches (in seconds). Earlier RL commands are
                    ignored.
    * tl_type: whether the traffic lights should be actuated by sumo or
                RL, options are respectively "actuated" and "controlled"
    * discrete: determines whether the action space is meant to be
                discrete or continuous.

    Q-Learning
    ----------
    * epsilon: small positive number representing the change of
                the agent taking a random action [1].
    * alpha: positive number between 0 and 1 representing the
                 update rate [1].
    * gamma: positive number between 0 and 1 representing the
                discount rate for the rewards [1].

   References:
    [1] Sutton et Barto, Reinforcement Learning 2nd Ed 2018

    States
        An observation is the vehicle data, taken at each intersecting
        edge for the k-th traffic light. Futhermore, the traffic light
        accesses only the vehicles on the range of half an edge length.
        Currently only the number of vehicles and their speed are being
        evaluated and ordered into 3 categories:

                        0 ("low"), 1 ("medium"), 2 ("high").

    Actions
        The action space consist of a list of float variables ranging
        from 0-1 specifying whether a traffic light is supposed to
        switch or not. The actions are sent to the traffic light in the
        grid from left to right and then top to bottom.

    Rewards
        The reward is the negative per vehicle delay minus a penalty for
        switching traffic lights

    Termination
        A rollout is terminated once the time horizon is reached.

    Additional
        Vehicles are rerouted to the start of their original routes
        once they reach the end of the network in order to ensure a
        constant number of vehicles.
    """

    def __init__(self,
                 env_params,
                 sim_params,
                 ql_params,
                 scenario,
                 simulator='traci'):

        super(TrafficLightQLGridEnv, self).__init__(env_params,
                                                    sim_params,
                                                    scenario,
                                                    simulator=simulator)

        #THIS IS FROM ACCELL ENV
        # variables used to sort vehicles by their initial position plus
        # distance traveled
        self.prev_pos = dict()
        self.absolute_position = dict()

        # those parameters are going to be forwarded to learning engine
        for p in QL_PARAMS:
            if not hasattr(ql_params, p):
                raise KeyError(
                    'Q-learning parameter "{}" not supplied'.format(p))
            else:
                # dynamicaly set attributes for Q-learning attributes
                val = getattr(ql_params, p)
                setattr(self, p, val)

        # those parameters are extensions from the standard
        # traffic light green wave additional parameters
        for p in ADDITIONAL_TLS_PARAMS:
            if p not in env_params.additional_params:
                raise KeyError(
                    'Traffic Light parameter "{}" not supplied'.format(p))
            else:
                # dynamicaly set attributes for Q-learning attributes
                val = env_params.additional_params[p]
                setattr(self, p, val)

        # Check constrains on minimum duration
        if self.switch_time > self.short_cycle_time:
            raise ValueError('''Fast phase time must be
                greater than minimum switch time''')

        if self.short_cycle_time > self.long_cycle_time:
            raise ValueError('''Fast phase time must be
                lesser than or equal to slow phase time''')
        self.cycle_time = self.long_cycle_time + self.short_cycle_time

        # duration measures the amount of time the current
        # configuration has been going on
        self.duration = 0

        # keeps the internal value of sim step
        self.sim_step = sim_params.sim_step

        # Q learning stuff
        # neighbouring maps neighbourhood edges
        self._init_observation_space_filter()
        self.ql_params = ql_params
        self.dpq = DPQ(ql_params)
        self.reward_calculator = RewardCalculator(ql_params)
        self.rl_action = None
        self._observation_space = {
            tls: {}
            for tls in range(self.num_traffic_lights)
        }
        self.memo_speeds = {tls: {} for tls in range(self.num_traffic_lights)}
        self.memo_counts = {tls: {} for tls in range(self.num_traffic_lights)}

    def _init_observation_space_filter(self):
        """Returns edges attached to the node center#{node_id}

        This function should be provided by Kernel#Scenario,
        effectively unbinding the Grid enviroment names from
        the agent implementation -- that doesn't seem to be
        an option.

        """
        # map traffic light to edges
        self._observation_space_filter = defaultdict(list)
        incoming = self.filter_incoming_edges is None or\
            self.filter_incoming_edges is True
        outgoing = self.filter_incoming_edges is None or\
            self.filter_incoming_edges is False

        for n in range(self.num_traffic_lights):
            edge_ids = []

            i = int(n / self.grid_array["col_num"])  # row counter
            j = n - i * self.grid_array["col_num"]  # column counter

            # handles left and right of the n-th traffic light
            if incoming:
                edge_ids.append('right{}_{}'.format(i, j))
                edge_ids.append('top{}_{}'.format(i, j + 1))
                edge_ids.append('bot{}_{}'.format(i, j))
                edge_ids.append('left{}_{}'.format(i + 1, j))

            if outgoing:
                edge_ids.append('right{}_{}'.format(i + 1, j))
                edge_ids.append('top{}_{}'.format(i, j))
                edge_ids.append('bot{}_{}'.format(i, j + 1))
                edge_ids.append('left{}_{}'.format(i, j))

            self._observation_space_filter[n] = edge_ids

    def set_observation_space(self):
        """updates the observation space

        Assumes that each traffic light carries a speed sensor. The agent
        has the traffic light information if the vehicle is as close as 50%
        the edge's length.

        * _observation_space: nested dict
               outer keys: int
                    traffic_light_id
               inner keys: float
                    frame_id of observations ranging from 0 to duration
               values: list
                    vehicle speeds at frame or edge
        RETURNS:
        --------
        """
        # backwards compatibility
        # prevent same vehicle to be accounted twice
        for tls in range(self.num_traffic_lights):
            for edge_id in self._observation_space_filter[tls]:
                k = self.duration
                self._observation_space[tls][k] = \
                    [self.k.vehicle.get_speed(veh_id)
                     for veh_id in self.k.vehicle.get_ids_by_edge(edge_id)
                     if self.get_distance_to_intersection(veh_id) <
                        0.5 * self.k.scenario.edge_length(edge_id)]

    def get_observation_space(self):
        """consolidates the observation space

        WARNING:
            when all cars are dispatched the
            state will be encoded with speed zero --
            change for when there aren't any cars
            the state is equivalent to maximum speed
        """
        # get ids from vehicles -- closer to the edges
        ret = []
        # prev = round(max(self.duration - self.sim_step, 0), 2)
        d = round(max(self.duration - self.sim_step, 0), 2)

        for tls in range(self.num_traffic_lights):
            # testing incremental script
            veh_speeds = {
                t: 0.0 if not any(values) else round(np.mean(values), 2)
                for t, values in self._observation_space[tls].items() if t == d
            }

            self.memo_speeds[tls].update(veh_speeds)
            if any(self.memo_speeds[tls].values()):
                ret.append(np.mean(list(self.memo_speeds[tls].values())))
            else:
                ret.append(0.0)

            veh_counts = {
                t: len(_values)
                for t, _values in self._observation_space[tls].items()
                if t == d
            }
            self.memo_counts[tls].update(veh_counts)
            if any(self.memo_counts[tls].values()):
                ret.append(np.mean(list(self.memo_counts[tls].values())))
            else:
                ret.append(0.0)

        return tuple(ret)

    def get_state(self):
        """See class definition."""
        # categorize
        # s_max = self.k.scenario.max_speed()
        return self.ql_params.categorize_states(self.get_observation_space())

    def rl_actions(self, state):
        """
        rl_action:
            0 fast green — on vertical ~ slow green on horizontal
            1 slow green — on vertical ~ fast green on horizontal

                                    direction = 0   direction = 1
                                    ------------    -------------

            rl_action = 0           |    |               |    |
            -------------           | FG |               | Sr |
                                    |    |               |    |
                            --------x    x------    -----x    x-----
                                Fr          Fr        SG        SG
                            --------x    x------    -----x    x-----
                                    |    |               |    |
                                    | FG |               | Sr |
                                    |    |               |    |

            rl_action = 1           |    |               |    |
            -------------           | SG |               | Fr |
                                    |    |               |    |
                            --------x    x------    -----x    x-----
                                Sr          Sr        FG        FG
                            --------x    x------    -----x    x-----
                                    |    |               |    |
                                    | SG |               | Fr |
                                    |    |               |    |

        """
        action = self.dpq.rl_actions(tuple(state))

        if self.rl_action is None or self.duration == 0.0:
            self.rl_action = action
        return action

    def control_actions(self, static=False):
        """Either switch traffic light for the frame or
            keep orientation


        """
        ret = []
        if static:
            if self.duration == 0 and self.step_counter > 1 or \
                    self.duration == self.short_cycle_time or \
                    self.duration == self.cycle_time - self.switch_time:
                ret = tuple([1] * self.num_traffic_lights)
            else:
                ret = tuple([0] * self.num_traffic_lights)

        else:
            if self.duration == self.short_cycle_time - self.switch_time:
                # Short phase
                if self.short_cycle_time == self.long_cycle_time:
                    # handles the case of both phases are the same
                    ret = tuple([1] * self.num_traffic_lights)
                else:
                    diracts = zip(self.direction, self.rl_action)
                    ret = tuple([int(d == a) for d, a in diracts])
                    self.control_action = ret

            elif self.duration == self.long_cycle_time - self.switch_time:
                # Long phase
                ret = tuple(
                    [1 if ca == 0 else 0 for ca in self.control_action])
            elif self.duration == self.cycle_time - self.switch_time:
                ret = tuple([1] * self.num_traffic_lights)  # switch to init
            else:
                ret = tuple([0] * self.num_traffic_lights)  # do nothing
        return ret

    # @logger
    def _apply_rl_actions(self, rl_actions):
        """Q-Learning

        Algorithm as in Sutton et Barto, 2018 [1]
        for a single agent controlling all traffic
        light.

        Parameters
        ----------

        rl_actions: list of actions or None
        """
        self.set_observation_space()

        if self.duration == 0.0:
            if rl_actions is None:
                action = self.rl_actions(self.get_state())
            else:
                action = rl_actions

            # some function is changing self._state to numpy array
            # if self._state is None or isinstance(self._state, np.ndarray):
            if self.step_counter == 1:
                self.prev_state = self.get_state()

            self.memo_rewards = {}
        action = self.control_actions(static=False)

        #  _apply_rl_actions -- actions have to be on integer format
        idx = self._to_index(action)

        super(TrafficLightQLGridEnv, self)._apply_rl_actions(idx)

        if self.duration == 0 and self.step_counter > 1:
            # place q-learning here
            reward = self.compute_reward(rl_actions)

            state = self.get_state()
            self.dpq.update(self.prev_state, action, reward, state)
            self.prev_state = state

        self.duration = round(
            self.duration + self.sim_step,
            2,
        ) % self.cycle_time

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition.
        """
        # return rewards.average_velocity(self, fail=False)
        # return np.mean(self.k.vehicle.get_speed(self.k.vehicle.get_ids()))
        if self.duration not in self.memo_rewards:
            # rew = rewards.average_velocity(self, fail=False)
            rew = self.reward_calculator.calculate(
                self.get_observation_space())
            self.memo_rewards[self.duration] = rew
        return self.memo_rewards[self.duration]

    def _to_index(self, action):
        """"Converts an action in tuple form to an integer"""
        # defines a generator on the reverse of the action
        # the super class defines actions oposite as ours
        gen_act = enumerate(action[::-1])

        # defines PowerOf2
        def po2(k, pwr):
            return int(k * pow(2, pwr))

        return sum([po2(k, n) for n, k in gen_act])

    def additional_command(self):
        """See parent class.

        Define which vehicles are observed for visualization purposes, and
        update the sorting of vehicles using the self.sorted_ids variable.
        """
        # specify observed vehicles
        if self.k.vehicle.num_rl_vehicles > 0:
            for veh_id in self.k.vehicle.get_human_ids():
                self.k.vehicle.set_observed(veh_id)

        # update the "absolute_position" variable
        for veh_id in self.k.vehicle.get_ids():
            this_pos = self.k.vehicle.get_x_by_id(veh_id)

            if this_pos == -1001:
                # in case the vehicle isn't in the network
                self.absolute_position[veh_id] = -1001
            else:
                change = this_pos - self.prev_pos.get(veh_id, this_pos)
                self.absolute_position[veh_id] = \
                    (self.absolute_position.get(veh_id, this_pos) + change) \
                    % self.k.scenario.length()
                self.prev_pos[veh_id] = this_pos

    @property
    def sorted_ids(self):
        """Sort the vehicle ids of vehicles in the network by position.

        This environment does this by sorting vehicles by their absolute
        position, defined as their initial position plus distance traveled.

        Returns
        -------
        list of str
            a list of all vehicle IDs sorted by position
        """
        if self.env_params.additional_params['sort_vehicles']:
            return sorted(self.k.vehicle.get_ids(), key=self._get_abs_position)
        else:
            return self.k.vehicle.get_ids()

    def _get_abs_position(self, veh_id):
        """Return the absolute position of a vehicle."""
        return self.absolute_position.get(veh_id, -1001)

    def reset(self):
        """See parent class.

        This also includes updating the initial absolute position and previous
        position.
        """
        obs = super().reset()

        for veh_id in self.k.vehicle.get_ids():
            self.absolute_position[veh_id] = self.k.vehicle.get_x_by_id(veh_id)
            self.prev_pos[veh_id] = self.k.vehicle.get_x_by_id(veh_id)

        return obs
