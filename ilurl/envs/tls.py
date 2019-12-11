'''
            Traffic Light Environments

    Extends the flow's green wave environmenets
'''
__author__ = "Guilherme Varela"
__date__ = "2019-12-10"
from collections import defaultdict

import numpy as np

from flow.core import rewards
from flow.envs.base_env import Env

from ilurl.core.ql.dpq import DPQ
from ilurl.core.ql.reward import RewardCalculator
from ilurl.utils.serialize import Serializer

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
    'switch_time': 5
}


class TrafficLightQLEnv(Env, Serializer):
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


        #THIS IS FROM ACCEL ENV
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

        super(TrafficLightQLEnv, self).__init__(env_params,
                                                    sim_params,
                                                    scenario,
                                                    simulator=simulator)
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
        self.duration = 0.0

        # keeps the internal value of sim step
        self.sim_step = sim_params.sim_step

        self.num_traffic_lights = 1

        #TODO: evaluate from TrafficLightGridEnv
        ##### TrafficLightGridEnv ######
        self.steps = env_params.horizon
        self.obs_var_labels = {
            'edges': np.zeros((self.steps, self.k.vehicle.num_vehicles)),
            'velocities': np.zeros((self.steps, self.k.vehicle.num_vehicles)),
            'positions': np.zeros((self.steps, self.k.vehicle.num_vehicles))
        }

        # Keeps track of the last time the traffic lights in an intersection
        # were allowed to change (the last time the lights were allowed to
        # change from a red-green state to a red-yellow state.)
        self.last_change = np.zeros((self.num_traffic_lights, 1))
        # Keeps track of the direction of the intersection (the direction that
        # is currently being allowed to flow. 0 indicates flow from top to
        # bottom, and 1 indicates flow from left to right.)
        self.direction = np.zeros((self.num_traffic_lights, 1))
        # Value of 1 indicates that the intersection is in a red-yellow state.
        # value 0 indicates that the intersection is in a red-green state.
        self.currently_yellow = np.zeros((self.num_traffic_lights, 1))

        # when this hits min_switch_time we change from yellow to red
        # the second column indicates the direction that is currently being
        # allowed to flow. 0 is flowing top to bottom, 1 is left to right
        # For third column, 0 signifies yellow and 1 green or red
        self.min_switch_time = env_params.additional_params["switch_time"]

        self.tl_type = env_params.additional_params.get('tl_type')
        if self.tl_type != "actuated":
            for i in range(self.num_traffic_lights):
                self.k.traffic_light.set_state(
                    node_id='GS_247123161'.format(i), state="GGgrrrrGGGrrr")
                    #node_id='center' + str(i), state="GrGr")
                self.currently_yellow[i] = 0


        self.discrete = env_params.additional_params.get("discrete", False)
        #####
        # initializes the observable scope
        self.incoming = {
            tls: defaultdict(list)
            for tls in range(self.num_traffic_lights)
        }

        self.outgoing = {
            tls: defaultdict(list)
            for tls in range(self.num_traffic_lights)
        }
        self.incoming_edge_ids = {
            tls: []
            for tls in range(self.num_traffic_lights)
        }

        self.outgoing_edge_ids = {
            tls: []
            for tls in range(self.num_traffic_lights)
        }
        # neighbouring maps neighbourhood edges
        self.init_observation_scope_filter()

        # Q learning stuff
        self.ql_params = ql_params
        self.dpq = DPQ(ql_params)
        self.reward_calculator = RewardCalculator(ql_params)
        self.rl_action = None

        self.memo_speeds = {tls: {} for tls in range(self.num_traffic_lights)}
        self.memo_counts = {tls: {} for tls in range(self.num_traffic_lights)}
        self.memo_flows = {tls: {} for tls in range(self.num_traffic_lights)}
        self.memo_queue = {tls: {} for tls in range(self.num_traffic_lights)}

        self.memo_rewards = {}
        self.memo_observation_space = {}

    def init_observation_scope_filter(self):
        """Returns edges attached to the node center#{node_id}

        This function should be provided by Kernel#Scenario,
        effectively unbinding the Grid enviroment names from
        the agent implementation -- that doesn't seem to be
        an option.

        """
        # TODO: Replace hard coded edges with scenario data
        for n in range(self.num_traffic_lights):
            # RIGHT to LEFT
            self.incoming_edge_ids[n].append('309265401#0')
            self.incoming_edge_ids[n].append('309265401#0')

            # TOP DOWN
            self.incoming_edge_ids[n].append('392619842#0')

            # LEFT to RIGHT
            self.incoming_edge_ids[n].append('-238059328#2')
            self.incoming_edge_ids[n].append('-238059328#2')

            # BOTTOM UP
            self.incoming_edge_ids[n].append('-238059324#1')

            # RIGHT to LEFT
            self.outgoing_edge_ids[n].append('238059328#0')
            self.outgoing_edge_ids[n].append('238059328#0')

            # TOP DOWN
            self.outgoing_edge_ids[n].append('-383432312#1')

            # LEFT to RIGHT
            self.outgoing_edge_ids[n].append('-309265401#2')
            self.outgoing_edge_ids[n].append('-309265401#2')

            # BOTTOM UP
            self.outgoing_edge_ids[n].append('238059324#0')

    ### TrafficLightGridEnv
    def get_state(self):
        """See class definition."""
        # compute the normalizers
        max_dist = 132
        # max_dist = max(grid_array["short_length"],
        #                grid_array["long_length"],
        #                grid_array["inner_length"])

        # get the state arrays
        speeds = [
            self.k.vehicle.get_speed(veh_id) / self.k.scenario.max_speed()
            for veh_id in self.k.vehicle.get_ids()
        ]
        dist_to_intersec = [
            self.get_distance_to_intersection(veh_id) / max_dist
            for veh_id in self.k.vehicle.get_ids()
        ]
        edges = [
            self._convert_edge(self.k.vehicle.get_edge(veh_id)) /
            (self.k.scenario.network.num_edges - 1)
            for veh_id in self.k.vehicle.get_ids()
        ]

        state = [
            speeds, dist_to_intersec, edges,
            self.last_change.flatten().tolist(),
            self.direction.flatten().tolist(),
            self.currently_yellow.flatten().tolist()
        ]
        return np.array(state)

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        # check if the action space is discrete
        if self.discrete:
            # convert single value to list of 0's and 1's
            rl_mask = [int(x) for x in list('{0:0b}'.format(rl_actions))]
            rl_mask = [0] * (self.num_traffic_lights - len(rl_mask)) + rl_mask
        else:
            # convert values less than 0 to zero and above 0 to 1. 0 indicates
            # that should not switch the direction, and 1 indicates that switch
            # should happen
            rl_mask = rl_actions > 0.0

        for i, action in enumerate(rl_mask):
            if self.currently_yellow[i] == 1:  # currently yellow
                self.last_change[i] += self.sim_step
                # Check if our timer has exceeded the yellow phase, meaning it
                # should switch to red
                if self.last_change[i] >= self.min_switch_time:
                    if self.direction[i] == 0:
                        self.k.traffic_light.set_state(
                            node_id='GS_247123161'.format(i),
                            state='GGgrrrrGGGrrr')
                            # node_id='center{}'.format(i),
                            # state="GrGr")
                    else:
                        self.k.traffic_light.set_state(
                            node_id='GS_247123161'.format(i),
                            state='rrrGGggrrrGGg')
                            # node_id='center{}'.format(i),
                            # state='rGrG')
                    self.currently_yellow[i] = 0
            else:
                if action:
                    if self.direction[i] == 0:
                        self.k.traffic_light.set_state(
                            node_id='GS_247123161'.format(i),
                            state='yyyrrrryyyrrr')
                        # node_id='center{}'.format(i),
                        # state='yryr')
                    else:
                        self.k.traffic_light.set_state(
                            node_id='GS_247123161'.format(i),
                            state='rrryyyyrrryyy')
                            # node_id='center{}'.format(i),
                            # state='ryry')
                    self.last_change[i] = 0.0
                    self.direction[i] = not self.direction[i]
                    self.currently_yellow[i] = 1

    #############
    def set_observation_space(self):
        """updates the observation space

        Assumes that each traffic light carries a speed sensor. The agent
        has the traffic light information if the vehicle is as close as 50%
        the edge's length.

        Updates the following data structures:

        * incoming: nested dict
               outer keys: int
                    traffic_light_id
               inner keys: float
                    frame_id of observations ranging from 0 to duration
               values: list
                    vehicle speeds at frame or edge
        * outgoing: nested dict
               outer keys: int
                    traffic_light_id
               inner keys: float
                    frame_id of observations ranging from 0 to duration
               values: list
                    vehicle speeds at frame or edge
        RETURNS:
        --------
        """
        def extract(edge_ids):
            veh_ids = []
            for edge_id in edge_ids:
                veh_ids += \
                    [veh_id
                     for veh_id in self.k.vehicle.get_ids_by_edge(edge_id)
                     if self.get_distance_to_intersection(veh_id) <
                        0.5 * self.k.scenario.edge_length(edge_id)]
            speeds = [
                self.k.vehicle.get_speed(veh_id)
                for veh_id in veh_ids
            ]
            return veh_ids, speeds

        for tls in range(self.num_traffic_lights):

            self.incoming[tls][self.duration] = \
                                extract(self.incoming_edge_ids[tls])

            self.outgoing[tls][self.duration] = \
                                extract(self.outgoing_edge_ids[tls])

    def get_observation_space(self):
        """consolidates the observation space

        WARNING:
            when all cars are dispatched the
            state will be encoded with speed zero --
            change for when there aren't any cars
            the state is equivalent to maximum speed
        """
        data = []

        def delay(t):
            return round(
                t - self.sim_step
                if t >= self.sim_step else
                self.cycle_time - self.sim_step
                if self.step_counter > 1 else 0.0, 2)
        prev = delay(self.duration)

        if (prev not in self.memo_observation_space) or self.step_counter <= 2:
            # Make the average between cycles
            t =  max(self.duration, self.sim_step) \
                 if self.step_counter * self.sim_step < self.cycle_time \
                 else self.cycle_time

            for tls in range(self.num_traffic_lights):

                for label in self.ql_params.states_labels:

                    if label in ('count',):
                        count = 0
                        count += len(self.incoming[tls][prev][1]) \
                                 if prev in self.incoming[tls] else 0.0
                        count += len(self.outgoing[tls][prev][1]) \
                                 if prev in self.outgoing[tls] else 0.0
                        self.memo_counts[tls][prev] = count
                        value = np.mean(list(self.memo_counts[tls].values()))

                    elif label in ('flow',):
                        # outflow measures cumulate number of the
                        # vehicles leaving the intersection
                        veh_set = set(self.outgoing[tls][prev][0]) \
                                 if prev in self.outgoing[tls] else set()


                        # The vehicles which we should accumulate over
                        prevprev = delay(prev)
                        prev_veh_set = self.memo_flows[tls][prevprev] \
                                       if prevprev in self.memo_flows[tls] else set()

                        # The vehicles which should be deprecated
                        old_veh_set = self.memo_flows[tls][prev] \
                                      if prev in self.memo_flows[tls] else set()

                        self.memo_flows[tls][prev] = \
                            (veh_set | prev_veh_set) - (prev_veh_set & old_veh_set)

                        value = len(self.memo_flows[tls][prev]) / t 
                    elif label in ('queue',):
                        # vehicles are slowing :
                        queue = []
                        queue += self.incoming[tls][prev][1] \
                                  if prev in self.incoming[tls] else []

                        queue = [q for q in queue
                                 if q < 0.10 * self.k.scenario.max_speed()]

                        self.memo_queue[tls][prev] = len(queue)
                        value = np.mean(list(self.memo_queue[tls].values())) / t

                    elif label in ('speed',):
                        speeds = []
                        speeds += self.incoming[tls][prev][1] \
                                 if prev in self.incoming[tls] else []
                        speeds += self.outgoing[tls][prev][1] \
                                 if prev in self.outgoing[tls] else []

                        self.memo_speeds[tls][prev] = \
                            0.0 if not any(speeds) else round(np.mean(speeds), 2)
                        value = np.mean(list(self.memo_speeds[tls].values()))
                    else:
                        raise ValueError('Label not found')

                    data.append(round(value, 2))

            self.memo_observation_space[prev] = tuple(data)
        return self.memo_observation_space[prev]

    def get_state(self):
        """See class definition."""
        # categorize
        # s_max = self.k.scenario.max_speed()
        return self.ql_params.categorize_space(self.get_observation_space())

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
        if self.duration == 0:
            action = self.dpq.rl_actions(tuple(state))

            self.rl_action = action
        else:
            action = None
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
                    directions = [int(d[0]) for d in self.direction]
                    diracts = zip(directions, self.rl_action)
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

    def apply_rl_actions(self, rl_actions):
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
                rl_action = self.rl_actions(self.get_state())
            else:
                rl_action = rl_actions

            # some function is changing self._state to numpy array
            # if self._state is None or isinstance(self._state, np.ndarray):
            if self.step_counter == 1:
                self.prev_state = self.get_state()
                self.prev_action = rl_action
            self.memo_rewards = {}
            self.memo_observation_space = {}

        #  _apply_rl_actions -- actions have to be on integer format
        idx = self._to_index(self.control_actions(static=False))

        # updates traffic lights' control signals
        self._apply_rl_actions(idx)

        if self.duration == 0.0 and self.step_counter > 1:
            # place q-learning here
            reward = self.compute_reward(rl_actions)

            state = self.get_state()
            self.dpq.update(self.prev_state, self.prev_action, reward, state)
            self.prev_state = state
            self.prev_action = rl_action

        self.duration = round(
            self.duration + self.sim_step,
            2,
        ) % self.cycle_time

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition.
        """
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


    # TODO: Copy & Paste dependency on TrafficLightGridEnv
    # ===============================
    # ============ UTILS ============
    # ===============================

    def get_distance_to_intersection(self, veh_ids):
        """Determine the distance from a vehicle to its next intersection.

        Parameters
        ----------
        veh_ids : str or str list
            vehicle(s) identifier(s)

        Returns
        -------
        float (or float list)
            distance to closest intersection
        """
        if isinstance(veh_ids, list):
            return [self.get_distance_to_intersection(veh_id)
                    for veh_id in veh_ids]
        return self.find_intersection_dist(veh_ids)

    def find_intersection_dist(self, veh_id):
        """Return distance from intersection.

        Return the distance from the vehicle's current position to the position
        of the node it is heading toward.
        """
        edge_id = self.k.vehicle.get_edge(veh_id)
        # FIXME this might not be the best way of handling this
        if edge_id == "":
            return -10

        # TODO: Remove reference GS_247123161
        if 'GS_247123161' in edge_id:
            return 0
        edge_len = self.k.scenario.edge_length(edge_id)
        relative_pos = self.k.vehicle.get_position(veh_id)
        dist = edge_len - relative_pos
        return dist
