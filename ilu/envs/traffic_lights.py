'''
            Traffic Light Environments

    Extends the flow's green wave environmenets
'''
__author__ = "Guilherme Varela"
from collections import defaultdict

from flow.core import rewards
from flow.envs.green_wave_env import ADDITIONAL_ENV_PARAMS, TrafficLightGridEnv
from ilu.ql.choice import choice_eps_greedy, choice_optimistic
from ilu.ql.define import dpq_tls
from ilu.ql.update import dpq_update
from ilu.ql.reward import reward_fixed_apply
from ilu.utils.serialize import Serializer
from ilu.utils.decorators import logger
ADDITIONAL_QL_PARAMS = {
    # epsilon is the chance to adopt a random action instead of
    # a greedy action - if is None then adopt optimistic values
    # see class definitino for details
    'epsilon': None,
    # alpha is the learning rate the weight given to new knowledge
    'alpha': 5e-2,
    # gamma is the discount rate for value function
    'gamma': 0.999,
    # phase_time is the time a given traffic light has to stay
    # at the same configuration: phase_time >= min_switch_time
    'phase_time': 40,
    # use only incoming edges to account for observation states
    # None means use both incoming and outgoing
    'filter_incoming_edges': None,
    'p': 0.5,
    'q': 1.0,
}
ADDITIONAL_QL_ENV_PARAMS = {**ADDITIONAL_ENV_PARAMS, **ADDITIONAL_QL_PARAMS}


class TrafficLightQLGridEnv(TrafficLightGridEnv, Serializer):
    """Environment used to train traffic lights.

    This is a single TFLQLAgent controlling a variable number of
    traffic lights (TFL) with discrete features defined as such:

    1. One TFLQLAgent controlling k = 1, 2, ..., K TFL

    2. The actions for the agent is for each of the
       K-TFL is to:
        2.1 0 - short green (10s Green, 5s Yellow, 25s Red)
        2.2 1 - long green (25s Green, 5s Yellow, 10s Red)
        2.4 A switch action might no be taken while a traffic
            light is on yellow state.


    3. Each k-TFL can only observe it's subjacent edges -
        meaning the state is described by the cars locally
        available on the neighborhood of K-TFL.

        3.1 The neighborhood of the traffic light is the half
        of the edge which is closest to the traffic light on the
        edge that's adjacent to it.

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
    ENV
    ---

    * switch_time: minimum time a light must be constant before
      it switches (in seconds). Earlier RL commands are ignored.
    * tl_type: whether the traffic lights should be actuated by sumo or RL,
      options are respectively "actuated" and "controlled"
    * discrete: determines whether the action space is meant to be discrete or
      continuous

    Q-Learning
    ----------
    * epsilon: [1]  small positive number representing the change of the agent
               taking a random action.
    * alpha: [1]  positive number between 0 and 1 representing the update rate.
    * gamma: [1]  positive number between 0 and 1 representing the discount
             rate for the rewards.

   References:
    [1] Sutton et Barto, Reinforcement Learning 2nd Ed 2018

    States
        An observation is the vehicle data, taken at each intersecting edge for
        the k-th traffic light. Futhermore, the traffic light accesses only the
        vehicles on the range of half an edge length. Currently only the number
        of vehicles and their speed are being evaluated and ordered into 3
        categories: 0 ("low"), 1 ("medium"), 2 ("high").

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

    def __init__(self, env_params, sim_params, scenario, simulator='traci'):

        super(TrafficLightQLGridEnv, self).__init__(env_params,
                                                    sim_params,
                                                    scenario,
                                                    simulator=simulator)

        self.switch_time = 5.0
        for p in ADDITIONAL_QL_PARAMS:
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))
            else:
                # dynamicaly set attributes for Q-learning attributes
                val = env_params.additional_params[p]
                setattr(self, p, val)

        # Check constrains on minimum duration
        if self.min_switch_time > self.phase_time:
            raise ValueError(
                '''Minimum phase time must be
                greater than minimum switch time''')

        # duration measures the amount of time the current
        # phase has been going on
        self.duration = 0

        # keeps the internal value of sim step
        self.sim_step = sim_params.sim_step

        # Q learning stuff
        # every state is composed of 2 features:
        # velocity, number
        self.num_features = 2
        # every feature has tree possible values
        # low, medium, high
        self.feature_depth = 3
        # keep on current state or skip to the next
        self.action_depth = 2
        # neighbouring maps neightborhood edges
        self._init_traffic_light_to_edges()
        self.Q = dpq_tls(self.num_features * self.num_traffic_lights,
                         self.feature_depth,
                         self.num_traffic_lights,
                         self.action_depth,
                         initial_value=self.k.scenario.max_speed())

    def rl_actions(self, state, compute=True):
        if compute:
            S = tuple(state)

            # direction are the current values for traffic lights
            actions_values = list(self.Q[S].items())
            # actions_values = self._action_value_filter(actions_values)

            if self.epsilon is None:
                self.action = choice_optimistic(actions_values)
            else:
                self.action = choice_eps_greedy(actions_values,
                                                self.epsilon)
            return self.action
    def _init_traffic_light_to_edges(self):
        """Returns edges attached to the node center#{node_id}

        This function should be provided by Kernel#Scenario,
        effectively unbinding the Grid enviroment names from
        the agent implementation -- that doesn't seem to be
        an option.

        """
        # map traffic light to edges
        self.traffic_light_to_edges = defaultdict(list)
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

            self.traffic_light_to_edges[n] = edge_ids

    @logger
    def _apply_rl_actions(self, rl_actions):
        """Q-Learning

        Algorithm as in Sutton et Barto, 2018 [1]
        for a single agent controlling +1 traffic
        light.

        Parameters
        ----------

        rl_actions: list of actions or None
        """
        if self.duration == 0:
            # the case where the agent might interact with
            # the system
            if rl_actions is None:
                rl_actions = self.rl_actions(self.get_state())

            # check if the action space is discrete
            state, action = self.get_state(), rl_actions

            #  _apply_rl_actions -- actions have to be on integer format
            idx = self._action_to_index(rl_actions)

            super(TrafficLightQLGridEnv, self)._apply_rl_actions(idx)

            # place q-learning here
            reward = self.compute_reward(rl_actions)

            next_state = self.get_state()
            dpq_update(self.gamma, self.alpha, self.Q,
                       state, action, reward, next_state)

            self.previous_action = action
            self.previous_direction = self.direction
        else:
            # the case where the agent is waiting to act but collects
            # a reward

            # The actions from the agent are different from the system
            # system's actions are keep and switch
            # agent's actions are short green and long green
            # this code converts one for the other
            action_list = []
            for t in range(self.num_traffic_lights):
                a = self.previous_action[t]
                if a == 0:
                    # Fast green
                    green_time = 10
                else:
                    # Slow green
                    green_time = 30

                if self.duration == green_time:
                    action_list.append(1)
                else:
                    action_list.append(0)

            idx = self._action_to_index(tuple(action_list))

            super(TrafficLightQLGridEnv, self)._apply_rl_actions(idx)
        self.duration = (self.duration + self.sim_step) % self.phase_time

    def get_state(self):
        """See class definition."""
        # build a list of tuples (vk, nk)
        # where vk is the speed for vehicles on observable edges
        #       from k-th traffic light
        #       nk is the count of vehicles on observable edges
        #       from k-th traffic light
        # for k =0..num_traffic_lights-1
        speed_count_list = []
        for n in range(self.num_traffic_lights):
            vehicle_list = self._get_observable_state_by(n)
            speed_list = [
                self.k.vehicle.get_speed(veh_id)
                for veh_id in vehicle_list
            ]
            speed_count_list.append(
                (sum(speed_list) / len(speed_list), len(speed_list)))
        # categorize
        s_max = self.k.scenario.max_speed()
        c_list = [c for _, c in speed_count_list]
        c_max = sum(c_list) / len(c_list)

        ret = []
        for s, c in speed_count_list:
            ret.append(self._categorize_speed(s, s_max))
            ret.append(self._categorize_count(c, c_max))
        return tuple(ret)

    def _categorize_speed(self, s, s_max):
        """Converts a float speed into a category"""
        if s >= .66 * s_max:
            return 2
        elif s <= .25 * s_max:
            return 0
        else:
            return 1
 
    def _categorize_count(self, c, c_max):
        """Converts a int count into a category"""
        if c >= .66 * c_max:
            return 2
        elif c <= .15 * c_max:
            return 0
        else:
            return 1

    def _get_observable_state_by(self, node_id):
        """Returns a vehicle list observed from the node_id

        This function should be provided by Kernel#Scenario,
        effectively unbinding the Grid enviroment names from
        the agent implementation -- that doesn't seem to be
        an option.

        Parameters
        ----------
        node_id: int
            a central node_id which a traffic light has control

        Returns
        -------
        state: tuple
            the observed state from traffic light on node_id
        """
        edges_list = self.traffic_light_to_edges[node_id]
        vehicle_per_edge_dict = {
            edge_id: self.k.vehicle.get_ids_by_edge(edge_id)
            for edge_id in edges_list
            if any(self.k.vehicle.get_ids_by_edge(edge_id))
        }

        vehicle_list = []
        for edge_id, vehicle_edge_list in vehicle_per_edge_dict.items():
            for veh_id in vehicle_edge_list:
                pos = self.k.vehicle.get_position(veh_id)
                length = self.k.scenario.edge_length(edge_id)
                if pos > length / 2:
                    vehicle_list.append(veh_id)

        return vehicle_list

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        # return rewards.average_velocity(self, fail=False)
        return reward_fixed_apply(self)

    def _action_to_index(self, action):
        """"Converts an action in tuple form to an integer"""
        # defines a generator on the reverse of the action
        # the super class defines actions oposite as ours
        if action is None:
            return None

        gen_act = enumerate(action[::-1])

        # defines PowerOf2
        def po2(k, pwr):
            return int(k * pow(2, pwr))

        return sum([po2(k, n) for n, k in gen_act])

    # VARELA: DEPRECATED?
    # WITH SYNC ACTIONS EITHER DEFAULT OR EVERYTHING
#     def _action_value_filter(self, actions_values):
#         """filters a list of tuples based on a mask"""
# 
#         # avaliates duration: 1 lets the pattern to pass otherwise
#         filter_mask = [
#             1 if d <= self.phase_time else 0 for d in self.duration
#         ]
# 
#         def apply_mask(x):
#             return not any([a * m for a, m in zip(action, filter_mask)])
# 
#         ret = [
#             (action, value)
#             for action, value in actions_values
#             if apply_mask(action)
#         ]
# 
#         return ret

