"""Implementation of dynamic programming TD methods with function approximation"""
from copy import deepcopy
import numpy as np

from ilurl.utils.meta import MetaAgentQ
from ilurl.core.params import QLParams, Bounds
from ilurl.core.ql.choice import choice_eps_greedy, choice_ucb
from ilurl.core.ql.define import dpq_tls
from ilurl.core.ql.update import dpq_update


class DPQ(object, metaclass=MetaAgentQ):

    def __init__(self, ql_params):
        """
        Q-Learning agent

        ----------
        * epsilon: small positive number representing the change of
                    the agent taking a random action [1].
        * alpha: positive number between 0 and 1 representing the
                    update rate [1].
        * gamma: positive number between 0 and 1 representing the
                    discount rate for the rewards [1].

        References:
            [1] Sutton et Barto, Reinforcement Learning 2nd Ed 2018
            
        """

        # Store Q-learning parameters.
        self.ql_params = ql_params

        self.stop = False

        # Learning rate.
        self.alpha = ql_params.alpha

        # Action choice.
        self.choice_type = ql_params.choice_type

        # Discount factor.
        self.gamma = ql_params.gamma

        # Q-table.
        self.Q = dpq_tls(ql_params.states.rank, ql_params.states.depth,
                         ql_params.actions.rank, ql_params.actions.depth,
                         ql_params.initial_value)

        # State-action counter (for learning rate decay).
        self.state_action_counter = dpq_tls(ql_params.states.rank,
                                            ql_params.states.depth,
                                            ql_params.actions.rank,
                                            ql_params.actions.depth,
                                            0)

        # Boolean list that stores whether actions
        # were randomly picked (exploration) or not.
        self.explored = []

        # Boolean list that stores the newly visited states.
        self.visited_states = []

        # Float list that stores the distance between
        # Q-tables between updates.
        self.Q_distances = []

        # Epsilon-greedy (exploration rate).
        if self.choice_type in ('eps-greedy',):
            self.epsilon = ql_params.epsilon

        # UCB (extra-stuff).
        if self.choice_type in ('ucb',):
            self.c = ql_params.c
            self.decision_counter = 0
            self.actions_counter = {
                state: {
                    action: 1.0
                    for action in actions
                }
                for state, actions in self.Q.items()
            }

    def act(self, s):
        if self.stop:
            # Argmax greedy choice.
            actions, values = zip(*self.Q[s].items())
            choosen, exp = choice_eps_greedy(actions, values, 0)
            self.explored.append(exp)
        else:
            
            if self.choice_type in ('eps-greedy',):
                actions, values = zip(*self.Q[s].items())

                num_state_visits = sum(self.state_action_counter[s].values())
                eps = 1 / np.power(1 + num_state_visits, 2/3)

                choosen, exp = choice_eps_greedy(actions, values, eps)
                self.explored.append(exp)

            elif self.choice_type in ('optimistic',):
                raise NotImplementedError

            elif self.choice_type in ('ucb',):
                self.decision_counter += 1 if not self.stop else 0
                choosen = choice_ucb(self.Q[s].items(),
                                     self.c,
                                     self.decision_counter,
                                     self.actions_counter[s])
                self.actions_counter[s][choosen] += 1 if not self.stop else 0
            else:
                raise NotImplementedError

        return choosen

    def update(self, s, a, r, s1):

        # Track the visited states.
        if sum(self.state_action_counter[s].values()) == 0:
            self.visited_states.append(s)
        else:
            self.visited_states.append(None)

        if not self.stop:

            # Update (state, action) counter.
            self.state_action_counter[s][a] += 1

            # Calculate learning rate.
            lr = 1 / np.power(1 + self.state_action_counter[s][a], 2/3)

            Q_old = self.Q[s][a]

            # Q-learning update.
            try:
                r = sum(r)
            except TypeError:
                pass
            dpq_update(self.gamma, lr, self.Q, s, a, r, s1)

            # Calculate Q-tables distance.
            dist = np.abs(Q_old - self.Q[s][a])
            self.Q_distances.append(dist)

    @property
    def Q(self):
        return self._Q

    @Q.setter
    def Q(self, Q):
        self._Q = Q

    @property
    def stop(self):
        """Stops exploring"""
        return self._stop

    @stop.setter
    def stop(self, stop):
        self._stop = stop


class MAIQ(object, metaclass=MetaAgentQ):
    """MAIQ is the individual combination of Q agents"""
    
    def __init__(self, ql_params):

        QL_agents = []
        state_slices = []
        action_slices = []
        num_variables = len(ql_params.states_labels)
        states_depth = ql_params.states.depth
        actions_depth = ql_params.actions.depth
        
        state_rank = 0
        action_rank = 0
        for num_phases in ql_params.phases_per_traffic_light:

            ql_params_ = deepcopy(ql_params)
            ql_params_.phases_per_traffic_light = [num_phases]
            ql_params_.states = Bounds(num_phases * num_variables, states_depth)
            ql_params_.actions = Bounds(1, actions_depth)
            QL_agents.append(DPQ(ql_params_))

            state_slice = slice(state_rank, state_rank + num_phases * num_variables)
            state_slices.append(state_slice)

            action_slice = slice(action_rank, action_rank + ql_params_.actions.rank)
            action_slices.append(action_slice)

            state_rank += num_phases * num_variables
            action_rank += ql_params_.actions.rank

        self._QL_agents = QL_agents
        self._action = action_slices
        self._state = state_slices
        self.ql_params = ql_params

    def act(self, state):
        def si(x):
            return self._individual_state(state, x)

        choices = [
            _QL_agent.act(si(i)) for i, _QL_agent in enumerate(self._QL_agents)
        ]
        choosen = tuple([ichoice for choice in choices for ichoice in choice])
        return choosen

    def _individual_state(self, state, i):
        return state[self._state[i]]

    def _individual_action(self, action, i):
        return action[self._action[i]]

    # decorator <mapping> might aliviate?
    # generator <iteration> might aliviate?
    # def _individual_act(self, state):
    #     def si(x):
    #         return self._individual_state(state, x)

    #     return tuple([
    #         self._QL_agents[i].act(si(i))
    #         for i in range(self._QL_agents[i])])

    def update(self, s, a, r, s1):

            def si(x):
                return self._individual_state(s, x)

            def si1(x):
                return self._individual_state(s1, x)

            def ai(x):
                return self._individual_action(a, x)

            for i, QL_agent in enumerate(self._QL_agents):
                QL_agent.update(si(i), ai(i), r[i], si1(i))

    @property
    def Q(self):
        self._Q = {i: _QL_agent.Q
                   for i, _QL_agent in enumerate(self._QL_agents)}

        return self._Q

    @Q.setter
    def Q(self, Q):
        for i, Qi in Q.items():
              self._QL_agents[i].Q = Qi

    @property
    def explored(self):
        self._explored = {i: _QL_agent.explored
                   for i, _QL_agent in enumerate(self._QL_agents)}
        return self._explored

    @property
    def visited_states(self):
        self._visited_states = {i: _QL_agent.visited_states
                   for i, _QL_agent in enumerate(self._QL_agents)}
        return self._visited_states

    @property
    def Q_distances(self):
        self._Q_distances = {i: _QL_agent.Q_distances
                   for i, _QL_agent in enumerate(self._QL_agents)}
        return self._Q_distances

    @property
    def stop(self):
        """all or nothing stops"""
        stops = [_QL_agent.stop for _QL_agent in self._QL_agents]
        return all(stops)

    @stop.setter
    def stop(self, stop):
        for _QL_agent in self._QL_agents:
            _QL_agent.stop = stop
        return stop
