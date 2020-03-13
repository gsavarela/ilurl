"""Implementation of dynamic programming TD methods with function approximation"""
import numpy as np

from ilurl.core.params import QLParams
from ilurl.core.ql.choice import choice_eps_greedy, choice_ucb
from ilurl.core.ql.define import dpq_tls
from ilurl.core.ql.update import dpq_update


class DPQ(object):
    def __init__(self, ql_params):

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

    def rl_actions(self, s):
        if self.stop:
            # Argmax greedy choice.
            actions, values = zip(*self.Q[s].items())
            choosen, _ = choice_eps_greedy(actions, values, 0)
            self.explored.append(False)
        else:
            
            if self.choice_type in ('eps-greedy',):
                actions, values = zip(*self.Q[s].items())
                choosen, exp = choice_eps_greedy(actions, values, self.epsilon)
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

        if not self.stop:

            # Update (state, action) counter.
            self.state_action_counter[s][a] += 1

            # Calculate learning rate.
            lr = 1 / np.power(1 + self.state_action_counter[s][a], 2/3)

            # Q-learning update.
            dpq_update(self.gamma, lr, self.Q, s, a, r, s1)

    @property
    def stop(self):
        """Stops exploring"""
        return self._stop

    @stop.setter
    def stop(self, stop):
        self._stop = stop
