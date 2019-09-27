from ilu.core.params import QLParams
from ilu.ql.choice import choice_eps_greedy, choice_ucb
from ilu.ql.define import dpq_tls
from ilu.ql.update import dpq_update


class DPQ(object):
    def __init__(self, ql_params):
        self.alpha = ql_params.alpha
        self.choice_type = ql_params.choice_type
        self.gamma = ql_params.gamma
        self.epsilon = ql_params.epsilon
        self.Q = dpq_tls(ql_params.states.rank, ql_params.states.depth,
                         ql_params.actions.rank, ql_params.actions.depth,
                         ql_params.initial_value)

        # ucb extra-stuff
        self.c = ql_params.c
        if self.choice_type in ('ucb',):
            self.decision_counter = 0
            self.actions_counter = {
                state: {
                    action: 1.0
                    for action in actions
                }
                for state, actions in self.Q.items()
            }

    def rl_actions(self, s):
        if self.choice_type in ('eps-greedy',):
            actions, values = zip(*self.Q[s].items())
            choosen = choice_eps_greedy(actions, values, self.epsilon)
        elif self.choice_type in ('optimistic',):
            raise NotImplementedError
        elif self.choice_type in ('ucb',):
            self.decision_counter += 1
            choosen = choice_ucb(self.Q[s].items(),
                                 self.c,
                                 self.decision_counter,
                                 self.actions_counter[s])
            self.actions_counter[s][choosen] += 1
        else:
            raise NotImplementedError

        return choosen

    def update(self, s, a, r, s1):
        dpq_update(self.gamma, self.alpha, self.Q, s, a, r, s1)