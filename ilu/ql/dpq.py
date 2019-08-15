from ilu.core.params import QLParams
from ilu.ql.choice import choice_eps_greedy
from ilu.ql.define import dpq_tls
from ilu.ql.update import dpq_update


class DPQ(object):
    def __init__(self, ql_params):
        self.alpha = ql_params.alpha
        self.gamma = ql_params.gamma
        self.epsilon = ql_params.epsilon
        self.Q = dpq_tls(ql_params.states.rank, ql_params.states.depth,
                         ql_params.actions.rank, ql_params.actions.depth,
                         ql_params.initial_value)

    def rl_actions(self, s):
        actions, values = zip(*self.Q[s].items())
        return choice_eps_greedy(actions, values, self.epsilon)

    def update(self, s, a, r, s1):
        dpq_update(self.gamma, self.alpha, self.Q, s, a, r, s1)
