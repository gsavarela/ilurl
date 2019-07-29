'''
            Traffic Light Environments

    defines utilities functionality such as logging

'''
__author__ = 'Guilherme Varela'
__data__ = '2019-07-29'
from collections import defaultdict


def logger(function):
    def decorator(*args, **kwargs):

        self, rl_actions = args
        if not hasattr(self, 'log'):
            self.log = defaultdict(list)

        ret = function(*args, **kwargs)
        self.log['t'].append(self.sim_step * self.step_counter)
        self.log['S'].append(self.get_state())
        self.log['R'].append(self.compute_reward(rl_actions))
        self.log['A'].append(self.action)
        return ret

    return decorator
