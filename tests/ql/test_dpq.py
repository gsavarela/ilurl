import os
import pickle
import unittest

import numpy as np
from ilu.core.params import QLParams, Bounds
from ilu.ql.dpq import DPQ


class TestDPQUpdate(unittest.TestCase):
    '''Tests update Q
    Builds a simple process MDP

    actions:
        left ((0,)) and  right ((1,))

    states:
        terminal ((0, 0), (1, 1))
        start ((1, 0))

    rewards:
        -1, 0, +1
                        (1,)        (1,)
                        ---> +0      --->+1
    (0, 0)      (0, 1)      (1, 0)      (1, 1)
       -1   <---    +0    <---
            (0,)        (0,)
    '''
    def setUp(self):
        ql_params = QLParams(
            alpha=0.5,
            gamma=1.0,
            states=('count',),
            actions=('fast_green', 'slow_green')
        )
        # This shouldn't be used like that but
        # outside testing scenarios
        ql_params.states = Bounds(rank=2, depth=2)
        ql_params.actions = Bounds(rank=1, depth=2)
        self.dpq = DPQ(ql_params)
        #  self.rands: list[5000]
        #       With 5000 3-digit random numbers (0,1)
        with open('tests/data/rands.pickle', 'rb') as f:
            self.rands = pickle.load(f)

    def test_update(self):
        # episodes
        ri = 0
        for i in range(500):
            state = (0, 1)
            # rotate random numbers
            ri = 0 if ri == len(self.rands) else ri

            for r in self.rands[ri:]:
                actions, values = zip(*self.dpq.Q[state].items())
                idx = np.argmax(values)
                if r < 0.1:
                    # choose randomly == flip bit
                    idx = 0 if idx == 1 else 0

                action = actions[idx]

                # act using list
                if state == (0, 1):
                    if action == (0, ):
                        reward = -1
                        next_state = (0, 0)
                    else:
                        reward = 0
                        next_state = (1, 0)
                elif state == (1, 0):
                    if action == (0, ):
                        reward = -1
                        next_state = (0, 1)
                    else:
                        reward = 1
                        next_state = (1, 1)

                self.dpq.update(state, action, reward, next_state)
                state = next_state
                ri += 1
                # terminal states
                if state in ((0, 0), (1, 1)):
                    break
        self.assertLess(self.dpq.Q[(0, 1)][(0, )], -0.999)
        self.assertGreater(self.dpq.Q[(0, 1)][(1, )], 0.9999999999)
        self.assertLess(self.dpq.Q[(1, 0)][(0, )], 1e-3)
        self.assertGreater(self.dpq.Q[(1, 0)][(0, )], -1e-3)
        self.assertGreater(self.dpq.Q[(1, 0)][(1, )], 0.999)
