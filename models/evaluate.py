'''Evaluation script for smart grid scenario'''

__author__ = 'Guilherme Varela'
__date__ = '2019-09-24'

from glob import glob
import json

import dill

from ilurl.core.experiment import Experiment
from ilurl.envs.base import TrafficLightQLEnv


def evaluate(envs):
    for i, env in enumerate(envs):
        # HACK: This should be handled by the proper loader
        # env_eval = TrafficLightQLGridEnv(env.env_params,
        #                                  env.sim_params,
        #                                  env.ql_params,
        #                                  env.scenario)

        env_eval = TrafficLightQLEnv(env.env_params,
                                     env.sim_params,
                                     env.ql_params,
                                     env.scenario)
        env_eval.dpq.Q = env.dpq.Q
        env_eval.dpq.stop = True
        exp_eval = Experiment(env_eval)
        print(f"Running evaluation {i + 1}")
        info = exp_eval.run(100, 360000)
        with open(f'{env_eval.scenario.name}.eval.info.json', 'w') as f:
            json.dump(info, f)


        


if __name__ == '__main__':
    # TODO: provide command line arguments in order to customize
    # both path and examples
    paths = glob("*.pickle")
    if  not any(paths):
        raise Exception("Environment pickles must be saved on root")

    envs = []
    for path in paths:
        with open(path, mode='rb') as f:
            envs.append(dill.load(f))
    evaluate(envs)
