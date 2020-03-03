"""Contains an experiment class for running simulations.
   2019-06-2019
   ------------
   This file was copied from flow.core.experiment in order to
   add the following features:
   * periodically save the running data: server seems to
   be restarting every 400 steps, the rewards are being changed
   radically after each restart

   * extend outputs to costumized reward functions
   * fix bug of averaging speeds when no cars are on the simulation
   """
import warnings
import datetime
import json
import logging
import os
import tempfile
import time
from collections import defaultdict


from tqdm import tqdm

import numpy as np
from flow.core.util import emission_to_csv

# TODO: Track those anoying warning
warnings.filterwarnings('ignore')

# TODO: Generalize for any parameter
ILURL_HOME = os.environ['ILURL_HOME']

EMISSION_PATH = \
    f'{ILURL_HOME}/data/emissions/'

class Experiment:
    """
    Class for systematically running simulations in any supported simulator.

    This class acts as a runner for a scenario and environment. In order to use
    it to run an scenario and environment in the absence of a method specifying
    the actions of RL agents in the network, type the following:

        >>> from flow.envs import Env
        >>> env = Env(...)
        >>> exp = Experiment(env)  # for some env and scenario
        >>> exp.run(num_runs=1, num_steps=1000)

    If you wish to specify the actions of RL agents in the network, this may be
    done as follows:

        >>> rl_actions = lambda state: 0  # replace with something appropriate
        >>> exp.run(num_runs=1, num_steps=1000, rl_actions=rl_actions)

    Finally, if you would like to like to plot and visualize your results, this
    class can generate csv files from emission files produced by sumo. These
    files will contain the speeds, positions, edges, etc... of every vehicle
    in the network at every time step.

    In order to ensure that the simulator constructs an emission file, set the
    ``emission_path`` attribute in ``SimParams`` to some path.

        >>> from flow.core.params import SimParams
        >>> sim_params = SimParams(emission_path="./data")

    Once you have included this in your environment, run your Experiment object
    as follows:

        >>> exp.run(num_runs=1, num_steps=1000, convert_to_csv=True)

    After the experiment is complete, look at the "./data" directory. There
    will be two files, one with the suffix .xml and another with the suffix
    .csv. The latter should be easily interpretable from any csv reader (e.g.
    Excel), and can be parsed using tools such as numpy and pandas.

    Attributes
    ----------
    env : flow.envs.Env
        the environment object the simulator will run
    """

    def __init__(self,
                env,
                dir_path=EMISSION_PATH,
                train=True,
                log_info=False,
                log_info_interval=20,
                save_agent=False):
        """

        Parameters
        ----------
        env : flow.envs.Env
            the environment object the simulator will run
        dir_path : int
            path to dump experiment results
        train : bool
            whether to train agent
        log_info : bool
            whether to log experiment info into json file throughout training
        log_info_interval : int
            json file log interval (in number of agent-update steps)
        save_agent : bool
            whether to save RL agent parameters throughout training

        """
        sim_step = env.sim_params.sim_step
        # garantees that the enviroment has stoped
        if not train:
            env.stop = True

        self.env = env
        self.train = train
        self.dir_path = dir_path
        # fails gracifully if an environment with no cycle time
        # is provided
        self.cycle = getattr(env, 'cycle_time', None)
        self.save_step = getattr(env, 'cycle_time', 1) / sim_step
        self.log_info = log_info
        self.log_info_interval = log_info_interval
        self.save_agent = save_agent

        logging.info(" Starting experiment {} at {}".format(
            env.network.name, str(datetime.datetime.utcnow())))

        logging.info("Initializing environment.")


    def run(
            self,
            num_steps,
            rl_actions=None,
            convert_to_csv=False
    ):
        """
        Run the given scenario for a set number of runs and steps per run.

        Parameters
        ----------
        num_steps : int
            number of steps to be performs in each run of the experiment
        rl_actions : method, optional
            maps states to actions to be performed by the RL agents (if
            there are any)
        convert_to_csv : bool
            Specifies whether to convert the emission file created by sumo
            into a csv file

        Returns
        -------
        info_dict : dict
            contains returns, average speed per step (last run)
        """

        # Raise an error if convert_to_csv is set to True but no emission
        # file will be generated, to avoid getting an error at the end of the
        # simulation.
        if convert_to_csv and self.env.sim_params.emission_path is None:
            raise ValueError(
                'The experiment was run with convert_to_csv set '
                'to True, but no emission file will be generated. If you wish '
                'to generate an emission file, you should set the parameter '
                'emission_path in the simulation parameters (SumoParams or '
                'AimsunParams) to the path of the folder where emissions '
                'output should be generated. If you do not wish to generate '
                'emissions, set the convert_to_csv parameter to False.')

        if rl_actions is None:

            def rl_actions(*_):
                return None

        info_dict = {}
        info_dict["id"] = self.env.network.name
        info_dict["cycle"] = self.cycle
        info_dict["save_step"] = self.save_step

        vels = []
        vehs = []
        observation_spaces = []
        actions = []
        rewards = []

        veh_i = []
        vel_i = []

        agent_updates_counter = 0

        state = self.env.reset()

        for j in tqdm(range(num_steps)):

            state, reward, done, _ = self.env.step(rl_actions(state))

            veh_i.append(len(self.env.k.vehicle.get_ids()))
            vel_i.append(
                np.nanmean(self.env.k.vehicle.get_speed(
                    self.env.k.vehicle.get_ids()
                    )
                )
            )

            if self._is_save_step():

                observation_spaces.append(
                    list(self.env.get_observation_space()))
                actions.append(
                    getattr(self.env, 'rl_action', None))
                rewards.append(round(reward, 4))

                vehs.append(np.nanmean(veh_i).round(4))
                vels.append(np.nanmean(vel_i).round(4))
                veh_i = []
                vel_i = []

                agent_updates_counter += 1

                # Save train log.
                if self.log_info and \
                    (agent_updates_counter % self.log_info_interval == 0):

                    filename = \
                        f"{self.dir_path}{self.env.network.name}.train.json"

                    info_dict["rewards"] = rewards
                    info_dict["velocities"] = vels
                    info_dict["vehicles"] = vehs
                    info_dict["observation_spaces"] = observation_spaces
                    info_dict["rl_actions"] = actions

                    with open(filename, 'w') as fj:
                        json.dump(info_dict, fj)

            if done:
                break

            if self.save_agent and self._is_save_q_table_step():
                n = int(j / self.save_step) + 1
                filename = \
                    f'{self.env.network.name}.Q.{n}-1.pickle' # TODO: this is a fix to not break models/evaluate.py

                self.env.dump(self.dir_path,
                                filename,
                                attr_name='Q')

        info_dict["rewards"] = rewards
        info_dict["velocities"] = vels
        info_dict["vehicles"] = vehs
        info_dict["observation_spaces"] = observation_spaces
        info_dict["rl_actions"] = actions

        self.env.terminate()

        print('Experiment:', f'{self.dir_path}{self.env.network.name}')

        # Convert xml to csv.
        if self.env.sim_params.emission_path:
            # wait a short period of time to ensure the xml file is readable
            time.sleep(0.1)

            if convert_to_csv:
                emission_filename = \
                    "{0}-emission.xml".format(self.env.network.name)

                emission_path = os.path.join(
                    self.env.sim_params.emission_path,self.dir_path,
                    emission_filename
                )

                emission_to_csv(emission_path)

        return info_dict

    def _is_save_step(self):
        if self.cycle is not None:
            return self.env.duration == 0.0
        return self.step_counter % self.save_step == 0

    def _is_save_q_table_step(self):
        if self.env.step_counter % (100 * self.save_step) == 0:
            return self.train and hasattr(self.env, 'dump') and self.dir_path
        return False
