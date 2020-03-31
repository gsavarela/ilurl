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

        >>> exp.run(num_runs=1, num_steps=1000)

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
                save_agent=False,
                save_agent_interval=100):
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
        save_agent_interval : int
            save RL agent interval (in number of agent-update steps)

        """
        sim_step = env.sim_params.sim_step
        # guarantees that the enviroment has stopped
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
        self.save_agent_interval = save_agent_interval

        logging.info(" Starting experiment {} at {}".format(
            env.network.name, str(datetime.datetime.utcnow())))

        logging.info("Initializing environment.")


    def run(
            self,
            num_steps,
            rl_actions=None,
            stop_on_teleports=False
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
        stop_on_teleport : boolean
            if true will break execution on teleport which occur on:
            * OSM scenarios with faulty connections
            * collisions
            * timeouts a vehicle is unable to move for 
            -time-to-teleport seconds (default 300) which is caused by
            wrong lane, yield or jam

        Returns
        -------
        info_dict : dict
            contains returns, average speed per step (last run)

        References
        ---------


        https://sourceforge.net/p/sumo/mailman/message/33244698/
        http://sumo.sourceforge.net/userdoc/Simulation/Output.html
        http://sumo.sourceforge.net/userdoc/Simulation/Why_Vehicles_are_teleporting.html
        """
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
                rewards.append(round(sum(reward), 4))

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
                    info_dict["rl_actions"] = list(self.env.actions_log.values())
                    info_dict["states"] = list(self.env.states_log.values())
                    info_dict["explored"] = getattr(self.env.agent, 'explored', None)
                    info_dict["visited_states"] = getattr(self.env.agent, 'visited_states', None)
                    info_dict["Q_distances"] = getattr(self.env.agent, 'Q_distances', None)

                    with open(filename, 'w') as fj:
                        json.dump(info_dict, fj)

            if done and stop_on_teleports:
                break

            if self.save_agent and self._is_save_q_table_step(agent_updates_counter):
                filename = \
                    f'{self.env.network.name}.Q.1-{agent_updates_counter}.pickle'

                self.env.dump(self.dir_path,
                                filename,
                                attr_name='Q')

        info_dict["rewards"] = rewards
        info_dict["velocities"] = vels
        info_dict["vehicles"] = vehs
        info_dict["observation_spaces"] = observation_spaces
        info_dict["rl_actions"] = list(self.env.actions_log.values())
        info_dict["states"] = list(self.env.states_log.values())
        info_dict["explored"] = getattr(self.env.agent, 'explored', None)
        info_dict["visited_states"] = getattr(self.env.agent, 'visited_states', None)
        info_dict["Q_distances"] = getattr(self.env.agent, 'Q_distances', None)

        self.env.terminate()

        return info_dict

    def _is_save_step(self):
        if self.cycle is not None:
            return self.env.duration == 0.0
        return self.step_counter % self.save_step == 0

    def _is_save_q_table_step(self, counter):
        if counter % self.save_agent_interval == 0:
            return self.train and hasattr(self.env, 'dump') and self.dir_path
        return False
