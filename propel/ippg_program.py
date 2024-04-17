import numpy as np
from scipy import spatial
import logging
from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction
import gymnasium as gym
import argparse
import random
import sys
import os

import utils
from controllers import Controller
from neural_update import NeuralAgent
from pendulum import PendulumThetaEnv

ENV_NAME = "Pendulum-v1"
MAX_EPISODE_LEN = 200
ACTION_MIN = (-2,)
ACTION_MAX = (2,)
BEST_VAL_IND = 0
BEST_VAL_NAME = "Theta"

# TODO: list
# DONE env render to understand env
# DONE change env to include theta
# DONE possibly add two more parameters (like in acceleration)
# DONE initialize parameters for sensible policy (test)
# DONE save and load model for quick testing
# DONE change initialized parameters
# DONE adjust bayesian optimization action ranges
# figure out lambda mixing stuff
# DONE check whether 1 run for collection is enough
# change logging to include bayesian optimization results and current neural_update iteration
# DONE test for multiple iterations

class ParameterFinder():
    def __init__(self, inputs, actions, action_prog):
        self.inputs = inputs
        self.actions = actions
        self.action = action_prog
    
    def find_distance_paras(self, ap0, ap1, ap2, apt, api, apc):
        self.action.update_parameters([ap0, ap1, ap2], apt, api, apc)
        action_acts = []
        for window_list in self.inputs:
            action_acts.append(utils.clip_to_range(self.action.pid_execute(window_list), ACTION_MIN[0], ACTION_MAX[0]))
        action_diff = spatial.distance.euclidean(action_acts, np.array(self.actions)[:, 0])
        diff_total = -action_diff / float(len(self.actions))
        return diff_total

    def pid_parameters(self, info_list):
        gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 10} # Optimizer configuration
        logging.info("Optimizing controller")
        bo_pid = BayesianOptimization(self.find_distance_paras,
                                      {"ap0": info_list[0][0][0], "ap1": info_list[0][0][1], "ap2": info_list[0][0][2], "apt": info_list[0][1], "api": info_list[0][2], "apc": info_list[0][3]},
                                      verbose=0, allow_duplicate_points=True)
        bo_pid.set_gp_params(**gp_params)
        bo_pid.maximize(init_points=50, n_iter=100, acquisition_function=UtilityFunction(kappa=5))
        return bo_pid.max

def programmatic_game(action):
    episode_count = 5
    max_steps = 2 * MAX_EPISODE_LEN
    window = 5

    # Generate an environment
    env = PendulumThetaEnv(gym.make(ENV_NAME))

    logging.info(f"{ENV_NAME} experiment start with priors")
    logging.info(f"Steering controller {action.pid_info()}")
    for _ in range(episode_count):
        ob, _ = env.reset()

        total_reward = 0.0
        temp_obs = [[ob[i]] for i in range(len(ob))] + [[0]]
        window_list = [temp_obs[:] for _ in range(window)]

        for _ in range(max_steps):
            action_action = utils.clip_to_range(action.pid_execute(window_list), ACTION_MIN[0], ACTION_MAX[0])
            action_prior = [action_action]

            temp_obs = [[ob[i]] for i in range(len(ob))] + [action_prior]
            window_list.pop(0)
            window_list.append(temp_obs[:])

            ob, r_t, term, trunc, _ = env.step(action_prior)
            done = term or trunc

            total_reward += r_t

            if done:
                break
        else:
            raise AssertionError("\"max_steps\" has been reached.")
        
        logging.info(f"Total Reward {total_reward}, {BEST_VAL_NAME} {ob[BEST_VAL_IND]}, Last State {ob}")
        logging.info("")

    env.close() # This is for shutting down the environment
    logging.info("Finish")
    logging.info("")

def learn_policy():
    # Define pi_0
    # action_prog = Controller([5, 0.5, 50], 0, 0, 0, 2 * np.pi, 0.0, "obs[-1][1][0] > self.para_condition") # TODO: change initial values?
    # action_prog = Controller([1, 0.05, 50], 0, 0, 0, 2 * np.pi, 0.0, "obs[-1][1][0] > self.para_condition")
    action_prog = Controller([0, 0.5, 50], 0, 0, 0, 2 * np.pi, 0.0, "obs[-1][1][0] > self.para_condition")

    programmatic_game(action_prog)

    nn_agent = NeuralAgent() # TODO: reset neural agent each step?
    all_observations = []
    all_actions = []
    # for i_iter in range(6):
    for i_iter in range(10): # TODO: delete?
        logging.info(f"Iteration {i_iter}")
        # nn_agent = NeuralAgent() #TODO: delete?
        # Learn/update neural policy
        if i_iter == 0:
            nn_agent.update_neural([action_prog], episode_count=200)
        else:
            nn_agent.update_neural([action_prog], episode_count=100)
        
        # Collect trajectories
        for _ in range(10):
            observation_list, _ = nn_agent.collect_data([action_prog])
            all_observations += observation_list
        # Relabel observations
        _, _, all_actions = nn_agent.label_data([action_prog], all_observations)

        # Learn new programmatic policy
        param_finder = ParameterFinder(all_observations, all_actions, action_prog)

        action_ranges = [tuple([utils.create_interval(action_prog.pid_info()[0][const], 0.2) for const in range(3)]), utils.create_interval(action_prog.pid_info()[1], 0.01), utils.create_interval(action_prog.pid_info()[2], 0.01), utils.create_interval(action_prog.pid_info()[3], 0.01)]
        # action_ranges = [tuple([utils.create_interval(action_prog.pid_info()[0][const], 1) for const in range(3)]), utils.create_interval(action_prog.pid_info()[1], 0.01), utils.create_interval(action_prog.pid_info()[2], 0.01), utils.create_interval(action_prog.pid_info()[3], 0.01)]
        # action_ranges = [tuple([utils.create_interval(action_prog.pid_info()[0][0], 1), utils.create_interval(action_prog.pid_info()[0][1], 0.2), utils.create_interval(action_prog.pid_info()[0][2], 0.2)]), utils.create_interval(action_prog.pid_info()[1], 0.01), utils.create_interval(action_prog.pid_info()[2], 0.01), utils.create_interval(action_prog.pid_info()[3], 0.01)]
        pid_ranges = [action_ranges]
        new_paras = param_finder.pid_parameters(pid_ranges)

        action_prog.update_parameters([new_paras["params"][i] for i in ["ap0", "ap1", "ap2"]], new_paras["params"]["apt"], new_paras["params"]["api"], new_paras["params"]["apc"])

        programmatic_game(action_prog)

    logging.info(f"Final steering controller {action_prog.pid_info()}")

    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=None)
    parser.add_argument("--logname", default="IPPGProgram")
    args = parser.parse_args()

    os.makedirs("run_ippg_program/", exist_ok=True)

    random.seed(args.seed)
    log_path = "run_ippg_program"
    log_filename = args.logname
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s] %(message)s",
        handlers=[
            logging.FileHandler("{0}/{1}.log".format(log_path, log_filename)),
            logging.StreamHandler(sys.stdout)
        ])
    logging.info("Logging started with level: INFO")
    learn_policy()
