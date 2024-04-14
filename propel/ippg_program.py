import utils
import numpy as np
from scipy import spatial
import logging
from bayes_opt import BayesianOptimization
import gymnasium as gym

ENV_NAME = "Pendulum-v1"
MAX_EPISODE_LEN = 200
ACTION_MIN = (-2,)
ACTION_MAX = (2,)
BEST_VAL_IND = 0
BEST_VAL_NAME = "X"

class ParameterFinder():
    def __init__(self, inputs, actions, action_prog):
        self.inputs = inputs
        self.actions = actions
        self.action = action_prog
    
    def find_distance_paras(self, ap0, ap1, ap2, apt, api, apc): # TODO: need 3 extra or 1 extra?
        self.action.update_parameters([ap0, ap1, ap2], apt, api, apc)
        action_acts = []
        for window_list in self.inputs:
            action_acts.append(utils.clip_to_range(self.action.pid_execute(window_list), ACTION_MIN[0], ACTION_MAX[1]))
        action_diff = spatial.distance.euclidean(action_acts, np.array(self.actions)[:, 0])
        diff_total = -action_diff / float(len(self.actions)) # TODO: is this correct?
        return diff_total

    def pid_parameters(self, info_list):
        gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 10} # Optimizer configuration
        logging.info("Optimizing controller")
        bo_pid = BayesianOptimization(self.find_distance_paras,
                                      {"ap0": info_list[0][0], "ap1": info_list[0][1], "ap2": info_list[0][2], "apt": info_list[0][3], "api": info_list[0][4], "apc": info_list[0][5]}, verbose=0)
        bo_pid.maximize(init_points=50, n_iter=100, kappa=5, **gp_params)
        return bo_pid.res["max"]

def programmatic_game(action):
    episode_count = 2
    max_steps = 2 * MAX_EPISODE_LEN
    window = 5

    # Generate an environment
    env = gym.make(ENV_NAME)

    logging.info(f"{ENV_NAME} experiment start with priors")
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
                print("Done")
                break
        else:
            raise AssertionError("\"max_steps\" has been reached.")
        
        logging.info(f"Total Reward {total_reward}, {BEST_VAL_NAME} {ob[BEST_VAL_IND]}, Last State {ob}")
        logging.info("")

        env.close() # This is for shutting down the environment
        logging.info("Finish")
